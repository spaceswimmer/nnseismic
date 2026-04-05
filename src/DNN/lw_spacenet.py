import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter  # Added import

class SeismicDataset(Dataset):
    """
    Dataset class for seismic data and corresponding relative geological time maps.
    
    Args:
        seismic_data (list or array): List of seismic data arrays (numpy arrays of size 128x128x128)
        rgt_data (list or array): List of relative geological time arrays (same dimensions as seismic)
    """
    def __init__(self, seismic_data, rgt_data, augm=True):
        if augm:
            seismic_data, rgt_data = self._augment_data(seismic_data, rgt_data)
        self.seismic_data = [torch.tensor(arr).bfloat16() if not isinstance(arr, torch.Tensor) else arr.bfloat16() for arr in seismic_data]
        self.rgt_data = [torch.tensor(arr).bfloat16() if not isinstance(arr, torch.Tensor) else arr.bfloat16() for arr in rgt_data]

    def __len__(self):
        return len(self.seismic_data)

    def __getitem__(self, idx):
        seismic = self.seismic_data[idx].unsqueeze(0)  # Add channel dimension
        rgt = self.rgt_data[idx].unsqueeze(0)  # Add channel dimension
        return seismic, rgt
    
    @staticmethod
    def _flip_axis(data, axis, reverse_sign=False):
        """Flip data along specified axis. Optionally reverse sign."""
        flipped = np.flip(data, axis=axis)
        if reverse_sign:
            return -flipped
        return flipped
    
    def _augment_data(self, seismic_data, rgt_data):
        augmented_seismic = []
        augmented_rgt = []
        
        for seismic, rgt in zip(seismic_data, rgt_data):
            # Each seismic/rgt is shape (128, 128, 128) - axes 0, 1, 2 only
            
            # 1. Original
            augmented_seismic.append(seismic)
            augmented_rgt.append(rgt)
            
            # 2. X-flip (axis=0)
            augmented_seismic.append(self._flip_axis(seismic, axis=0))
            augmented_rgt.append(self._flip_axis(rgt, axis=0))
            
            # 3. Y-flip (axis=1)
            augmented_seismic.append(self._flip_axis(seismic, axis=1))
            augmented_rgt.append(self._flip_axis(rgt, axis=1))
            
            # 4. Z-flip (axis=2, RGT sign reversed)
            augmented_seismic.append(self._flip_axis(seismic, axis=2))
            augmented_rgt.append(self._flip_axis(rgt, axis=2, reverse_sign=True))
        
        return np.stack(augmented_seismic), np.stack(augmented_rgt)


class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=16,
                 smoothing_kernel_size=5):
        super(UNet3D, self).__init__()

        features = init_features
        # Encoder (downsampling path)
        self.encoder1 = UNet3D._block(in_channels, features, name="enc1")
        self.pool1 = nn.AvgPool3d(kernel_size=2, stride=2)
        self.encoder2 = UNet3D._block(features, features*2, name="enc2")
        self.pool2 = nn.AvgPool3d(kernel_size=2, stride=2)
        self.encoder3 = UNet3D._block(features*2, features*4, name="enc3")
        self.pool3 = nn.AvgPool3d(kernel_size=2, stride=2)
        self.encoder4 = UNet3D._block(features*4, features*8, name="enc4")
        self.pool4 = nn.AvgPool3d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = UNet3D._block(features*8, features*16, name="bottleneck")

        # Decoder (upsampling path) - Replaced ConvTranspose3d with Upsample + Conv3d
        self.upconv4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
            nn.Conv3d(features*16, features*8, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=features*8),
            nn.ReLU(inplace=True),
        )
        self.decoder4 = UNet3D._block((features*8)*2, features*8, name="dec4")
        
        self.upconv3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
            nn.Conv3d(features*8, features*4, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=features*4),
            nn.ReLU(inplace=True),
        )
        self.decoder3 = UNet3D._block((features*4)*2, features*4, name="dec3")
        
        self.upconv2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
            nn.Conv3d(features*4, features*2, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=features*2),
            nn.ReLU(inplace=True),
        )
        self.decoder2 = UNet3D._block((features*2)*2, features*2, name="dec2")
        
        self.upconv1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
            nn.Conv3d(features*2, features, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=features),
            nn.ReLU(inplace=True),
        )
        self.decoder1 = UNet3D._block(features*2, features, name="dec1")
        # Output layer
        self.outconv = nn.Conv3d(in_channels=features, out_channels=out_channels, kernel_size=1)
        # Smothing layer
        self.smooth_conv = nn.Sequential(
            nn.ReplicationPad3d(smoothing_kernel_size // 2),
            nn.Conv3d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=smoothing_kernel_size,
                padding=0,  # Same padding
                bias=False,
            ),
        )

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))

        # Decoder
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        # Output
        out = self.outconv(dec1)
        # Smothing
        out_smth = self.smooth_conv(out)
        return out_smth

    @staticmethod
    def _block(in_channels, features, name):
        """Basic convolutional block: Conv3d -> GroupNorm -> ReLU"""
        return nn.Sequential(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=features,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(num_groups=8, num_channels=features),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                in_channels=features,
                out_channels=features,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(num_groups=8, num_channels=features),
            nn.ReLU(inplace=True),
        )
    

class SeismicTrainer:
    def __init__(self, model, train_loader, val_loader, device='cuda',
                 log_dir='./logs', tv_loss_weight=0.01):
        # Convert model to bfloat16
        self.model = model.to(device).bfloat16()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Initialize TensorBoard writer
        self.writer = SummaryWriter(log_dir=log_dir)
        
        self.criterion = nn.MSELoss()
        self.tv_loss_weight = tv_loss_weight
        # Convert optimizer to work with bfloat16 model
        self.optimizer = optim.Adam(model.parameters(), lr=1e-4)
        
        # Scheduler for learning rate adjustment
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5, verbose=True
        )
        
        # Track training history
        self.train_losses = []
        self.val_losses = []
    
    def total_variation_loss(self, outputs, epsilon=1e-8):
        """
        Compute 3D total variation loss to encourage smooth outputs.
        Computes TV separately for each dimension and sums them.
        """
        # TV along X dimension (axis 2)
        tv_x = torch.mean(torch.abs(outputs[:, :, :-1, :, :] - outputs[:, :, 1:, :, :]))
        
        # TV along Y dimension (axis 3)
        tv_y = torch.mean(torch.abs(outputs[:, :, :, :-1, :] - outputs[:, :, :, 1:, :]))
        
        # TV along Z dimension (axis 4)
        tv_z = torch.mean(torch.abs(outputs[:, :, :, :, :-1] - outputs[:, :, :, :, 1:]))
        
        return tv_x + tv_y + tv_z

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        batch_count = 0
        
        for batch_idx, (seismic, rgt) in enumerate(self.train_loader):
            # Ensure data is moved to device with correct dtype
            seismic, rgt = seismic.to(self.device), rgt.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(seismic)
            mse_loss = self.criterion(outputs, rgt)
            
            # Add total variation regularization (NEW)
            tv_loss = self.total_variation_loss(outputs)
            loss = mse_loss + self.tv_loss_weight * tv_loss
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            batch_count += 1
            
            # Log training loss for each batch
            global_step = epoch * len(self.train_loader) + batch_idx
            # Log both losses (NEW)
            self.writer.add_scalar('Train/MSE_Loss', mse_loss.item(), global_step)
            if self.tv_loss_weight != 0:
                self.writer.add_scalar('Train/TV_Loss', tv_loss.item(), global_step)
            
            # Print progress every 10 batches
            if batch_idx % 5 == 0:
                print(f'Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.6f}')
        
        epoch_loss = running_loss / batch_count
        return epoch_loss

    def validate(self, epoch):
        self.model.eval()
        running_total_loss = 0.0
        running_mse_loss = 0.0
        running_tv_loss = 0.0
        batch_count = 0
        
        with torch.no_grad():
            for batch_idx, (seismic, rgt) in enumerate(self.val_loader):
                # Ensure data is moved to device with correct dtype
                seismic, rgt = seismic.to(self.device), rgt.to(self.device)
                
                outputs = self.model(seismic)
                mse_loss = self.criterion(outputs, rgt)
                tv_loss = self.total_variation_loss(outputs)
                total_loss = mse_loss + self.tv_loss_weight * tv_loss
                
                running_total_loss += total_loss.item()
                running_mse_loss += mse_loss.item()
                running_tv_loss += tv_loss.item()
                batch_count += 1
                
                # Log validation loss for each batch
                global_step = epoch * len(self.val_loader) + batch_idx
                self.writer.add_scalar('Validation/Total_Loss', total_loss.item(), global_step)
                if self.tv_loss_weight != 0:
                    self.writer.add_scalar('Validation/MSE_Loss', mse_loss.item(), global_step)
                    self.writer.add_scalar('Validation/TV_Loss', tv_loss.item(), global_step)
        
        epoch_total_loss = running_total_loss / batch_count
        return epoch_total_loss

    def train(self, num_epochs, save_dir='./checkpoints'):
        os.makedirs(save_dir, exist_ok=True)
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 10  # Number of epochs to wait before early stopping
        
        for epoch in range(num_epochs):
            print(f'\nEpoch {epoch+1}/{num_epochs}')
            print('-' * 30)
            
            # Train
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate(epoch)
            self.val_losses.append(val_loss)
            
            # Update learning rate based on validation loss
            self.scheduler.step(val_loss)
            
            # Log epoch losses to TensorBoard
            self.writer.add_scalar('Train/Loss', train_loss, epoch)
            self.writer.add_scalar('Validation/Loss', val_loss, epoch)
            self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            print(f'Train Loss: {train_loss:.6f}')
            print(f'Val Loss: {val_loss:.6f}')
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, os.path.join(save_dir, 'best_model.pth'))
                print(f'Saved best model with val_loss: {val_loss:.6f}')
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print(f'Early stopping triggered after {patience} epochs without improvement.')
                break
        
        # Close TensorBoard writer
        self.writer.close()
        
        # Save final model
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, os.path.join(save_dir, 'final_model.pth'))
        
        print(f'Best validation loss: {best_val_loss:.6f}')
        
        return self.train_losses, self.val_losses

    def load_best_model(self, checkpoint_path):
        """Load the best saved model."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Loaded model from {checkpoint_path}")

def create_data_loaders(dataset, batch_size=1, val_split=0.2, shuffle=True):
    """
    Create train and validation data loaders from a dataset.
    
    Args:
        dataset: Dataset object containing seismic and RGT data
        batch_size: Size of batches for training
        val_split: Fraction of data to use for validation
        shuffle: Whether to shuffle the data
    
    Returns:
        train_loader, val_loader: PyTorch DataLoader objects
    """
    # Split dataset into train and validation
    indices = list(range(len(dataset)))
    train_indices, val_indices = train_test_split(
        indices, test_size=val_split, random_state=42
    )
    
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # Test the model with sample input
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet3D(in_channels=1, out_channels=1, init_features=16)
    # Convert model to bfloat16 before moving to device
    model = model.bfloat16().to(device)
    
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Test with a sample input (batch_size=1, channels=1, depth=128, height=128, width=128)
    x = torch.randn(1, 1, 128, 128, 128).bfloat16().to(device)
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")