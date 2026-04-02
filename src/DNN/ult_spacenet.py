import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class Bottleneck3D(nn.Module):
    """3D ResNet bottleneck block."""
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(Bottleneck3D, self).__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(self.expansion * planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet50Encoder3D(nn.Module):
    """3D ResNet-50 encoder with skip connections."""
    
    def __init__(self, in_channels=1):
        super(ResNet50Encoder3D, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(Bottleneck3D, 64, 3, stride=1)
        self.layer2 = self._make_layer(Bottleneck3D, 128, 4, stride=2)
        self.layer3 = self._make_layer(Bottleneck3D, 256, 6, stride=2)
        self.layer4 = self._make_layer(Bottleneck3D, 512, 3, stride=2)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # 1/4 resolution

        x = self.layer1(x)
        skip1 = x  # 1/4 resolution (64 channels * 4 = 256)
        x = self.layer2(x)
        skip2 = x  # 1/8 resolution (128 * 4 = 512)
        x = self.layer3(x)
        skip3 = x  # 1/16 resolution (256 * 4 = 1024)
        x = self.layer4(x)
        skip4 = x  # 1/32 resolution (512 * 4 = 2048)
        
        return x, [skip1, skip2, skip3, skip4]

class UpProjectionBlock3D(nn.Module):
    """3D up-projection block with residual connections."""
    
    def __init__(self, in_channels, out_channels, skip_channels=None):
        super(UpProjectionBlock3D, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.use_skip = skip_channels is not None
        if self.use_skip:
            # Always use 1x1 conv to ensure channel and spatial dimension matching
            self.skip_conv = nn.Conv3d(skip_channels, out_channels, kernel_size=1, bias=False)
            self.skip_bn = nn.BatchNorm3d(out_channels)
        else:
            self.skip_conv = None
            self.skip_bn = None

        # Match dimensions for residual connection - transform upsampled input to output channels
        self.residual_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        self.residual_bn = nn.BatchNorm3d(out_channels)

    def forward(self, x, skip=None):
        x_up = self.up(x)  # First upsample
        
        # Apply convolutions to upsampled input
        x = self.conv1(x_up)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        
        # Create residual connection from upsampled input
        residual = self.residual_conv(x_up)
        residual = self.residual_bn(residual)
        
        # Add residual connection before final activation
        x = x + residual
        x = self.relu(x)

        if self.use_skip and skip is not None:
            skip = self.skip_conv(skip)
            skip = self.skip_bn(skip)
            x = x + skip
            
        return x

class RefinementModule3D(nn.Module):
    """Final refinement module."""
    
    def __init__(self, in_channels=64):
        super(RefinementModule3D, self).__init__()
        self.conv3 = nn.Conv3d(in_channels, in_channels, kernel_size=5, padding=2, bias=False)
        self.bn3 = nn.BatchNorm3d(in_channels)
        self.conv4 = nn.Conv3d(in_channels, in_channels, kernel_size=5, padding=2, bias=False)
        self.bn4 = nn.BatchNorm3d(in_channels)
        self.conv5 = nn.Conv3d(in_channels, 1, kernel_size=5, padding=2, bias=False)
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        
        x = self.conv5(x)
        return x

class RGTNetwork3D(nn.Module):
    """Full 3D RGT Network with ResNet-50 encoder and up-projection decoder."""
    
    def __init__(self, input_size=128):
        super(RGTNetwork3D, self).__init__()
        self.encoder = ResNet50Encoder3D(in_channels=1)
        
        self.conv2 = nn.Conv3d(2048, 2048, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm3d(2048)
        self.relu = nn.ReLU(inplace=True)

        self.up1 = UpProjectionBlock3D(2048, 1024, skip_channels=1024)
        self.up2 = UpProjectionBlock3D(1024, 512, skip_channels=512)
        self.up3 = UpProjectionBlock3D(512, 256, skip_channels=256)
        self.up4 = UpProjectionBlock3D(256, 64, skip_channels=64)  # Changed out_channels from 128 to 64
        self.up5 = UpProjectionBlock3D(64, 64, skip_channels=None)  # Changed in_channels from 128 to 64

        self.refinement = RefinementModule3D(in_channels=64)

    def forward(self, x):
        x_enc, skips = self.encoder(x)
        skip1, skip2, skip3, skip4 = skips

        x = self.conv2(x_enc)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.up1(x, skip=skip4)   # 1/16 -> 1/8
        x = self.up2(x, skip=skip3)   # 1/8 -> 1/4
        x = self.up3(x, skip=skip2)   # 1/4 -> 1/2
        x = self.up4(x, skip=skip1)   # 1/2 -> full

        out = self.refinement(x)
        return out

class SeismicDataset(Dataset):
    def __init__(self, seismic_data, rgt_data):
        self.seismic_data = [torch.tensor(arr).bfloat16() if not isinstance(arr, torch.Tensor) else arr.bfloat16() for arr in seismic_data]
        self.rgt_data = [torch.tensor(arr).bfloat16() if not isinstance(arr, torch.Tensor) else arr.bfloat16() for arr in rgt_data]

    def __len__(self):
        return len(self.seismic_data)

    def __getitem__(self, idx):
        seismic = self.seismic_data[idx].unsqueeze(0)
        rgt = self.rgt_data[idx].unsqueeze(0)
        return seismic, rgt

def create_data_loaders(dataset, batch_size=1, val_split=0.2, shuffle=True):
    indices = list(range(len(dataset)))
    train_indices, val_indices = train_test_split(
        indices, test_size=val_split, random_state=42
    )
    
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

class SeismicTrainer:
    """Trainer class for 3D seismic RGT network."""
    
    def __init__(
        self,
        model: nn.Module,
        train_dataset: Dataset,
        val_dataset: Dataset,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        device: str = None
    ):
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        
        self.train_loader = train_dataset
        self.val_loader = val_dataset
        
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=5,
            gamma=0.1
        )
        
        self.scaler = torch.amp.GradScaler(device=device)
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
        
        self.best_val_loss = float('inf')

    def train_epoch(self) -> float:
        """Train the model for one epoch."""
        self.model.train()
        running_loss = 0.0
        num_batches = len(self.train_loader)
        
        for inputs, targets in self.train_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            with torch.amp.autocast(self.device):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            running_loss += loss.item()
        
        avg_train_loss = running_loss / num_batches
        return avg_train_loss

    def validate(self) -> float:
        """Validate the model on the validation dataset."""
        self.model.eval()
        val_loss = 0.0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                
                val_loss += loss.item()
        
        avg_val_loss = val_loss / num_batches
        return avg_val_loss

    def save_checkpoint(self, filepath: str, epoch: int, is_best: bool = False):
        """Save a training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'train_loss': self.history['train_loss'],
            'val_loss': self.history['val_loss'],
            'best_val_loss': self.best_val_loss
        }
        
        torch.save(checkpoint, filepath)
        
        if is_best:
            best_filepath = filepath.replace('.pth', '_best.pth')
            torch.save(checkpoint, best_filepath)

    def load_checkpoint(self, filepath: str):
        """Load a training checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.history['train_loss'] = checkpoint.get('train_loss', [])
        self.history['val_loss'] = checkpoint.get('val_loss', [])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        return checkpoint['epoch']

    def train(
        self,
        num_epochs: int,
        checkpoint_dir: str = './checkpoints',
        save_every: int = 5,
        verbose: bool = True
    ):
        """Train the model for multiple epochs."""
        import os
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate()
            
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rate'].append(current_lr)
            
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            if verbose:
                print(f'Epoch [{epoch + 1}/{num_epochs}]')
                print(f'  Train Loss: {train_loss:.6f}')
                print(f'  Val Loss:   {val_loss:.6f}')
                print(f'  LR:         {current_lr:.2e}')
                if is_best:
                    print(f'  *** New best model! ***')
            
            if (epoch + 1) % save_every == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch + 1}.pth')
                self.save_checkpoint(checkpoint_path, epoch + 1, is_best=is_best)
            
            if is_best:
                best_path = os.path.join(checkpoint_dir, 'checkpoint_best.pth')
                self.save_checkpoint(best_path, epoch + 1, is_best=True)
        
        final_path = os.path.join(checkpoint_dir, 'checkpoint_final.pth')
        self.save_checkpoint(final_path, num_epochs)
        
        if verbose:
            print(f'\nTraining completed!')
            print(f'Best validation loss: {self.best_val_loss:.6f}')
