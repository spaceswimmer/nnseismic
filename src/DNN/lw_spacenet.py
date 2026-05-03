import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import os
import json
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
import random

from DNN.dataset import SeismicDataset, create_dataloader, sort_list_IDs
from DNN.loss import SSIM3DLoss
from util.metrics import mae, rmse, mrpd


def HorizontalFlip1(dat):
    return torch.flip(dat, dims=[3])


def HorizontalFlip2(dat):
    return torch.flip(dat, dims=[4])


def VerticalFlip(dat):
    return torch.flip(dat, dims=[2])


def VerticalFlip_reverse(dat):
    return -torch.flip(dat, dims=[2])


class UNet3D(nn.Module):
    def __init__(
        self, in_channels=1, out_channels=1, init_features=16, smoothing_kernel_size=5
    ):
        super(UNet3D, self).__init__()

        features = init_features
        self.encoder1 = UNet3D._block(in_channels, features, name="enc1")
        self.pool1 = nn.AvgPool3d(kernel_size=2, stride=2)
        self.encoder2 = UNet3D._block(features, features * 2, name="enc2")
        self.pool2 = nn.AvgPool3d(kernel_size=2, stride=2)
        self.encoder3 = UNet3D._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.AvgPool3d(kernel_size=2, stride=2)
        self.encoder4 = UNet3D._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.AvgPool3d(kernel_size=2, stride=2)

        self.bottleneck = UNet3D._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
            nn.Conv3d(
                features * 16, features * 8, kernel_size=3, padding=1, bias=False
            ),
            nn.GroupNorm(num_groups=8, num_channels=features * 8),
            nn.ReLU(inplace=True),
        )
        self.decoder4 = UNet3D._block((features * 8) * 2, features * 8, name="dec4")

        self.upconv3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
            nn.Conv3d(features * 8, features * 4, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=features * 4),
            nn.ReLU(inplace=True),
        )
        self.decoder3 = UNet3D._block((features * 4) * 2, features * 4, name="dec3")

        self.upconv2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
            nn.Conv3d(features * 4, features * 2, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=features * 2),
            nn.ReLU(inplace=True),
        )
        self.decoder2 = UNet3D._block((features * 2) * 2, features * 2, name="dec2")

        self.upconv1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
            nn.Conv3d(features * 2, features, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=features),
            nn.ReLU(inplace=True),
        )
        self.decoder1 = UNet3D._block(features * 2, features, name="dec1")

        self.outconv = nn.Conv3d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

        self.smooth_conv = nn.Sequential(
            nn.ReplicationPad3d(smoothing_kernel_size // 2),
            nn.Conv3d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=smoothing_kernel_size,
                padding=0,
                bias=False,
            ),
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu", a=0.1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.GroupNorm, nn.BatchNorm3d)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

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

        out = self.outconv(dec1)
        out_smth = self.smooth_conv(out)
        return out_smth

    class ResBlock(nn.Module):
        def __init__(self, in_channels, features):
            super(UNet3D.ResBlock, self).__init__()
            self.conv1 = nn.Conv3d(
                in_channels=in_channels,
                out_channels=features,
                kernel_size=3,
                padding=1,
                bias=False,
            )
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=features)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv3d(
                in_channels=features,
                out_channels=features,
                kernel_size=3,
                padding=1,
                bias=False,
            )
            self.norm2 = nn.GroupNorm(num_groups=8, num_channels=features)

            if in_channels != features:
                self.downsample = nn.Sequential(
                    nn.Conv3d(in_channels, features, kernel_size=1, bias=False),
                    nn.GroupNorm(num_groups=8, num_channels=features),
                )
            else:
                self.downsample = None

        def forward(self, x):
            residual = x
            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)
            out = self.conv2(out)
            out = self.norm2(out)
            if self.downsample is not None:
                residual = self.downsample(x)
            out += residual
            out = self.relu(out)
            return out

    @staticmethod
    def _block(in_channels, features, name):
        return UNet3D.ResBlock(in_channels, features)


class SeismicTrainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device="cuda",
        log_dir="./logs",
        lr=1e-4,
        weight_decay=1e-4,
        data_augmentation=True,
        grad_clip=1.0,
        ssim_max_val=1.0,
        checkpoint_interval=10,
        pictures_dir=None,
        patience=10,
        accumulation_steps=1,
    ):
        self.model = model.to(device).bfloat16()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.data_augmentation = data_augmentation
        self.grad_clip = grad_clip
        self.checkpoint_interval = checkpoint_interval
        self.pictures_dir = pictures_dir
        self.patience = patience
        self.accumulation_steps = accumulation_steps

        self.writer = SummaryWriter(log_dir=log_dir)

        self.criterion = SSIM3DLoss(max_val=ssim_max_val)
        self.optimizer = optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

        self.scheduler = lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", patience=5, factor=0.5
        )

        self.train_losses = []
        self.val_losses = []

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        batch_count = 0
        
        self.optimizer.zero_grad()
        
        target_batch_size = self.train_loader.batch_size * self.accumulation_steps
        
        for batch_idx, (seismic, rgt) in enumerate(self.train_loader):
            if self.data_augmentation:
                seismic = torch.cat([seismic, HorizontalFlip1(seismic), HorizontalFlip2(seismic), VerticalFlip(seismic)], dim=0)
                rgt = torch.cat([rgt, HorizontalFlip1(rgt), HorizontalFlip2(rgt), VerticalFlip_reverse(rgt)], dim=0)
                
                noise_std = 0.01 * seismic.std()
                noise = torch.randn_like(seismic) * noise_std
                seismic = seismic + noise
            
            actual_batch_size = seismic.shape[0]
            seismic, rgt = seismic.to(self.device), rgt.to(self.device)
            
            outputs = self.model(seismic)
            loss = self.criterion(outputs, rgt)
            
            original_loss_value = loss.item()
            
            loss = loss * actual_batch_size / target_batch_size
            
            loss.backward()
            
            running_loss += original_loss_value
            batch_count += 1
            
            if (batch_idx + 1) % self.accumulation_steps == 0 or batch_idx == len(self.train_loader) - 1:
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            global_step = epoch * len(self.train_loader) + batch_idx
            self.writer.add_scalar("Train/SSIM_Loss", original_loss_value, global_step)
            
            if batch_idx % 5 == 0:
                tqdm.write(
                    f"Batch {batch_idx}/{len(self.train_loader)}, Loss: {original_loss_value:.6f}"
                )
        
        epoch_loss = running_loss / batch_count
        return epoch_loss

    def validate(self, epoch, save_plots=False):
        self.model.eval()
        running_loss = 0.0
        batch_count = 0
        metrics_sum = {"mae": 0.0, "mse": 0.0, "rmse": 0.0, "mrpd": 0.0}

        if save_plots and self.pictures_dir:
            epoch_pictures_dir = os.path.join(self.pictures_dir, f"epoch_{epoch:04d}")
            os.makedirs(epoch_pictures_dir, exist_ok=True)

        with torch.no_grad():
            for batch_idx, (seismic, rgt) in enumerate(self.val_loader):
                seismic, rgt = seismic.to(self.device), rgt.to(self.device)

                outputs = self.model(seismic)
                loss = self.criterion(outputs, rgt)

                running_loss += loss.item()
                batch_count += 1

                pred_np = outputs.squeeze().cpu().float().numpy()
                target_np = rgt.squeeze().cpu().float().numpy()
                
                metrics_sum["mae"] += mae(pred_np, target_np)
                metrics_sum["mse"] += np.mean((pred_np - target_np) ** 2)
                metrics_sum["rmse"] += rmse(pred_np, target_np)
                metrics_sum["mrpd"] += mrpd(pred_np, target_np)

                global_step = epoch * len(self.val_loader) + batch_idx
                self.writer.add_scalar("Validation/SSIM_Loss", loss.item(), global_step)

                if save_plots and self.pictures_dir:
                    self._save_prediction_plot(seismic, outputs, rgt, epoch_pictures_dir, batch_idx)

        epoch_loss = running_loss / batch_count
        avg_metrics = {k: v / batch_count for k, v in metrics_sum.items()}
        
        self.writer.add_scalar("Validation/MAE", avg_metrics["mae"], epoch)
        self.writer.add_scalar("Validation/MSE", avg_metrics["mse"], epoch)
        self.writer.add_scalar("Validation/RMSE", avg_metrics["rmse"], epoch)
        self.writer.add_scalar("Validation/MRPD", avg_metrics["mrpd"], epoch)
        
        tqdm.write(f"Val Metrics - MAE: {avg_metrics['mae']:.4f}, RMSE: {avg_metrics['rmse']:.4f}, MRPD: {avg_metrics['mrpd']:.4f}")
        
        return epoch_loss

    def _save_prediction_plot(self, seismic, pred_rgt, target_rgt, save_dir, batch_idx):
        seis_np = seismic[0, 0].cpu().float().numpy()
        pred_np = pred_rgt[0, 0].cpu().float().numpy()
        target_np = target_rgt[0, 0].cpu().float().numpy()

        sl = (slice(None), 96, slice(None))

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        
        axs[0].imshow(seis_np[sl].T, cmap='gray', interpolation='nearest')
        axs[0].contour(pred_np[sl].T, np.linspace(np.min(pred_np), np.max(pred_np), 20), 
                       colors='black', linewidths=2)
        axs[0].set_title('Seismic chunk (Predicted RGT contours)')

        axs[1].imshow(pred_np[sl].T, cmap='prism', interpolation='nearest')
        axs[1].contour(pred_np[sl].T, np.linspace(np.min(pred_np), np.max(pred_np), 20), 
                       colors='black', linewidths=2)
        axs[1].set_title('Predicted RGT')

        plt.tight_layout()
        
        save_path = os.path.join(save_dir, f"batch_{batch_idx:04d}.png")
        fig.savefig(save_path)
        plt.close(fig)

    def train(
        self,
        num_epochs,
        save_dir="./checkpoints",
        checkpoint_interval=10,
        start_epoch=0,
        best_val_loss=float("inf"),
    ):
        os.makedirs(save_dir, exist_ok=True)

        patience_counter = 0

        for epoch in range(start_epoch, num_epochs):
            tqdm.write(f"\nEpoch {epoch + 1}/{num_epochs}")
            tqdm.write("-" * 30)

            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)

            if self.val_loader is not None:
                save_plots = self.checkpoint_interval > 0 and epoch % self.checkpoint_interval == 0
                val_loss = self.validate(epoch, save_plots=save_plots)
                self.val_losses.append(val_loss)

                self.scheduler.step(val_loss)

                self.writer.add_scalar("Train/Loss", train_loss, epoch)
                self.writer.add_scalar("Validation/Loss", val_loss, epoch)
                self.writer.add_scalar(
                    "Learning_Rate", self.optimizer.param_groups[0]["lr"], epoch
                )

                tqdm.write(f"Train Loss: {train_loss:.6f}")
                tqdm.write(f"Val Loss: {val_loss:.6f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "train_loss": train_loss,
                            "val_loss": val_loss,
                        },
                        os.path.join(save_dir, "best_model.pth"),
                    )
                    tqdm.write(f"Saved best model with val_loss: {val_loss:.6f}")
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= self.patience:
                    tqdm.write(
                        f"Early stopping triggered after {self.patience} epochs without improvement."
                    )
                    break

            if epoch % checkpoint_interval == 0:
                save_name = os.path.join(save_dir, f"checkpoint_{epoch}.pth")
                torch.save(self.model.state_dict(), save_name)

        self.writer.close()

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss if self.val_loader else None,
            },
            os.path.join(save_dir, "final_model.pth"),
        )

        tqdm.write(f"Best validation loss: {best_val_loss:.6f}")

        return self.train_losses, self.val_losses

    def load_model(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        tqdm.write(f"Loaded model from {checkpoint_path}")

    def resume(self, checkpoint_path, num_epochs, new_weight_decay=None):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if new_weight_decay is not None:
            for param_group in self.optimizer.param_groups:
                param_group["weight_decay"] = new_weight_decay
            tqdm.write(f"Updated weight_decay to {new_weight_decay}")

        start_epoch = checkpoint.get("epoch", 0) + 1
        best_val_loss = checkpoint.get("val_loss", float("inf"))

        if best_val_loss is None:
            best_val_loss = float("inf")

        tqdm.write(
            f"Resuming from epoch {start_epoch}, previous val_loss: {best_val_loss:.6f}"
        )

        return start_epoch, best_val_loss


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_model(
    dataroot,
    dataroot_val=None,
    shape=(128, 128, 128, 1),
    batch_size=1,
    num_epochs=100,
    lr=1e-4,
    weight_decay=1e-4,
    dataset_size=float("inf"),
    dataset_size_val=float("inf"),
    num_workers=0,
    save_dir="./output",
    log_dir="./logs",
    checkpoint_interval=10,
    pretrained_model=None,
    data_augmentation=False,
    resume=None,
    name="experiment",
    grad_clip=1.0,
    ssim_max_val=1.0,
    patience=10,
    accumulation_steps=1,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tqdm.write(f"Using device: {device}")

    n1, n2, n3, n_channels = shape
    shape_with_channels = (n1, n2, n3, n_channels)

    train_loader = create_dataloader(
        dataroot=dataroot,
        shape=shape_with_channels,
        batch_size=batch_size,
        dataset_size=dataset_size,
        num_workers=num_workers,
        shuffle=True,
    )

    val_loader = None
    if dataroot_val is not None:
        val_loader = create_dataloader(
            dataroot=dataroot_val,
            shape=shape_with_channels,
            batch_size=1,
            dataset_size=dataset_size_val,
            num_workers=num_workers,
            shuffle=False,
        )

    model = UNet3D(in_channels=n_channels, out_channels=1, init_features=16)

    if pretrained_model is not None:
        model.load_state_dict(torch.load(pretrained_model, map_location=device))
        tqdm.write(f"Loaded pretrained model from {pretrained_model}")

    num_GPU = torch.cuda.device_count()
    if num_GPU > 1:
        model = nn.DataParallel(model, device_ids=list(range(num_GPU)))
    model = model.to(device)

    tqdm.write(f"Model parameters: {count_parameters(model):,}")

    full_save_dir = os.path.join(save_dir, name)
    full_log_dir = os.path.join(log_dir, name)
    pictures_dir = os.path.join(full_save_dir, "pictures")
    
    os.makedirs(full_save_dir, exist_ok=True)
    os.makedirs(pictures_dir, exist_ok=True)

    training_params = {
        "dataroot": dataroot,
        "dataroot_val": dataroot_val,
        "shape": shape,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "lr": lr,
        "weight_decay": weight_decay,
        "dataset_size": dataset_size,
        "dataset_size_val": dataset_size_val,
        "num_workers": num_workers,
        "save_dir": save_dir,
        "log_dir": log_dir,
        "checkpoint_interval": checkpoint_interval,
        "pretrained_model": pretrained_model,
        "data_augmentation": data_augmentation,
        "resume": resume,
        "name": name,
        "grad_clip": grad_clip,
        "ssim_max_val": ssim_max_val,
        "patience": patience,
        "accumulation_steps": accumulation_steps,
    }
    params_path = os.path.join(full_save_dir, "parameters.json")
    with open(params_path, "w") as f:
        json.dump(training_params, f, indent=4)
    tqdm.write(f"Saved training parameters to {params_path}")

    trainer = SeismicTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        log_dir=full_log_dir,
        lr=lr,
        weight_decay=weight_decay,
        data_augmentation=data_augmentation,
        grad_clip=grad_clip,
        ssim_max_val=ssim_max_val,
        checkpoint_interval=checkpoint_interval,
        pictures_dir=pictures_dir,
        patience=patience,
        accumulation_steps=accumulation_steps,
    )

    start_epoch = 0
    best_val_loss = float("inf")

    if resume is not None:
        start_epoch, best_val_loss = trainer.resume(resume, num_epochs, new_weight_decay=weight_decay)

    train_losses, val_losses = trainer.train(
        num_epochs=num_epochs,
        save_dir=full_save_dir,
        checkpoint_interval=checkpoint_interval,
        start_epoch=start_epoch,
        best_val_loss=best_val_loss,
    )

    return train_losses, val_losses, model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train UNet3D for RGT prediction")
    parser.add_argument(
        "--dataroot",
        type=str,
        required=True,
        help="path to training data (should have subfolders seis, rgt)",
    )
    parser.add_argument(
        "--dataroot_val", type=str, default=None, help="path to validation data"
    )
    parser.add_argument(
        "--shape",
        type=int,
        nargs=4,
        default=[128, 128, 128, 1],
        help="shape of input (n1, n2, n3, n_channels)",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="input batch size")
    parser.add_argument(
        "--nepochs", type=int, default=100, help="number of epochs for training"
    )
    parser.add_argument(
        "--lr", type=float, default=0.0001, help="initial learning rate for adam"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-4, help="weight decay rate for adam"
    )
    parser.add_argument(
        "--dataset_size",
        type=int,
        default=float("inf"),
        help="size of training dataset",
    )
    parser.add_argument(
        "--dataset_size_val",
        type=int,
        default=float("inf"),
        help="size of validation dataset",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="number of CPU workers for data loading",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./checkpoints",
        help="directory to save checkpoints",
    )
    parser.add_argument(
        "--log_dir", type=str, default="./logs", help="directory for tensorboard logs"
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=10,
        help="interval between saving checkpoints",
    )
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default=None,
        help="path to pretrained model weights",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="path to checkpoint to resume training from (resumes epoch, optimizer, scheduler)",
    )
    parser.add_argument(
        "--data_augmentation",
        type=lambda x: x.lower() in ("yes", "true", "t", "y", "1"),
        default=True,
        help="use data augmentation (default: True)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="experiment",
        help="experiment name for logs and checkpoints",
    )
    parser.add_argument(
        "--grad_clip",
        type=float,
        default=1.0,
        help="gradient clipping max norm (0 to disable, default: 1.0)",
    )
    parser.add_argument(
        "--ssim_max_val",
        type=float,
        default=1.0,
        help="max value for SSIM loss normalization (default: 1.0)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="early stopping patience (epochs without improvement, default: 10)",
    )
    parser.add_argument(
        "--accumulation_steps",
        type=int,
        default=1,
        help="gradient accumulation steps for effective batch size (default: 1, no accumulation)",
    )

    opt = parser.parse_args()

    train_model(
        dataroot=opt.dataroot,
        dataroot_val=opt.dataroot_val,
        shape=tuple(opt.shape),
        batch_size=opt.batch_size,
        num_epochs=opt.nepochs,
        lr=opt.lr,
        weight_decay=opt.weight_decay,
        dataset_size=opt.dataset_size,
        dataset_size_val=opt.dataset_size_val,
        num_workers=opt.num_workers,
        save_dir=opt.save_dir,
        log_dir=opt.log_dir,
        checkpoint_interval=opt.checkpoint_interval,
        pretrained_model=opt.pretrained_model,
        data_augmentation=opt.data_augmentation,
        resume=opt.resume,
        name=opt.name,
        grad_clip=opt.grad_clip,
        ssim_max_val=opt.ssim_max_val,
        patience=opt.patience,
        accumulation_steps=opt.accumulation_steps,
    )
