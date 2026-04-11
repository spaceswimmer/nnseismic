import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import SeismicDataset, create_dataloader, sort_list_IDs
from loss import SSIM3DLoss


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

    @staticmethod
    def _block(in_channels, features, name):
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
    ):
        self.model = model.to(device).float()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.data_augmentation = data_augmentation

        self.writer = SummaryWriter(log_dir=log_dir)

        self.criterion = SSIM3DLoss()
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

        for batch_idx, (seismic, rgt) in enumerate(self.train_loader):
            if self.data_augmentation:
                seismic = torch.cat(
                    [seismic, HorizontalFlip1(seismic)], dim=0
                )
                rgt = torch.cat(
                    [rgt, HorizontalFlip1(rgt)], dim=0
                )

            seismic, rgt = seismic.to(self.device), rgt.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(seismic)
            loss = self.criterion(outputs, rgt)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            batch_count += 1

            global_step = epoch * len(self.train_loader) + batch_idx
            self.writer.add_scalar("Train/SSIM_Loss", loss.item(), global_step)

            if batch_idx % 5 == 0:
                tqdm.write(
                    f"Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.6f}"
                )

        epoch_loss = running_loss / batch_count
        return epoch_loss

    def validate(self, epoch):
        self.model.eval()
        running_loss = 0.0
        batch_count = 0

        with torch.no_grad():
            for batch_idx, (seismic, rgt) in enumerate(self.val_loader):
                seismic, rgt = seismic.to(self.device), rgt.to(self.device)

                outputs = self.model(seismic)
                loss = self.criterion(outputs, rgt)

                running_loss += loss.item()
                batch_count += 1

                global_step = epoch * len(self.val_loader) + batch_idx
                self.writer.add_scalar("Validation/SSIM_Loss", loss.item(), global_step)

        epoch_loss = running_loss / batch_count
        return epoch_loss

    def train(self, num_epochs, save_dir="./checkpoints", checkpoint_interval=10):
        os.makedirs(save_dir, exist_ok=True)

        best_val_loss = float("inf")
        patience_counter = 0
        patience = 10

        for epoch in range(num_epochs):
            tqdm.write(f"\nEpoch {epoch + 1}/{num_epochs}")
            tqdm.write("-" * 30)

            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)

            if self.val_loader is not None:
                val_loss = self.validate(epoch)
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

                if patience_counter >= patience:
                    tqdm.write(
                        f"Early stopping triggered after {patience} epochs without improvement."
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
    save_dir="./checkpoints",
    log_dir="./logs",
    checkpoint_interval=10,
    pretrained_model=None,
    data_augmentation=True,
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

    trainer = SeismicTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        log_dir=log_dir,
        lr=lr,
        weight_decay=weight_decay,
        data_augmentation=data_augmentation,
    )

    train_losses, val_losses = trainer.train(
        num_epochs=num_epochs,
        save_dir=save_dir,
        checkpoint_interval=checkpoint_interval,
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
        "--pretrained_model", type=str, default=None, help="path to pretrained model"
    )
    parser.add_argument(
        "--data_augmentation",
        type=lambda x: x.lower() in ("yes", "true", "t", "y", "1"),
        default=True,
        help="use data augmentation (default: True)",
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
    )
