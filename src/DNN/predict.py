import torch
import numpy as np
import argparse
from .lw_spacenet import UNet3D


def predict_single_chunk(
    seis_path, checkpoint_path, shape=(128, 128, 128), device="cuda"
):
    model = UNet3D(in_channels=1, out_channels=1, init_features=16)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device).float()
    model.eval()

    seis = np.fromfile(seis_path, dtype=np.float32)
    seis = seis.reshape(shape)
    seis = (seis - np.mean(seis)) / np.std(seis)

    seis_tensor = torch.from_numpy(seis).float().unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(seis_tensor)

    pred = pred.squeeze().cpu().numpy()
    return pred


def predict_large_volume(
    seis_path, checkpoint_path, shape, chunk_size=(128, 128, 128), device="cuda"
):
    from large_volume_predictor import LargeVolumePredictor

    predictor = LargeVolumePredictor(
        model_path=checkpoint_path, device=device, chunk_size=chunk_size
    )

    seismic = np.fromfile(seis_path, dtype=np.float32).reshape(shape)
    pred = predictor.predict_full_volume(seismic)
    return pred


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict RGT from seismic data")
    parser.add_argument(
        "--seis_path", type=str, required=True, help="Path to seismic binary file"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--output", type=str, default="pred.bin", help="Output path for prediction"
    )
    parser.add_argument(
        "--shape",
        type=int,
        nargs=3,
        required=True,
        help="Shape of input volume (H W D)",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        nargs=3,
        default=[128, 128, 128],
        help="Chunk size for large volume prediction",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use (cuda/cpu)"
    )
    parser.add_argument(
        "--large", action="store_true", help="Use large volume predictor with chunking"
    )

    args = parser.parse_args()

    shape = tuple(args.shape)

    if args.large:
        pred = predict_large_volume(
            args.seis_path, args.checkpoint, shape, tuple(args.chunk_size), args.device
        )
    else:
        pred = predict_single_chunk(args.seis_path, args.checkpoint, shape, args.device)

    pred.astype(np.float32).tofile(args.output)
    print(f"Saved prediction to {args.output}, shape: {pred.shape}")
