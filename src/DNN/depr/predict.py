import torch
import numpy as np
import argparse
import os
from pathlib import Path
from lw_spacenet import UNet3D


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
    from DNN.depr.large_volume_predictor import LargeVolumePredictor

    predictor = LargeVolumePredictor(
        model_path=checkpoint_path, device=device, chunk_size=chunk_size
    )

    seismic = np.fromfile(seis_path, dtype=np.float32).reshape(shape)
    pred = predictor.predict_full_volume(seismic)
    return pred


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict RGT from seismic data")
    parser.add_argument("--seis_path", type=str, help="Path to seismic binary file")
    parser.add_argument(
        "--folder", type=str, help="Path to folder containing binary files"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--output", type=str, default="pred.bin", help="Output path for prediction"
    )
    parser.add_argument(
        "--output_dir", type=str, help="Output directory for folder processing"
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

    if not args.seis_path and not args.folder:
        parser.error("Either --seis_path or --folder is required")

    shape = tuple(args.shape)

    def process_file(seis_path, output_path):
        if args.large:
            pred = predict_large_volume(
                seis_path, args.checkpoint, shape, tuple(args.chunk_size), args.device
            )
        else:
            pred = predict_single_chunk(seis_path, args.checkpoint, shape, args.device)

        pred.astype(np.float32).tofile(output_path)
        print(f"Saved prediction to {output_path}, shape: {pred.shape}")

    if args.folder:
        folder_path = Path(args.folder)
        output_dir = (
            Path(args.output_dir) if args.output_dir else folder_path / "predictions"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        binary_files = sorted(folder_path.glob("*.dat"))
        if not binary_files:
            print(f"No .dat files found in {args.folder}")
            exit(1)

        for seis_file in binary_files:
            output_path = output_dir / f"pred_{seis_file.name}"
            print(f"Processing {seis_file.name}...")
            process_file(str(seis_file), str(output_path))

        print(f"Processed {len(binary_files)} files. Output saved to {output_dir}")
    else:
        process_file(args.seis_path, args.output)
