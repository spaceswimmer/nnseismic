"""Script to predict RGT on a large seismic volume from .npy file.

Run from src directory: python DNN/predict_volume.py ...
"""

import argparse
import gc
import os
import sys
import numpy as np
import torch

script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(script_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from DNN.large_volume_predictor import LargeVolumePredictor


def main():
    parser = argparse.ArgumentParser(
        description="Predict RGT from seismic volume (.npy)"
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Path to model checkpoint (.pth)"
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Path to input seismic volume (.npy)"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to output RGT volume (.npy)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        nargs=3,
        default=[128, 128, 128],
        help="Chunk size for prediction (default: 128 128 128)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        nargs=3,
        default=None,
        help="Stride for overlapping chunks (default: half of chunk size)",
    )
    parser.add_argument(
        "--smoothing-kernel",
        type=int,
        default=5,
        help="Smoothing kernel size (default: 5)",
    )
    parser.add_argument(
        "--in-channels",
        type=int,
        default=1,
        help="Number of input channels (default: 1)",
    )
    parser.add_argument(
        "--init-features",
        type=int,
        default=16,
        help="Initial features for UNet (default: 16)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (default: cuda)",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable input normalization (default: normalize)",
    )
    parser.add_argument(
        "--subvolume",
        type=int,
        nargs=6,
        default=None,
        help="Extract subvolume as z1:z2,y1:y2,x1:x2 (default: full volume)",
    )
    parser.add_argument(
        "--disk-mode",
        action="store_true",
        help="Use disk-based merging for large volumes (saves chunks to disk, uses less RAM)",
    )
    parser.add_argument(
        "--temp-dir",
        type=str,
        default=None,
        help="Temporary directory for chunk storage (default: auto-created temp dir)",
    )

    args = parser.parse_args()

    print(f"Loading seismic volume from {args.input}...")
    seismic = np.load(args.input)
    print(f"Loaded volume shape: {seismic.shape}, dtype: {seismic.dtype}")

    if args.subvolume:
        z1, z2, y1, y2, x1, x2 = args.subvolume
        seismic = seismic[z1:z2, y1:y2, x1:x2]
        print(f"Using subvolume shape: {seismic.shape}")

    chunk_size = tuple(args.chunk_size)
    stride = tuple(args.stride) if args.stride else None

    predictor = LargeVolumePredictor(
        model_path=args.model,
        device=args.device,
        chunk_size=chunk_size,
        stride=stride,
        smoothing_kernel_size=args.smoothing_kernel,
        in_channels=args.in_channels,
        init_features=args.init_features,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)

    if args.disk_mode:
        print("Using disk-based merging mode...")
        output_path = predictor.predict_full_volume_disk(
            seismic,
            output_path=args.output,
            normalize=not args.no_normalize,
            temp_dir=args.temp_dir,
        )
        print(f"Saved RGT volume to {output_path}")
    else:
        print("Using in-memory merging mode...")
        rgt_volume = predictor.predict_full_volume(
            seismic,
            normalize=not args.no_normalize,
        )
        print(f"Saving RGT volume to {args.output}...")
        np.save(args.output, rgt_volume.astype(np.float32))
        print(f"Saved RGT volume to {args.output}")

    print("Done!")

    del seismic
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
