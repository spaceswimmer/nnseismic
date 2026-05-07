"""Slice a seismic volume into overlapping chunks and save to disk.

Usage:
    python DNN/slice_volume.py --input volume.npy --output chunks_dir --chunk-size 128 128 128 --stride 64 64 64
"""

import argparse
import os
import sys
import numpy as np
from typing import Tuple, List

script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(script_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from DNN.volume_merger import VolumeMerger


def slice_volume_to_files(
    volume: np.ndarray,
    output_dir: str,
    chunk_size: Tuple[int, int, int] = (128, 128, 128),
    stride: Tuple[int, int, int] = None,
    normalize: bool = True,
) -> Tuple[List[Tuple[int, int, int]], Tuple[int, int, int]]:
    """Slice a volume into overlapping chunks and save as .dat files.

    Saves chunks as {h}_{w}_{d}.dat using numpy tofile (float32).

    Args:
        volume: Input seismic volume (3D numpy array)
        output_dir: Directory to save chunk files
        chunk_size: Size of each chunk
        stride: Stride for overlapping chunks (default: half of chunk_size)
        normalize: Whether to normalize the volume before slicing

    Returns:
        Tuple of (positions, original_shape) for use in merging
    """
    os.makedirs(output_dir, exist_ok=True)

    if normalize:
        volume = (volume - np.mean(volume)) / np.std(volume)

    merger = VolumeMerger(chunk_size, stride)
    chunks, positions = merger.slice_volume(volume)

    print(f"Slicing volume {volume.shape} into {len(chunks)} chunks...")

    for i, (chunk, pos) in enumerate(zip(chunks, positions)):
        h, w, d = pos
        filename = f"{h}_{w}_{d}.dat"
        filepath = os.path.join(output_dir, filename)
        chunk.astype(np.float32).tofile(filepath)
        print(f"Saved {filename} (chunk {i + 1}/{len(chunks)})")

    original_shape = merger.original_shape

    meta_path = os.path.join(output_dir, "metadata.txt")
    with open(meta_path, "w") as f:
        f.write(
            f"original_shape={original_shape[0]},{original_shape[1]},{original_shape[2]}\n"
        )
        f.write(f"chunk_size={chunk_size[0]},{chunk_size[1]},{chunk_size[2]}\n")
        stride_actual = merger.stride
        f.write(f"stride={stride_actual[0]},{stride_actual[1]},{stride_actual[2]}\n")
        f.write(f"num_chunks={len(chunks)}\n")

    print(f"Saved metadata to {meta_path}")
    print(f"Original shape: {original_shape}")

    return positions, original_shape


def main():
    parser = argparse.ArgumentParser(description="Slice seismic volume into chunks")
    parser.add_argument("--input", type=str, required=True, help="Input volume (.npy)")
    parser.add_argument(
        "--output", type=str, required=True, help="Output directory for chunks"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        nargs=3,
        default=[128, 128, 128],
        help="Chunk size (default: 128 128 128)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        nargs=3,
        default=None,
        help="Stride (default: half of chunk size)",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable normalization (default: normalize)",
    )
    parser.add_argument(
        "--subvolume",
        type=int,
        nargs=6,
        default=None,
        help="Extract subvolume z1:z2,y1:y2,x1:x2",
    )

    args = parser.parse_args()

    print(f"Loading volume from {args.input}...")
    volume = np.load(args.input)
    print(f"Volume shape: {volume.shape}, dtype: {volume.dtype}")

    if args.subvolume:
        z1, z2, y1, y2, x1, x2 = args.subvolume
        volume = volume[z1:z2, y1:y2, x1:x2]
        print(f"Subvolume shape: {volume.shape}")

    chunk_size = tuple(args.chunk_size)
    stride = tuple(args.stride) if args.stride else None

    slice_volume_to_files(
        volume,
        args.output,
        chunk_size=chunk_size,
        stride=stride,
        normalize=not args.no_normalize,
    )

    print("Done!")


if __name__ == "__main__":
    main()
