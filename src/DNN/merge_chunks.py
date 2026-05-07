"""Merge predicted RGT chunks into a full volume.

Chunks should be named as {h}_{w}_{d}_rgt.dat (or similar pattern with coordinates).
Reads float32 binary files and merges based on filename coordinates.

Usage:
    python DNN/merge_chunks.py --input chunks_dir --output rgt_volume.npy --chunk-size 128 128 128
"""

import argparse
import os
import re
import sys
import glob
import numpy as np
from typing import Tuple, List, Dict

script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(script_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from DNN.volume_merger import VolumeMerger, DiskVolumeMerger


def parse_position_from_filename(filename: str) -> Tuple[int, int, int]:
    """Parse position from filename like '64_128_0_rgt.dat' or '64_128_0.dat'."""
    basename = os.path.splitext(filename)[0]
    match = re.match(r"^(\d+)_(\d+)_(\d+)", basename)
    if match:
        return tuple(int(x) for x in match.groups())
    raise ValueError(f"Cannot parse position from filename: {filename}")


def load_metadata(meta_path: str) -> Dict:
    """Load metadata from slicing step."""
    meta = {}
    with open(meta_path, "r") as f:
        for line in f:
            if "=" in line:
                key, value = line.strip().split("=")
                if key in ["original_shape", "chunk_size", "stride"]:
                    meta[key] = tuple(int(x) for x in value.split(","))
                elif key == "num_chunks":
                    meta[key] = int(value)
    return meta


def load_chunks_from_dir(
    input_dir: str,
    chunk_size: Tuple[int, int, int],
    pattern: str = "*_rgt.dat",
) -> Tuple[List[np.ndarray], List[Tuple[int, int, int]]]:
    """Load all predicted chunks from directory.

    Args:
        input_dir: Directory containing chunk files
        chunk_size: Expected shape of each chunk
        pattern: Glob pattern for chunk files (default: *_rgt.dat)

    Returns:
        Tuple of (chunks, positions)
    """
    chunk_files = sorted(glob.glob(os.path.join(input_dir, pattern)))

    if not chunk_files:
        raise ValueError(f"No files found matching pattern '{pattern}' in {input_dir}")

    chunks = []
    positions = []

    print(f"Loading {len(chunk_files)} chunks from {input_dir}...")

    for filepath in chunk_files:
        filename = os.path.basename(filepath)
        pos = parse_position_from_filename(filename)

        chunk_data = np.fromfile(filepath, dtype=np.float32)
        chunk = chunk_data.reshape(chunk_size)

        chunks.append(chunk)
        positions.append(pos)
        print(f"Loaded {filename} at position {pos}")

    return chunks, positions


def merge_chunks(
    chunks: List[np.ndarray],
    positions: List[Tuple[int, int, int]],
    original_shape: Tuple[int, int, int],
    chunk_size: Tuple[int, int, int],
    stride: Tuple[int, int, int] = None,
    disk_mode: bool = False,
    output_path: str = None,
    temp_dir: str = None,
) -> np.ndarray:
    """Merge chunks into full volume.

    Args:
        chunks: List of predicted chunk arrays
        positions: List of chunk positions (h, w, d)
        original_shape: Shape of the original volume
        chunk_size: Size of each chunk
        stride: Stride used for slicing (default: half of chunk_size)
        disk_mode: Use disk-based merging for large volumes
        output_path: Path to save output (required for disk_mode)
        temp_dir: Temporary directory for disk mode

    Returns:
        Merged RGT volume
    """
    if disk_mode:
        if output_path is None:
            raise ValueError("output_path required for disk_mode")

        merger = DiskVolumeMerger(chunk_size, stride, temp_dir=temp_dir)
        merger.original_shape = original_shape

        for i, (chunk, pos) in enumerate(zip(chunks, positions)):
            merger.save_chunk(i, chunk, pos)

        merger.merge_all_predictions_disk(output_path)
        merger.cleanup()

        result = np.load(output_path)
    else:
        merger = VolumeMerger(chunk_size, stride)
        merger.original_shape = original_shape
        result = merger.merge_all_predictions(chunks, positions)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Merge predicted RGT chunks into volume"
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Directory containing chunk files"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Output volume (.npy)"
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
        help="Stride (default: half of chunk size, or read from metadata)",
    )
    parser.add_argument(
        "--original-shape",
        type=int,
        nargs=3,
        default=None,
        help="Original volume shape (default: read from metadata)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*_rgt.dat",
        help="Glob pattern for chunk files (default: *_rgt.dat)",
    )
    parser.add_argument(
        "--disk-mode",
        action="store_true",
        help="Use disk-based merging (for large volumes)",
    )
    parser.add_argument(
        "--temp-dir", type=str, default=None, help="Temp directory for disk mode"
    )

    args = parser.parse_args()

    meta_path = os.path.join(args.input, "metadata.txt")
    if os.path.exists(meta_path):
        print(f"Loading metadata from {meta_path}...")
        meta = load_metadata(meta_path)

        if args.original_shape is None:
            args.original_shape = list(meta.get("original_shape", args.chunk_size))
        if args.stride is None and "stride" in meta:
            args.stride = list(meta["stride"])

        print(
            f"From metadata: original_shape={meta.get('original_shape')}, stride={meta.get('stride')}"
        )

    chunk_size = tuple(args.chunk_size)
    original_shape = tuple(args.original_shape) if args.original_shape else None
    stride = tuple(args.stride) if args.stride else None

    chunks, positions = load_chunks_from_dir(args.input, chunk_size, args.pattern)

    if original_shape is None:
        max_h = max(p[0] + chunk_size[0] for p in positions)
        max_w = max(p[1] + chunk_size[1] for p in positions)
        max_d = max(p[2] + chunk_size[2] for p in positions)
        original_shape = (max_h, max_w, max_d)
        print(f"Estimated original shape: {original_shape}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)

    print(f"Merging {len(chunks)} chunks...")
    result = merge_chunks(
        chunks,
        positions,
        original_shape,
        chunk_size,
        stride,
        disk_mode=args.disk_mode,
        output_path=args.output,
        temp_dir=args.temp_dir,
    )

    if not args.disk_mode:
        print(f"Saving to {args.output}...")
        np.save(args.output, result.astype(np.float32))

    print(f"Output shape: {result.shape}")
    print("Done!")


if __name__ == "__main__":
    main()
