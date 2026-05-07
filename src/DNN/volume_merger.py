"""Volume merger for combining overlapping RGT predictions from large-scale seismic volumes.

Supports both in-memory and disk-based merging for large volumes.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
import gc
import os
import tempfile
import shutil
from pathlib import Path


class VolumeMerger:
    """Merges overlapping RGT predictions into a full volume using weighted averaging."""

    def __init__(
        self,
        chunk_size: Tuple[int, int, int] = (128, 128, 128),
        stride: Tuple[int, int, int] = None,
    ):
        self.chunk_size = chunk_size
        self.stride = stride if stride else tuple(s // 2 for s in chunk_size)
        self.original_shape = None
        self.padding = None

    def slice_volume(self, volume):
        self.original_shape = volume.shape
        self.pads = [0, 0, 0]
        for i, (shape, stride, chunk) in enumerate(
            zip(volume.shape, self.stride, self.chunk_size)
        ):
            if shape < chunk:
                self.pads[i] = chunk - shape
            elif shape % stride != 0:
                remainder = shape % stride
                self.pads[i] = stride - remainder
        padding = ((0, self.pads[0]), (0, self.pads[1]), (0, self.pads[2]))
        volume = np.pad(volume, padding, mode="edge")

        chunks, positions = [], []
        steps = [
            shape // stride - 1 for shape, stride in zip(volume.shape, self.stride)
        ]
        ch, cw, cd = self.chunk_size
        sh, sw, sd = self.stride
        for i in range(steps[0]):
            for j in range(steps[1]):
                for k in range(steps[2]):
                    start_h, start_w, start_d = i * sh, j * sw, k * sd
                    end_h, end_w, end_d = start_h + ch, start_w + cw, start_d + cd
                    chunks.append(volume[start_h:end_h, start_w:end_w, start_d:end_d])
                    positions.append((start_h, start_w, start_d))

        return chunks, positions

    def slice_volume_positions_only(
        self, volume_shape: Tuple[int, int, int]
    ) -> List[Tuple[int, int, int]]:
        """Return positions for slicing without actually slicing the volume."""
        self.original_shape = volume_shape
        self.pads = [0, 0, 0]
        for i, (shape, stride, chunk) in enumerate(
            zip(volume_shape, self.stride, self.chunk_size)
        ):
            if shape < chunk:
                self.pads[i] = chunk - shape
            elif shape % stride != 0:
                remainder = shape % stride
                self.pads[i] = stride - remainder

        padded_shape = (
            volume_shape[0] + self.pads[0],
            volume_shape[1] + self.pads[1],
            volume_shape[2] + self.pads[2],
        )

        positions = []
        steps = [
            shape // stride - 1 for shape, stride in zip(padded_shape, self.stride)
        ]
        sh, sw, sd = self.stride
        for i in range(steps[0]):
            for j in range(steps[1]):
                for k in range(steps[2]):
                    start_h, start_w, start_d = i * sh, j * sw, k * sd
                    positions.append((start_h, start_w, start_d))

        return positions

    def vertical_merge(self, chunks, positions):
        groups = {}
        for chunk, pos in zip(chunks, positions):
            h, w, d = pos
            key = (h, w)
            if key not in groups:
                groups[key] = []
            groups[key].append((chunk, d))

        merged_chunks = []
        merged_positions = []

        for (h, w), items in groups.items():
            items.sort(key=lambda x: x[1])

            first_chunk, first_d = items[0]
            last_chunk, last_d = items[-1]
            total_depth = last_d + last_chunk.shape[2] - first_d

            ch, cw = first_chunk.shape[0], first_chunk.shape[1]
            merged = np.zeros((ch, cw, total_depth), dtype=np.float32)

            merged[:, :, : first_chunk.shape[2]] = first_chunk.astype(np.float32)
            write_end = first_chunk.shape[2]

            for i in range(1, len(items)):
                curr_chunk, curr_d = items[i]
                curr_chunk = curr_chunk.astype(np.float32)

                rel_start = curr_d - first_d
                rel_end = rel_start + curr_chunk.shape[2]

                overlap_size = write_end - rel_start
                overlap_size = min(
                    overlap_size, curr_chunk.shape[2], total_depth - rel_start
                )

                if overlap_size > 0:
                    merged_overlap = merged[:, :, rel_start : rel_start + overlap_size]
                    curr_overlap = curr_chunk[:, :, :overlap_size]

                    shift = np.mean(merged_overlap - curr_overlap)
                    adjusted_curr = curr_chunk + shift

                    weights = np.linspace(1, 0, overlap_size, dtype=np.float32).reshape(
                        1, 1, -1
                    )

                    merged[:, :, rel_start : rel_start + overlap_size] = (
                        merged_overlap * weights
                        + adjusted_curr[:, :, :overlap_size] * (1 - weights)
                    )

                    if rel_end > write_end:
                        merged[:, :, write_end:rel_end] = adjusted_curr[
                            :, :, overlap_size:
                        ]
                else:
                    merged[:, :, rel_start:rel_end] = curr_chunk

                write_end = max(write_end, rel_end)

            merged_chunks.append(merged)
            merged_positions.append((h, w, first_d))

        return merged_chunks, merged_positions

    def horizontal_merge(self, chunks, positions, axis=0):
        groups = {}

        for chunk, pos in zip(chunks, positions):
            h, w, d = pos
            if axis == 0:
                key = (w, d)
                merge_pos = h
            elif axis == 1:
                key = (h, d)
                merge_pos = w
            else:
                key = (h, w)
                merge_pos = d

            if key not in groups:
                groups[key] = []
            groups[key].append((chunk, merge_pos))

        merged_chunks = []
        merged_positions = []

        for key, items in groups.items():
            items.sort(key=lambda x: x[1])

            first_chunk, first_pos = items[0]
            last_chunk, last_pos = items[-1]
            total_size = last_pos + last_chunk.shape[axis] - first_pos

            shape = list(first_chunk.shape)
            shape[axis] = total_size
            merged = np.zeros(shape, dtype=np.float32)

            slice_obj = [slice(None)] * 3
            slice_obj[axis] = slice(0, first_chunk.shape[axis])
            merged[tuple(slice_obj)] = first_chunk.astype(np.float32)
            write_end = first_chunk.shape[axis]

            for i in range(1, len(items)):
                curr_chunk, curr_pos = items[i]
                curr_chunk = curr_chunk.astype(np.float32)

                rel_start = curr_pos - first_pos
                rel_end = rel_start + curr_chunk.shape[axis]

                overlap_size = write_end - rel_start
                overlap_size = min(
                    overlap_size, curr_chunk.shape[axis], total_size - rel_start
                )

                if overlap_size > 0:
                    merged_slice = [slice(None)] * 3
                    merged_slice[axis] = slice(rel_start, rel_start + overlap_size)
                    merged_overlap = merged[tuple(merged_slice)]

                    curr_slice = [slice(None)] * 3
                    curr_slice[axis] = slice(0, overlap_size)
                    curr_overlap = curr_chunk[tuple(curr_slice)]

                    scale, bias = self._fit_linear_tikhonov(
                        curr_overlap.flatten(), merged_overlap.flatten()
                    )

                    transformed_curr = scale * curr_chunk + bias

                    weights = np.linspace(1, 0, overlap_size, dtype=np.float32)
                    weight_shape = [1, 1, 1]
                    weight_shape[axis] = overlap_size
                    weights = weights.reshape(weight_shape)

                    merged[tuple(merged_slice)] = (
                        merged_overlap * weights
                        + transformed_curr[tuple(curr_slice)] * (1 - weights)
                    )

                    if rel_end > write_end:
                        new_slice = [slice(None)] * 3
                        new_slice[axis] = slice(overlap_size, None)
                        write_slice = [slice(None)] * 3
                        write_slice[axis] = slice(write_end, rel_end)
                        merged[tuple(write_slice)] = transformed_curr[tuple(new_slice)]
                else:
                    write_slice = [slice(None)] * 3
                    write_slice[axis] = slice(rel_start, rel_end)
                    merged[tuple(write_slice)] = curr_chunk

                write_end = max(write_end, rel_end)

            if axis == 0:
                w, d = key
                merged_positions.append((first_pos, w, d))
            elif axis == 1:
                h, d = key
                merged_positions.append((h, first_pos, d))
            else:
                h, w = key
                merged_positions.append((h, w, first_pos))

            merged_chunks.append(merged)

        return merged_chunks, merged_positions

    def _fit_linear_tikhonov(
        self, x: np.ndarray, y: np.ndarray, lambda_reg: float = 0.01
    ) -> Tuple[float, float]:
        """
        Fit linear model y = a*x + b using Tikhonov regularization.

        Args:
            x: Source values (flattened)
            y: Target values (flattened)
            lambda_reg: Regularization parameter

        Returns:
            Tuple of (scale, bias) coefficients
        """
        x = x.flatten().astype(np.float64)
        y = y.flatten().astype(np.float64)

        X = np.column_stack([x, np.ones_like(x)])

        XtX = X.T @ X
        XtX += lambda_reg * np.eye(2)
        Xty = X.T @ y

        coefficients = np.linalg.solve(XtX, Xty)
        scale, bias = coefficients[0], coefficients[1]

        return scale, bias

    def merge_all_predictions(
        self, predictions: List[np.ndarray], positions: List[Tuple[int, int, int]]
    ) -> np.ndarray:
        xchunk, xpositions = self.horizontal_merge(predictions, positions, axis=0)

        ychunk, ypositions = self.horizontal_merge(xchunk, xpositions, axis=1)
        del xchunk, xpositions
        gc.collect()

        zchunk, zpositions = self.vertical_merge(ychunk, ypositions)
        del ychunk, ypositions
        gc.collect()

        x, y, z = self.original_shape
        result = zchunk[0][:x, :y, :z]
        return result


class DiskVolumeMerger:
    """Memory-efficient merger that stores chunks on disk and uses memmap for output."""

    def __init__(
        self,
        chunk_size: Tuple[int, int, int] = (128, 128, 128),
        stride: Tuple[int, int, int] = None,
        temp_dir: str = None,
    ):
        self.chunk_size = chunk_size
        self.stride = stride if stride else tuple(s // 2 for s in chunk_size)
        self.original_shape = None
        self.padding = None
        self.temp_dir = temp_dir or tempfile.mkdtemp(prefix="volume_merger_")
        self.chunk_dir = os.path.join(self.temp_dir, "chunks")
        os.makedirs(self.chunk_dir, exist_ok=True)
        self.chunk_paths: Dict[int, str] = {}
        self.position_map: Dict[int, Tuple[int, int, int]] = {}
        self._cleanup_temp = temp_dir is None

    def __del__(self):
        if self._cleanup_temp and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def cleanup(self):
        """Manually clean up temporary files."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def save_chunk(self, idx: int, chunk: np.ndarray, position: Tuple[int, int, int]):
        """Save a predicted chunk to disk."""
        path = os.path.join(self.chunk_dir, f"chunk_{idx:06d}.npy")
        np.save(path, chunk.astype(np.float32))
        self.chunk_paths[idx] = path
        self.position_map[idx] = position

    def load_chunk(self, idx: int) -> np.ndarray:
        """Load a chunk from disk."""
        return np.load(self.chunk_paths[idx])

    def get_positions(self) -> List[Tuple[int, int, int]]:
        """Get all positions in order."""
        return [self.position_map[i] for i in sorted(self.position_map.keys())]

    def slice_volume_positions_only(
        self, volume_shape: Tuple[int, int, int]
    ) -> List[Tuple[int, int, int]]:
        """Return positions for slicing without actually slicing the volume."""
        self.original_shape = volume_shape
        self.pads = [0, 0, 0]
        for i, (shape, stride, chunk) in enumerate(
            zip(volume_shape, self.stride, self.chunk_size)
        ):
            if shape < chunk:
                self.pads[i] = chunk - shape
            elif shape % stride != 0:
                remainder = shape % stride
                self.pads[i] = stride - remainder

        padded_shape = (
            volume_shape[0] + self.pads[0],
            volume_shape[1] + self.pads[1],
            volume_shape[2] + self.pads[2],
        )

        positions = []
        steps = [
            shape // stride - 1 for shape, stride in zip(padded_shape, self.stride)
        ]
        sh, sw, sd = self.stride
        for i in range(steps[0]):
            for j in range(steps[1]):
                for k in range(steps[2]):
                    start_h, start_w, start_d = i * sh, j * sw, k * sd
                    positions.append((start_h, start_w, start_d))

        return positions

    def vertical_merge_disk(
        self,
        chunk_paths: List[str],
        positions: List[Tuple[int, int, int]],
        output_dir: str,
    ) -> Tuple[List[str], List[Tuple[int, int, int]]]:
        """Merge chunks vertically along depth axis, saving results to disk."""
        groups: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}

        for idx, pos in enumerate(positions):
            h, w, d = pos
            key = (h, w)
            if key not in groups:
                groups[key] = []
            groups[key].append((idx, d))

        for key in groups:
            groups[key].sort(key=lambda x: x[1])

        merged_dir = os.path.join(output_dir, "vertical_merged")
        os.makedirs(merged_dir, exist_ok=True)
        merged_paths = []
        merged_positions = []

        for group_idx, ((h, w), items) in enumerate(groups.items()):
            first_idx, first_d = items[0]
            first_chunk = np.load(chunk_paths[first_idx])
            last_idx, last_d = items[-1]
            last_chunk = np.load(chunk_paths[last_idx])

            total_depth = last_d + last_chunk.shape[2] - first_d
            ch, cw = first_chunk.shape[0], first_chunk.shape[1]

            merged = np.zeros((ch, cw, total_depth), dtype=np.float32)
            merged[:, :, : first_chunk.shape[2]] = first_chunk
            write_end = first_chunk.shape[2]

            del first_chunk, last_chunk
            gc.collect()

            for i in range(1, len(items)):
                curr_idx, curr_d = items[i]
                curr_chunk = np.load(chunk_paths[curr_idx])

                rel_start = curr_d - first_d
                rel_end = rel_start + curr_chunk.shape[2]

                overlap_size = write_end - rel_start
                overlap_size = min(
                    overlap_size, curr_chunk.shape[2], total_depth - rel_start
                )

                if overlap_size > 0:
                    merged_overlap = merged[:, :, rel_start : rel_start + overlap_size]
                    curr_overlap = curr_chunk[:, :, :overlap_size]

                    shift = np.mean(merged_overlap - curr_overlap)
                    adjusted_curr = curr_chunk + shift

                    weights = np.linspace(1, 0, overlap_size, dtype=np.float32).reshape(
                        1, 1, -1
                    )

                    merged[:, :, rel_start : rel_start + overlap_size] = (
                        merged_overlap * weights
                        + adjusted_curr[:, :, :overlap_size] * (1 - weights)
                    )

                    if rel_end > write_end:
                        merged[:, :, write_end:rel_end] = adjusted_curr[
                            :, :, overlap_size:
                        ]
                else:
                    merged[:, :, rel_start:rel_end] = curr_chunk

                write_end = max(write_end, rel_end)
                del curr_chunk
                gc.collect()

            merged_path = os.path.join(merged_dir, f"merged_{group_idx:06d}.npy")
            np.save(merged_path, merged)
            merged_paths.append(merged_path)
            merged_positions.append((h, w, first_d))

            del merged
            gc.collect()

        for path in chunk_paths:
            if os.path.exists(path):
                os.remove(path)

        return merged_paths, merged_positions

    def horizontal_merge_disk(
        self,
        chunk_paths: List[str],
        positions: List[Tuple[int, int, int]],
        axis: int,
        output_dir: str,
    ) -> Tuple[List[str], List[Tuple[int, int, int]]]:
        """Merge chunks horizontally along specified axis, saving results to disk."""
        groups: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}

        for idx, pos in enumerate(positions):
            h, w, d = pos
            if axis == 0:
                key = (w, d)
                merge_pos = h
            elif axis == 1:
                key = (h, d)
                merge_pos = w
            else:
                key = (h, w)
                merge_pos = d

            if key not in groups:
                groups[key] = []
            groups[key].append((idx, merge_pos))

        for key in groups:
            groups[key].sort(key=lambda x: x[1])

        merged_dir = os.path.join(output_dir, f"horizontal_merged_axis{axis}")
        os.makedirs(merged_dir, exist_ok=True)
        merged_paths = []
        merged_positions = []

        for group_idx, (key, items) in enumerate(groups.items()):
            first_idx, first_pos = items[0]
            first_chunk = np.load(chunk_paths[first_idx])
            last_idx, last_pos = items[-1]
            last_chunk = np.load(chunk_paths[last_idx])

            total_size = last_pos + last_chunk.shape[axis] - first_pos

            shape = list(first_chunk.shape)
            shape[axis] = total_size
            merged = np.zeros(shape, dtype=np.float32)

            slice_obj = [slice(None)] * 3
            slice_obj[axis] = slice(0, first_chunk.shape[axis])
            merged[tuple(slice_obj)] = first_chunk
            write_end = first_chunk.shape[axis]

            del first_chunk, last_chunk
            gc.collect()

            for i in range(1, len(items)):
                curr_idx, curr_pos = items[i]
                curr_chunk = np.load(chunk_paths[curr_idx])

                rel_start = curr_pos - first_pos
                rel_end = rel_start + curr_chunk.shape[axis]

                overlap_size = write_end - rel_start
                overlap_size = min(
                    overlap_size, curr_chunk.shape[axis], total_size - rel_start
                )

                if overlap_size > 0:
                    merged_slice = [slice(None)] * 3
                    merged_slice[axis] = slice(rel_start, rel_start + overlap_size)
                    merged_overlap = merged[tuple(merged_slice)]

                    curr_slice = [slice(None)] * 3
                    curr_slice[axis] = slice(0, overlap_size)
                    curr_overlap = curr_chunk[tuple(curr_slice)]

                    scale, bias = self._fit_linear_tikhonov(
                        curr_overlap.flatten(), merged_overlap.flatten()
                    )

                    transformed_curr = scale * curr_chunk + bias

                    weights = np.linspace(1, 0, overlap_size, dtype=np.float32)
                    weight_shape = [1, 1, 1]
                    weight_shape[axis] = overlap_size
                    weights = weights.reshape(weight_shape)

                    merged[tuple(merged_slice)] = (
                        merged_overlap * weights
                        + transformed_curr[tuple(curr_slice)] * (1 - weights)
                    )

                    if rel_end > write_end:
                        new_slice = [slice(None)] * 3
                        new_slice[axis] = slice(overlap_size, None)
                        write_slice = [slice(None)] * 3
                        write_slice[axis] = slice(write_end, rel_end)
                        merged[tuple(write_slice)] = transformed_curr[tuple(new_slice)]
                else:
                    write_slice = [slice(None)] * 3
                    write_slice[axis] = slice(rel_start, rel_end)
                    merged[tuple(write_slice)] = curr_chunk

                write_end = max(write_end, rel_end)
                del curr_chunk
                gc.collect()

            merged_path = os.path.join(merged_dir, f"merged_{group_idx:06d}.npy")
            np.save(merged_path, merged)
            merged_paths.append(merged_path)

            if axis == 0:
                w, d = key
                merged_positions.append((first_pos, w, d))
            elif axis == 1:
                h, d = key
                merged_positions.append((h, first_pos, d))
            else:
                h, w = key
                merged_positions.append((h, w, first_pos))

            del merged
            gc.collect()

        for path in chunk_paths:
            if os.path.exists(path):
                os.remove(path)

        return merged_paths, merged_positions

    def _fit_linear_tikhonov(
        self, x: np.ndarray, y: np.ndarray, lambda_reg: float = 0.01
    ) -> Tuple[float, float]:
        """Fit linear model y = a*x + b using Tikhonov regularization."""
        x = x.flatten().astype(np.float64)
        y = y.flatten().astype(np.float64)

        X = np.column_stack([x, np.ones_like(x)])

        XtX = X.T @ X
        XtX += lambda_reg * np.eye(2)
        Xty = X.T @ y

        coefficients = np.linalg.solve(XtX, Xty)
        scale, bias = coefficients[0], coefficients[1]

        return scale, bias

    def merge_all_predictions_disk(self, output_path: str) -> str:
        """Merge all predictions using disk-based approach.

        Order: horizontal (axis=0) → horizontal (axis=1) → vertical
        This preserves lateral horizon continuity before depth-wise stitching.
        """
        chunk_paths = [self.chunk_paths[i] for i in sorted(self.chunk_paths.keys())]
        positions = [self.position_map[i] for i in sorted(self.position_map.keys())]

        xpaths, xpositions = self.horizontal_merge_disk(
            chunk_paths, positions, axis=0, output_dir=self.temp_dir
        )

        ypaths, ypositions = self.horizontal_merge_disk(
            xpaths, xpositions, axis=1, output_dir=self.temp_dir
        )

        zpaths, zpositions = self.vertical_merge_disk(
            ypaths, ypositions, output_dir=self.temp_dir
        )

        final_chunk = np.load(zpaths[0])
        x, y, z = self.original_shape
        result = final_chunk[:x, :y, :z]

        np.save(output_path, result.astype(np.float32))

        del final_chunk, result
        gc.collect()

        return output_path
