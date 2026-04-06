"""Volume merger for combining overlapping RGT predictions from large-scale seismic volumes."""
import numpy as np
from typing import List, Tuple
import gc
from scipy import ndimage


class VolumeMerger:
    """Merges overlapping RGT predictions into a full volume using weighted averaging."""
    
    def __init__(self, chunk_size: Tuple[int, int, int] = (128, 128, 128),
                 stride: Tuple[int, int, int] = None):
        self.chunk_size = chunk_size
        self.stride = stride if stride else tuple(s // 2 for s in chunk_size)
        self.original_shape = None
        self.padding = None
    
    def slice_volume(self, volume):
        self.original_shape = volume.shape
        self.pads = [0, 0, 0]
        for i, (shape, stride, chunk) in enumerate(zip(volume.shape, self.stride, self.chunk_size)):
            if shape < chunk:
                self.pads[i] = (chunk - shape)
            elif shape % stride != 0:
                remainder = shape % stride 
                self.pads[i] = (stride - remainder)
        padding = ((0, self.pads[0]), (0, self.pads[1]), (0, self.pads[2]))
        volume = np.pad(volume, padding, mode='edge')

        # the slicing
        chunks, positions = [], []
        steps = [shape//stride - 1 for shape, stride in zip(volume.shape, self.stride)]
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
            
            # Pre-calculate final merged size to avoid repeated concatenation
            first_chunk, first_d = items[0]
            last_chunk, last_d = items[-1]
            total_depth = last_d + last_chunk.shape[2] - first_d
            
            # Pre-allocate merged array once
            ch, cw = first_chunk.shape[0], first_chunk.shape[1]
            merged = np.zeros((ch, cw, total_depth), dtype=np.float32)
            
            # Fill first chunk
            merged[:, :, :first_chunk.shape[2]] = first_chunk.astype(np.float32)
            write_end = first_chunk.shape[2]
            
            for i in range(1, len(items)):
                curr_chunk, curr_d = items[i]
                curr_chunk = curr_chunk.astype(np.float32)
                
                # Convert absolute position to relative position in merged array
                rel_start = curr_d - first_d
                rel_end = rel_start + curr_chunk.shape[2]
                
                # Calculate overlap with already written region
                overlap_size = write_end - rel_start
                overlap_size = min(overlap_size, curr_chunk.shape[2], total_depth - rel_start)
                
                if overlap_size > 0:
                    merged_overlap = merged[:, :, rel_start:rel_start + overlap_size]
                    curr_overlap = curr_chunk[:, :, :overlap_size]
                    
                    shift = np.mean(merged_overlap - curr_overlap)
                    adjusted_curr = curr_chunk + shift
                    
                    weights = np.linspace(1, 0, overlap_size, dtype=np.float32).reshape(1, 1, -1)
                    
                    # Blend in-place using pre-allocated array
                    merged[:, :, rel_start:rel_start + overlap_size] = (
                        merged_overlap * weights + 
                        adjusted_curr[:, :, :overlap_size] * (1 - weights)
                    )
                    
                    # Write non-overlapping portion
                    if rel_end > write_end:
                        merged[:, :, write_end:rel_end] = adjusted_curr[:, :, overlap_size:]
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
            
            # Pre-calculate final merged size to avoid repeated concatenation
            first_chunk, first_pos = items[0]
            last_chunk, last_pos = items[-1]
            total_size = last_pos + last_chunk.shape[axis] - first_pos
            
            # Pre-allocate merged array once
            shape = list(first_chunk.shape)
            shape[axis] = total_size
            merged = np.zeros(shape, dtype=np.float32)
            
            # Fill first chunk using dynamic slicing
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
                overlap_size = min(overlap_size, curr_chunk.shape[axis], total_size - rel_start)
                
                if overlap_size > 0:
                    # Extract overlap regions using dynamic slicing
                    merged_slice = [slice(None)] * 3
                    merged_slice[axis] = slice(rel_start, rel_start + overlap_size)
                    merged_overlap = merged[tuple(merged_slice)]
                    
                    curr_slice = [slice(None)] * 3
                    curr_slice[axis] = slice(0, overlap_size)
                    curr_overlap = curr_chunk[tuple(curr_slice)]
                    
                    scale, bias = self._fit_linear_tikhonov(
                        curr_overlap.flatten(), 
                        merged_overlap.flatten()
                    )
                    
                    transformed_curr = scale * curr_chunk + bias
                    
                    # Create axis-appropriate weight shape
                    weights = np.linspace(1, 0, overlap_size, dtype=np.float32)
                    weight_shape = [1, 1, 1]
                    weight_shape[axis] = overlap_size
                    weights = weights.reshape(weight_shape)
                    
                    # Blend in-place
                    merged[tuple(merged_slice)] = (
                        merged_overlap * weights +
                        transformed_curr[tuple(curr_slice)] * (1 - weights)
                    )
                    
                    # Write non-overlapping portion
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
            
            # Reconstruct position from key
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
    
    def _fit_linear_tikhonov(self, x: np.ndarray, y: np.ndarray, 
                             lambda_reg: float = 0.01) -> Tuple[float, float]:
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
        
        # Design matrix with bias term: [x, 1]
        X = np.column_stack([x, np.ones_like(x)])
        
        # Tikhonov regularization: (X'X + λI)^(-1) X'y
        XtX = X.T @ X
        XtX += lambda_reg * np.eye(2)
        Xty = X.T @ y
        
        coefficients = np.linalg.solve(XtX, Xty)
        scale, bias = coefficients[0], coefficients[1]
        
        return scale, bias

        
    def merge_all_predictions(self, predictions: List[np.ndarray],
                             positions: List[Tuple[int, int, int]]) -> np.ndarray:
        zchunk, zpositions = self.vertical_merge(predictions, positions)
        
        xchunk, xpositions = self.horizontal_merge(zchunk, zpositions, axis=0)
        del zchunk, zpositions  # Free vertical merge memory
        
        ychunk, ypositions = self.horizontal_merge(xchunk, xpositions, axis=1)
        del xchunk, xpositions  # Free horizontal merge (axis 0) memory
        gc.collect()            # Force immediate cleanup
        
        x, y, z = self.original_shape
        result = ychunk[0][:x, :y, :z]
        return result