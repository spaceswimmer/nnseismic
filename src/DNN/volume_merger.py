"""Volume merger for combining overlapping RGT predictions from large-scale seismic volumes."""
import numpy as np
from typing import List, Tuple
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
        
        sd = self.stride[2]  # depth stride for overlap calculation
        
        for (h, w), items in groups.items():
            # Sort by depth position (ascending)
            items.sort(key=lambda x: x[1])
            
            # Initialize with first chunk as base
            first_chunk, first_d = items[0]
            merged = first_chunk.copy().astype(np.float32)
            merged_end_d = first_d + merged.shape[2]  # absolute end depth
            
            for i in range(1, len(items)):
                curr_chunk, curr_d = items[i]
                
                # Calculate overlap size between merged result and current chunk
                overlap_size = merged_end_d - curr_d
                overlap_size = min(overlap_size, curr_chunk.shape[2], merged.shape[2])
                
                if overlap_size > 0:
                    # Extract overlap regions from both arrays
                    merged_overlap = merged[:, :, -overlap_size:]
                    curr_overlap = curr_chunk[:, :, :overlap_size]
                    
                    # Compute shift from average residual in overlap region
                    # This aligns the means to maintain continuity
                    shift = np.mean(merged_overlap - curr_overlap)
                    
                    # Apply shift to entire current chunk
                    adjusted_curr = curr_chunk + shift
                    
                    # Create smooth linear blending weights for overlap region
                    weights = np.linspace(1, 0, overlap_size, dtype=np.float32).reshape(1, 1, -1)
                    
                    # Blend overlap using weighted average
                    merged[:, :, -overlap_size:] = (
                        merged_overlap * weights + 
                        adjusted_curr[:, :, :overlap_size] * (1 - weights)
                    )
                    
                    # Append non-overlapping portion from current chunk
                    new_portion = adjusted_curr[:, :, overlap_size:]
                    if new_portion.shape[2] > 0:
                        merged = np.concatenate([merged, new_portion], axis=2)
                        merged_end_d = curr_d + curr_chunk.shape[2]
                else:
                    # No overlap - concatenate directly (edge case)
                    merged = np.concatenate([merged, curr_chunk], axis=2)
                    merged_end_d = curr_d + curr_chunk.shape[2]
            
            merged_chunks.append(merged)
            merged_positions.append((h, w, first_d))
        
        return merged_chunks, merged_positions
    
    def horizontal_merge(self, chunks, positions, axis=0):
        groups = {}
        
        for chunk, pos in zip(chunks, positions):
            h, w, d = pos
            # Key: position along the two axes NOT being merged
            if axis == 0:
                key = (w, d)  # group by crossline and depth
                merge_pos = h
            elif axis == 1:
                key = (h, d)  # group by inline and depth
                merge_pos = w
            else:
                key = (h, w)  # group by inline and crossline
                merge_pos = d
            
            if key not in groups:
                groups[key] = []
            groups[key].append((chunk, merge_pos))
        
        merged_chunks = []
        merged_positions = []
        
        stride = self.stride[axis]
        
        for key, items in groups.items():
            items.sort(key=lambda x: x[1])
            
            first_chunk, first_pos = items[0]
            merged = first_chunk.copy().astype(np.float32)
            merged_end = first_pos + merged.shape[axis]
            
            for i in range(1, len(items)):
                curr_chunk, curr_pos = items[i]
                
                overlap_size = merged_end - curr_pos
                overlap_size = min(overlap_size, curr_chunk.shape[axis], merged.shape[axis])
                
                if overlap_size > 0:
                    # Extract overlap regions
                    if axis == 0:
                        merged_overlap = merged[-overlap_size:, :, :]
                        curr_overlap = curr_chunk[:overlap_size, :, :]
                    elif axis == 1:
                        merged_overlap = merged[:, -overlap_size:, :]
                        curr_overlap = curr_chunk[:, :overlap_size, :]
                    else:
                        merged_overlap = merged[:, :, -overlap_size:]
                        curr_overlap = curr_chunk[:, :, :overlap_size]
                    
                    # Fit linear model: merged_overlap = scale * curr_overlap + bias
                    scale, bias = self._fit_linear_tikhonov(
                        curr_overlap.flatten(), 
                        merged_overlap.flatten()
                    )
                    
                    # Transform current chunk using fitted mapping
                    transformed_curr = scale * curr_chunk.astype(np.float32) + bias
                    
                    # Create blending weights along merge axis
                    weights = np.linspace(1, 0, overlap_size, dtype=np.float32)
                    if axis == 0:
                        weights = weights.reshape(-1, 1, 1)
                    elif axis == 1:
                        weights = weights.reshape(1, -1, 1)
                    else:
                        weights = weights.reshape(1, 1, -1)
                    
                    # Blend overlap regions
                    if axis == 0:
                        merged[-overlap_size:, :, :] = (
                            merged_overlap * weights + 
                            transformed_curr[:overlap_size, :, :] * (1 - weights)
                        )
                        new_portion = transformed_curr[overlap_size:, :, :]
                    elif axis == 1:
                        merged[:, -overlap_size:, :] = (
                            merged_overlap * weights + 
                            transformed_curr[:, :overlap_size, :] * (1 - weights)
                        )
                        new_portion = transformed_curr[:, overlap_size:, :]
                    else:
                        merged[:, :, -overlap_size:] = (
                            merged_overlap * weights + 
                            transformed_curr[:, :, :overlap_size] * (1 - weights)
                        )
                        new_portion = transformed_curr[:, :, overlap_size:]
                    
                    if new_portion.shape[axis] > 0:
                        merged = np.concatenate([merged, new_portion], axis=axis)
                        merged_end = curr_pos + curr_chunk.shape[axis]
                else:
                    merged = np.concatenate([merged, curr_chunk], axis=axis)
                    merged_end = curr_pos + curr_chunk.shape[axis]
            
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
        ychunk, ypositions = self.horizontal_merge(xchunk, xpositions, axis=1)
        x, y, z = self.original_shape
        result = ychunk[:x, :y, :z]
        return result