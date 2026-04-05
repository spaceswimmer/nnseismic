"""
Large-scale seismic volume prediction using chunked inference and merging.
"""
import torch
import numpy as np
from typing import Tuple, Optional
from .lw_spacenet import UNet3D
from .volume_merger import VolumeMerger


class LargeVolumePredictor:
    """
    Predicts RGT for large seismic volumes by chunking and merging.
    """
    
    def __init__(self, model_path: str, device: str = 'cuda',
                 chunk_size: Tuple[int, int, int] = (128, 128, 128),
                 stride: Tuple[int, int, int] = None,
                 smoothing_kernel_size: int = 5):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to trained model checkpoint
            device: Device to run inference on
            chunk_size: Size of chunks for prediction
            stride: Stride between chunks (default: 50% overlap)
            smoothing_kernel_size: Kernel size for output smoothing
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.chunk_size = chunk_size
        self.merger = VolumeMerger(chunk_size, stride)
        
        # Load model
        self.model = UNet3D(
            in_channels=1,
            out_channels=1,
            init_features=16,
            smoothing_kernel_size=smoothing_kernel_size
        )
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model = self.model.bfloat16().to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
    
    def predict_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """
        Predict RGT for a single chunk.
        
        Args:
            chunk: Seismic chunk array (may be smaller than 128³ near boundaries)
            
        Returns:
            rgt_prediction: RGT prediction (same size as input chunk)
        """
        # Get the actual size of the chunk
        actual_shape = chunk.shape
        
        # Pad chunk to required size if needed
        ch, cw, cd = self.chunk_size
        h, w, d = actual_shape
        
        # Pad if dimensions are smaller than required
        pad_h = max(0, ch - h)
        pad_w = max(0, cw - w)
        pad_d = max(0, cd - d)
        
        if pad_h > 0 or pad_w > 0 or pad_d > 0:
            chunk_padded = np.pad(chunk, 
                                 ((0, pad_h), (0, pad_w), (0, pad_d)), 
                                 mode='constant', constant_values=0)
        else:
            chunk_padded = chunk
        
        # Ensure correct shape for model
        if chunk_padded.ndim == 3:
            chunk_padded = chunk_padded[np.newaxis, np.newaxis, ...]
        
        chunk_tensor = torch.tensor(chunk_padded, dtype=torch.bfloat16).to(self.device)
        
        with torch.no_grad():
            output = self.model(chunk_tensor)
            prediction = output.cpu().float().numpy().squeeze()
        
        # Crop prediction back to original size if it was padded
        if pad_h > 0 or pad_w > 0 or pad_d > 0:
            prediction = prediction[:h, :w, :d]
        
        return prediction
    
    def predict_full_volume(self, seismic_volume: np.ndarray,
                           merge_method: str = 'weighted') -> np.ndarray:
        """
        Predict RGT for a full large-scale seismic volume.
        
        Args:
            seismic_volume: Full seismic volume (H, W, D)
            merge_method: 'weighted', 'vertical', or 'horizontal'
            
        Returns:
            rgt_volume: Full RGT prediction volume
        """
        # Normalize input
        seismic_norm = self._normalize_seismic(seismic_volume)
        
        # Slice into chunks
        chunks, positions = self.merger.slice_volume(seismic_norm)
        
        # Predict each chunk
        predictions = []
        for i, chunk in enumerate(chunks):
            print(f"Predicting chunk {i+1}/{len(chunks)}")
            pred = self.predict_chunk(chunk)
            predictions.append(pred)
        
        # Merge predictions
        return self.merger.merge_all_predictions(predictions, positions)
    
    @staticmethod
    def _normalize_seismic(seismic: np.ndarray) -> np.ndarray:
        """Normalize seismic to [0, 1] using min-max."""
        min_val = np.min(seismic)
        max_val = np.max(seismic)
        if max_val - min_val == 0:
            return np.zeros_like(seismic)
        return (seismic - min_val) / (max_val - min_val)