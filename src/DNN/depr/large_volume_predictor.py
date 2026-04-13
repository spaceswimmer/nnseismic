"""
Large-scale seismic volume prediction using chunked inference and merging.
"""

import torch
import numpy as np
from typing import Tuple, Optional
from lw_spacenet import UNet3D
from volume_merger import VolumeMerger


class LargeVolumePredictor:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        chunk_size: Tuple[int, int, int] = (128, 128, 128),
        stride: Tuple[int, int, int] = None,
        smoothing_kernel_size: int = 5,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.chunk_size = chunk_size
        self.merger = VolumeMerger(chunk_size, stride)

        self.model = UNet3D(
            in_channels=1,
            out_channels=1,
            init_features=16,
            smoothing_kernel_size=smoothing_kernel_size,
        )
        checkpoint = torch.load(model_path, map_location=self.device)
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)
        self.model = self.model.float().to(self.device)
        self.model.eval()

    def predict_chunk(self, chunk: np.ndarray) -> np.ndarray:
        actual_shape = chunk.shape

        ch, cw, cd = self.chunk_size
        h, w, d = actual_shape

        pad_h = max(0, ch - h)
        pad_w = max(0, cw - w)
        pad_d = max(0, cd - d)

        if pad_h > 0 or pad_w > 0 or pad_d > 0:
            chunk_padded = np.pad(
                chunk,
                ((0, pad_h), (0, pad_w), (0, pad_d)),
                mode="constant",
                constant_values=0,
            )
        else:
            chunk_padded = chunk

        if chunk_padded.ndim == 3:
            chunk_padded = chunk_padded[np.newaxis, np.newaxis, ...]

        chunk_tensor = torch.tensor(chunk_padded, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            output = self.model(chunk_tensor)
            prediction = output.cpu().float().numpy().squeeze()

        if pad_h > 0 or pad_w > 0 or pad_d > 0:
            prediction = prediction[:h, :w, :d]

        return prediction

    def predict_full_volume(self, seismic_volume: np.ndarray) -> np.ndarray:
        seismic_norm = self._normalize_seismic(seismic_volume)

        chunks, positions = self.merger.slice_volume(seismic_norm)

        predictions = []
        for i, chunk in enumerate(chunks):
            print(f"Predicting chunk {i + 1}/{len(chunks)}")
            pred = self.predict_chunk(chunk)
            predictions.append(pred)

        return self.merger.merge_all_predictions(predictions, positions)

    @staticmethod
    def _normalize_seismic(seismic: np.ndarray) -> np.ndarray:
        return (seismic - np.mean(seismic)) / np.std(seismic)
