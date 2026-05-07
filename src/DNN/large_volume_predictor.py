"""Large-scale seismic volume prediction using chunked inference and merging.

Supports both in-memory and disk-based approaches for large volumes.
"""

import torch
import numpy as np
from typing import Tuple, Optional
from tqdm import tqdm
import gc
import os

from DNN.lw_spacenet import UNet3D
from DNN.volume_merger import VolumeMerger, DiskVolumeMerger


class LargeVolumePredictor:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        chunk_size: Tuple[int, int, int] = (128, 128, 128),
        stride: Tuple[int, int, int] = None,
        smoothing_kernel_size: int = 5,
        in_channels: int = 1,
        init_features: int = 16,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.chunk_size = chunk_size
        self.stride = stride

        self.model = UNet3D(
            in_channels=in_channels,
            out_channels=1,
            init_features=init_features,
            smoothing_kernel_size=smoothing_kernel_size,
        )
        checkpoint = torch.load(
            model_path, map_location=self.device, weights_only=False
        )
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)
        self.model = self.model.to(self.device).bfloat16()
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

        chunk_tensor = torch.tensor(
            chunk_padded, dtype=torch.bfloat16, device=self.device
        )

        with torch.no_grad():
            output = self.model(chunk_tensor)
            prediction = output.cpu().float().numpy().squeeze()

        if pad_h > 0 or pad_w > 0 or pad_d > 0:
            prediction = prediction[:h, :w, :d]

        return prediction.astype(np.float32)

    def predict_full_volume(
        self, seismic_volume: np.ndarray, normalize: bool = True
    ) -> np.ndarray:
        """Predict using in-memory approach (for smaller volumes)."""
        if normalize:
            seismic_norm = self._normalize_seismic(seismic_volume)
        else:
            seismic_norm = seismic_volume

        merger = VolumeMerger(self.chunk_size, self.stride)
        chunks, positions = merger.slice_volume(seismic_norm)

        predictions = []
        for i, chunk in enumerate(tqdm(chunks, desc="Predicting chunks")):
            pred = self.predict_chunk(chunk)
            predictions.append(pred)

            if (i + 1) % 10 == 0:
                torch.cuda.empty_cache()

        return merger.merge_all_predictions(predictions, positions)

    def predict_full_volume_disk(
        self,
        seismic_volume: np.ndarray,
        output_path: str,
        normalize: bool = True,
        temp_dir: Optional[str] = None,
    ) -> str:
        """Predict using disk-based approach (for large volumes).

        Saves predicted chunks to disk and merges them using memmap-like approach.
        Returns path to the output .npy file.
        """
        if normalize:
            seismic_norm = self._normalize_seismic(seismic_volume)
        else:
            seismic_norm = seismic_volume

        merger = DiskVolumeMerger(self.chunk_size, self.stride, temp_dir=temp_dir)
        positions = merger.slice_volume_positions_only(seismic_norm.shape)

        for i, pos in enumerate(tqdm(positions, desc="Predicting chunks")):
            h, w, d = pos
            ch, cw, cd = self.chunk_size

            chunk = seismic_norm[h : h + ch, w : w + cw, d : d + cd]
            pred = self.predict_chunk(chunk)
            merger.save_chunk(i, pred, pos)

            if (i + 1) % 10 == 0:
                torch.cuda.empty_cache()
                gc.collect()

        del seismic_norm
        gc.collect()

        output_path = merger.merge_all_predictions_disk(output_path)
        merger.cleanup()

        return output_path

    @staticmethod
    def _normalize_seismic(seismic: np.ndarray) -> np.ndarray:
        return (seismic - np.mean(seismic)) / np.std(seismic)
