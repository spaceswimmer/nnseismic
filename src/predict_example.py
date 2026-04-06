from DNN.large_volume_predictor import LargeVolumePredictor
import util.filehandler as fh
import numpy as np
import gc
import matplotlib.pyplot as plt

def normalize_seismic(seismic):
    """Normalize seismic to [0, 1] using min-max."""
    min_val = np.min(seismic)
    max_val = np.max(seismic)
    if max_val - min_val == 0:
        return np.zeros_like(seismic)
    return (seismic - min_val) / (max_val - min_val)

# Initialize predictor
predictor = LargeVolumePredictor(
    model_path='../data/DNN models/run-20260404_152544/best_model.pth',
    chunk_size=(128, 128, 128),
    stride=(64, 64, 64),
    smoothing_kernel_size=3
)
file = "../data/Vankorskaya_s_p_5_03-04_Migrirovannyiy_PreStack.sgy"
print('reading segy')
il_range = (5110, 5510)
xl_range = (1100, 1500)
traces, _, _ = fh.read_sgy_selective(file, il_range, xl_range)
gc.collect()
# Predict full RGT volume
rgt_volume = predictor.predict_full_volume(traces, merge_method='weighted')
np.save('../data/predicted_rgt_volume.npy', rgt_volume)
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
axs[0].imshow(traces[75, :, :].T, cmap='seismic', aspect='auto')
# axs[1].imshow(rgt_orig[75, :, :].T, cmap='prism', aspect='auto')
axs[1].imshow(rgt_volume[75, :, :].T, cmap='prism', aspect='auto')
plt.show()
# Save result
# 