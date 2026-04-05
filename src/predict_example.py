from DNN.large_volume_predictor import LargeVolumePredictor
import numpy as np
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

seismic = np.load('../data/synthetic_data/run/seismic__2026.24405931_tagilsk_150-150-2000/seismicCubes_cumsum_fullstack_2026.24405931.npy')
seismic = normalize_seismic(seismic)
rgt_orig = np.load('../data/synthetic_data/run/seismic__2026.24405931_tagilsk_150-150-2000/faulted_age_2026.24405931.npy')
# Predict full RGT volume
# rgt_volume = predictor.predict_full_volume(seismic, merge_method='weighted')
slices, positions = predictor.merger.slice_volume(rgt_orig)
slices = normalize_seismic(slices)
mchunk, mpositions = predictor.merger.vertical_merge(slices, positions)
hchunk, hpositions = predictor.merger.horizontal_merge(mchunk, mpositions, axis=0)
fchunk, fpositions = predictor.merger.horizontal_merge(hchunk, hpositions, axis=1)
print(fpositions)
print(fchunk[0].shape)
x, y, z = rgt_orig.shape
fchunk = fchunk[0][:x, :y, :z]
fig, axs = plt.subplots(1, 3, figsize=(12, 6))
axs[0].imshow(seismic[75, :, :].T, cmap='seismic', aspect='auto')
axs[1].imshow(rgt_orig[75, :, :].T, cmap='prism', aspect='auto')
axs[2].imshow(fchunk[75, :, :].T, cmap='prism', aspect='auto')
plt.show()
# Save result
# np.save('predicted_rgt_volume.npy', rgt_volume)