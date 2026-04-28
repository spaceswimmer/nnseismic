import util.filehandler as fh
import numpy as np

def normalize_seismic(seismic):
    """Normalize seismic to [0, 1] using min-max."""
    min_val = np.min(seismic)
    max_val = np.max(seismic)
    if max_val - min_val == 0:
        return np.zeros_like(seismic)
    return (seismic - min_val) / (max_val - min_val)

file = "/mnt/storage/nnseismic/real_data/Vankorskaya_s_p_5_03-04_Migrirovannyiy_PreStack.sgy"
il_range = (5110, 5510)
xl_range = (1100, 1500)

print('reading segy')
traces, _, _ = fh.read_sgy_selective(file, il_range, xl_range)

print('normalizing')
normalized = normalize_seismic(traces).astype(np.float32)

print('extracting chunks')
subset = normalized[0:256, :, 1000:1512]

chunk_size = 128
chunk_idx = 0
for i in range(subset.shape[0] // chunk_size):
    for j in range(subset.shape[1] // chunk_size):
        for k in range(subset.shape[2] // chunk_size):
            chunk = subset[i*chunk_size:(i+1)*chunk_size, 
                           j*chunk_size:(j+1)*chunk_size, 
                           k*chunk_size:(k+1)*chunk_size]
            output_path = f'/mnt/storage/nnseismic/real_data/{chunk_idx}.dat'
            chunk.tofile(output_path)
            print(f'saved chunk {chunk_idx} to {output_path}')
            chunk_idx += 1