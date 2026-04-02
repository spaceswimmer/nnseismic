import torch
import numpy as np
from glob import glob
from lw_spacenet import UNet3D
import matplotlib.pyplot as plt


def load_model(checkpoint_path, device='cuda'):
    """Load trained model from checkpoint."""
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    model = UNet3D(in_channels=1, out_channels=1, init_features=16)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.bfloat16().to(device)
    model.eval()
    
    return model, device


def predict(model, seismic, device):
    """
    Run prediction on seismic volume.
    
    Args:
        model: Loaded UNet3D model
        seismic: numpy array (128, 128, 128) or torch tensor
        device: torch device
        
    Returns:
        predicted_rgt: numpy array (128, 128, 128)
    """
    # Convert to tensor if needed
    if isinstance(seismic, np.ndarray):
        seismic = torch.tensor(seismic, dtype=torch.bfloat16)
    
    # Ensure shape (1, 1, 128, 128, 128)
    if seismic.dim() == 3:
        seismic = seismic.unsqueeze(0).unsqueeze(0)
    
    seismic = seismic.to(device)
    
    with torch.no_grad():
        output = model(seismic)
        predicted = output.cpu().float().numpy().squeeze()
    
    return predicted

def slice_data_chunks_with_stride(data, chunk_size=(128, 128, 128), stride=None):
    """
    Slice input data of shape (150, 150, 2000) into overlapping chunks of specified size.
    
    Args:
        data: Input numpy array of shape (150, 150, 2000)
        chunk_size: Tuple of (height, width, depth) for each chunk
        stride: Step size between chunks (default is half the chunk size for 50% overlap)
    
    Returns:
        List of numpy arrays, each of shape chunk_size
    """
    h, w, d = data.shape
    ch, cw, cd = chunk_size
    
    # Default stride is half the chunk size for 50% overlap
    if stride is None:
        stride = (ch // 2, cw // 2, cd // 2)
    
    sh, sw, sd = stride
    
    chunks = []
    
    # Calculate how many chunks we can fit in each dimension with the given stride
    h_steps = max(1, (h - ch) // sh + 1)
    w_steps = max(1, (w - cw) // sw + 1)
    d_steps = max(1, (d - cd) // sd + 1)
    
    for i in range(h_steps):
        for j in range(w_steps):
            for k in range(d_steps):
                start_h, end_h = i * sh, i * sh + ch
                start_w, end_w = j * sw, j * sw + cw
                start_d, end_d = k * sd, k * sd + cd
                
                # Ensure we don't exceed the data bounds
                end_h = min(end_h, h)
                end_w = min(end_w, w)
                end_d = min(end_d, d)
                
                # Adjust start positions if we're near the boundary
                start_h = max(0, end_h - ch)
                start_w = max(0, end_w - cw)
                start_d = max(0, end_d - cd)
                
                chunk = data[start_h:end_h, start_w:end_w, start_d:end_d]
                chunks.append(chunk)
    del chunks[0:3]
    return chunks

def robust_normalize_seismic(chunk, p_min=1, p_max=99):
    """Robust normalization using percentiles for seismic [-1, 1]"""
    q_min, q_max = np.percentile(chunk, [p_min, p_max])
    if q_max == q_min:
        return np.zeros_like(chunk)
    # Scale to [-1, 1] using robust percentiles
    normalized = 2 * (chunk - q_min) / (q_max - q_min) - 1
    # Optional: clip remaining outliers beyond the percentile range
    return np.clip(normalized, -1, 1)

def normalize_age_chunk(chunk):
    """Normalize age data to [0, 1] range within chunk"""
    min_val, max_val = chunk.min(), chunk.max()
    if max_val == min_val:  # Avoid division by zero
        return np.zeros_like(chunk)
    normalized = (chunk - min_val) / (max_val - min_val)
    return normalized


if __name__ == "__main__":
    # Load model
    model, device = load_model('../../data/DNN models/final_model.pth')
    
    seismic = glob("../../data/synthetic_data/run/*150-150-2000*/seismicCubes_cumsum_fullstack*.npy")
    age = glob("../../data/synthetic_data/run/*150-150-2000*/faulted_age*.npy")
    seis_chunk_norm = []
    age_chunk = []
    for s, a in zip(seismic, age):
        sdf = np.load(s)
        sdf_norm = robust_normalize_seismic(sdf)
        adf = np.load(a)
        seis_chunk_norm.extend(slice_data_chunks_with_stride(sdf_norm))
        age_chunk.extend(slice_data_chunks_with_stride(adf))
    age_chunk_norm = []
    for chunk in age_chunk:
        age_chunk_norm.append(normalize_age_chunk(chunk))

    seismic = torch.tensor(seis_chunk_norm[10], dtype=torch.bfloat16) 
    predicted_rgt = predict(model, seismic, device)
    
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(predicted_rgt[50].T)
    im = axs[1].imshow(age_chunk_norm[10][50].T)
    fig.colorbar(im)
    print(model)
    plt.show()
    