import sys
import os
import gc
import numpy as np
import json
import lasio as las
from typing import Dict
# Add the parent directory to the system path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)
from src.util.gaussian_processes import GPModel, MultitaskGPModel
import gpytorch
import torch
import segyio
import time

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def read_las(las_files):
    las_dfs = {}

    for file_path in las_files:
        # Read the LAS file
        las_obj = las.read(file_path)
        
        # Convert to DataFrame
        df = las_obj.df()
        
        # Extract filename without extension (e.g., 'Tgl-30.las' -> 'Tgl-30')
        filename = os.path.basename(file_path)
        key_name = os.path.splitext(filename)[0]
        
        # Store in dictionary
        las_dfs[key_name] = df
        
        print(f"Loaded: {key_name} - {len(df)} rows, {len(df.columns)} curves")
    return las_dfs

def read_sgy_selective(filepath, endian='big'):
    with segyio.open(filepath, mode='r', endian=endian, ignore_geometry=True) as src:
        traces = segyio.tools.collect(src.trace[:])
        # traces = 123
        # Select only needed fields (still vectorized)
        il = src.attributes(segyio.TraceField.INLINE_3D)[:]
        xl = src.attributes(segyio.TraceField.CROSSLINE_3D)[:]
    return traces, il, xl

def find_viable_arrays(folder_path):
    """
    Cycle through .npy files in a folder and return arrays with more than 1 unique value.
    
    Args:
        folder_path (str): Path to the folder containing .npy files
    
    Returns:
        list: List of tuples containing (filename, array) for arrays with >1 unique value
    """
    valid_arrays = []
    
    # Check if folder exists
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist")
        return valid_arrays
    
    # Get all .npy files in the folder
    npy_files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
    
    if not npy_files:
        print(f"No .npy files found in '{folder_path}'")
        return valid_arrays
    
    print(f"Found {len(npy_files)} .npy files")
    
    for filename in npy_files:
        file_path = os.path.join(folder_path, filename)
        
        try:
            # Load the array
            array = np.load(file_path)
            
            # Count unique values
            unique_values = np.unique(array)
            
            # Check if array has more than 1 unique value
            if len(unique_values) > 1:
                valid_arrays.append((filename))
                print(f"✓ {filename}: {len(unique_values)} unique values")
            else:
                print(f"✗ {filename}: Only 1 unique value ({unique_values[0]})")
                
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    
    print(f"\nFound {len(valid_arrays)} arrays with multiple unique values")
    return valid_arrays

def save_gp_model(gp_result: dict, filepath: str) -> None:
    """
    Save GP model and all components needed for prediction.
    
    Args:
        gp_result: Dictionary from fit_gp_model()
        filepath: Path to save the model (e.g., 'model.pt')
    """
    model = gp_result['model']
    num_tasks = model.num_tasks if hasattr(model, 'num_tasks') else 1
    
    # CRITICAL: Get training data from the model
    # ExactGP stores train_inputs and train_targets
    train_x = model.train_inputs[0].cpu()
    train_y = model.train_targets.cpu()
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'likelihood_state_dict': gp_result['likelihood'].state_dict(),
        'train_x': train_x,
        'train_y': train_y,
        'scaler_x': gp_result['scaler_x'],
        'scaler_y': gp_result['scaler_y'],
        'depth_range': gp_result['depth_range'],
        'lengthscale': gp_result['lengthscale'],
        'num_tasks': num_tasks,
    }, filepath)


def load_gp_model(filepath: str, device: str = 'cpu') -> dict:
    """
    Load GP model for prediction.
    
    Args:
        filepath: Path to saved model
        device: Device to load model on ('cpu' or 'cuda'). 
                Use 'cpu' for inference to avoid GPU memory accumulation.
        
    Returns:
        Dictionary compatible with predict_gp_model()
    """
    target_device = torch.device(device)
    
    # Always load checkpoint to CPU first
    checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
    
    num_tasks = checkpoint['num_tasks']
    lengthscale = checkpoint['lengthscale']
    
    # Extract scalers (they stay on CPU)
    scaler_x = checkpoint.get('scaler_x')
    scaler_y = checkpoint.get('scaler_y')
    depth_range = checkpoint.get('depth_range')
    
    # Keep training data on CPU to avoid GPU memory bloat
    train_x = checkpoint['train_x']
    train_y = checkpoint['train_y']
    
    # Recreate likelihood and model
    if num_tasks == 1:
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = GPModel(train_x, train_y, likelihood, lengthscale)
    else:
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks)
        model = MultitaskGPModel(train_x, train_y, likelihood, lengthscale, num_tasks)
    
    # Load trained hyperparameters
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    likelihood.load_state_dict(checkpoint['likelihood_state_dict'])
    
    # Move to target device AFTER loading state dict
    model = model.to(target_device)
    likelihood = likelihood.to(target_device)
    
    # Clean up checkpoint
    del checkpoint
    gc.collect()
    
    return {
        'model': model,
        'likelihood': likelihood,
        'scaler_x': scaler_x,
        'scaler_y': scaler_y,
        'depth_range': depth_range,
        'lengthscale': lengthscale,
        'num_tasks': num_tasks,
        'device': device  # Track which device model is on
    }

def save_gp_grid_params(
    gp_result: dict,
    filepath: str,
    depth_min: float = 0.0,
    depth_max: float = 4000.0,
    depth_step: float = 1.0,
    log_transform: bool = False
) -> None:
    """
    Pre-compute and save GP parameters on a fixed depth grid.
    
    Enables fast sampling with interpolation to arbitrary depth points.
    
    Args:
        gp_result: Dictionary from fit_gp_model()
        filepath: Where to save the .npz file
        depth_min: Minimum depth value
        depth_max: Maximum depth value
        depth_step: Step size for the grid
        log_transform: Whether the GP was fit on log-transformed data
    """
    depth_grid = np.arange(depth_min, depth_max + depth_step, depth_step)
    n_points = len(depth_grid)
    
    model = gp_result['model']
    likelihood = gp_result['likelihood']
    scaler_x = gp_result['scaler_x']
    scaler_y = gp_result['scaler_y']
    num_tasks = gp_result['num_tasks']
    
    # Normalize depth
    x_norm = scaler_x.transform(depth_grid.reshape(-1, 1))
    x_tensor = torch.tensor(x_norm, dtype=torch.float32, device=DEVICE)
    
    model.eval()
    likelihood.eval()
    
    with torch.no_grad():
        # Use model(x_tensor) directly for smooth latent GP (no observation noise)
        preds = model(x_tensor)
        
        if num_tasks == 1:
            cov_norm = preds.covariance_matrix.cpu().numpy()
            mean_norm = preds.mean.cpu().numpy()
            
            mean = scaler_y.inverse_transform(mean_norm.reshape(-1, 1)).flatten()
            cov = cov_norm * (scaler_y.scale_[0] ** 2)
            
            if log_transform:
                variance = np.diag(cov)
                mean = np.exp(mean + variance / 2)
        else:
            cov_norm = preds.covariance_matrix.cpu().numpy()
            mean_norm = preds.mean.cpu().numpy()
            
            mean_grid = mean_norm.reshape(n_points, num_tasks)
            mean = scaler_y.inverse_transform(mean_grid)
            
            cov_reshaped = cov_norm.reshape(n_points, num_tasks, n_points, num_tasks)
            scale_matrix = np.outer(scaler_y.scale_, scaler_y.scale_)
            cov_scaled = cov_reshaped * scale_matrix[np.newaxis, :, np.newaxis, :]
            cov = cov_scaled.reshape(n_points * num_tasks, n_points * num_tasks)
        
        del preds, x_tensor
    
    # Ensure positive definiteness with adaptive jitter
    cov_stable = cov.copy()
    jitter = 1e-5
    max_attempts = 10
    for attempt in range(max_attempts):
        try:
            cov_stable = cov + jitter * np.eye(cov.shape[0])
            L = np.linalg.cholesky(cov_stable)
            break
        except np.linalg.LinAlgError:
            jitter *= 10
            if attempt == max_attempts - 1:
                raise RuntimeError(f"Could not compute Cholesky decomposition even with jitter={jitter}")
    
    np.savez_compressed(
        filepath,
        mean=mean.astype(np.float32),
        L=L.astype(np.float32),
        depth_grid=depth_grid.astype(np.float32),
        num_tasks=num_tasks,
        log_transform=log_transform
    )
    
    file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"Saved GP grid params to {filepath} ({file_size_mb:.1f} MB)")
    print(f"  Depth range: {depth_min} - {depth_max}, step {depth_step} ({n_points} points)")
    print(f"  Num tasks: {num_tasks}")
    if jitter > 1e-5:
        print(f"  Note: Used jitter={jitter:.2e} for numerical stability")

def load_gp_grid_params(filepath: str) -> dict:
    """
    Load pre-computed GP parameters for fast sampling.
    
    Args:
        filepath: Path to .npz file from save_gp_grid_params()
    
    Returns:
        Dictionary with mean, L, depth_grid, and metadata
    """
    data = np.load(filepath)
    return {
        'mean': data['mean'],
        'L': data['L'],
        'depth_grid': data['depth_grid'],
        'num_tasks': int(data['num_tasks']),
        'log_transform': bool(data['log_transform']),
        # Optional fields (not needed for sampling, but included if present)
        'scaler_y_mean': data.get('scaler_y_mean', None),
        'scaler_y_scale': data.get('scaler_y_scale', None)
    }

def save_chebyshev_approximation(cheb_params: dict, filepath: str) -> None:
    """
    Save Chebyshev parameters to JSON file.
    
    Args:
        cheb_params: Dict from fit_chebyshev_approximation
        filepath: Path to save JSON file
    """
    with open(filepath, 'w') as f:
        json.dump(cheb_params, f, indent=2)


def load_chebyshev_approximation(filepath: str) -> dict:
    """
    Load Chebyshev parameters from JSON file.
    
    Args:
        filepath: Path to JSON file
    
    Returns:
        Dict with Chebyshev parameters
    """
    with open(filepath, 'r') as f:
        return json.load(f)