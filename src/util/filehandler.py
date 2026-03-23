import os
import numpy as np
import lasio as las
from typing import Dict
from util.gaussian_processes import GPModel, MultitaskGPModel
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

import torch
from typing import Dict


import torch
from typing import Dict


def save_gp_model(gp_result: dict, filepath: str) -> None:
    """
    Save GP model and all components needed for prediction.
    
    Args:
        gp_result: Dictionary from fit_gp_model()
        filepath: Path to save the model (e.g., 'model.pt')
    """
    model = gp_result['model']
    num_tasks = model.num_tasks if hasattr(model, 'num_tasks') else 1
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'likelihood_state_dict': gp_result['likelihood'].state_dict(),
        'scaler_x': gp_result['scaler_x'],
        'scaler_y': gp_result['scaler_y'],
        'depth_range': gp_result['depth_range'],
        'lengthscale': gp_result['lengthscale'],
        'num_tasks': num_tasks,
    }, filepath)


def load_gp_model(filepath: str) -> dict:
    """
    Load GP model for prediction.
    
    Args:
        filepath: Path to saved model
        
    Returns:
        Dictionary compatible with predict_gp_model()
    """
    checkpoint = torch.load(filepath, map_location=DEVICE, weights_only=False)
    
    num_tasks = checkpoint['num_tasks']
    lengthscale = checkpoint['lengthscale']
    
    # Create dummy tensors for initialization (required by ExactGP)
    dummy_x = torch.empty(1, 1, device=DEVICE)
    dummy_y = torch.empty(1, num_tasks, device=DEVICE)
    
    # Recreate likelihood and model with correct architecture
    if num_tasks == 1:
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = GPModel(dummy_x, dummy_y, likelihood, lengthscale).to(DEVICE)
    else:
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks)
        model = MultitaskGPModel(dummy_x, dummy_y, likelihood, lengthscale, num_tasks).to(DEVICE)
    
    # Load trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    likelihood.load_state_dict(checkpoint['likelihood_state_dict'])
    
    return {
        'model': model,
        'likelihood': likelihood,
        'scaler_x': checkpoint['scaler_x'],
        'scaler_y': checkpoint['scaler_y'],
        'depth_range': checkpoint['depth_range'],
        'lengthscale': lengthscale,
    }