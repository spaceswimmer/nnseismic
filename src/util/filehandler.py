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


def save_gp_model(gp_result: Dict, filepath: str):
    """
    Save a GP model and its associated components to a file using PyTorch.
    
    Args:
        gp_result: Dictionary containing the GP model and related components
                   Expected keys: 'model', 'likelihood', 'scaler_x', 'scaler_y', 'depth_range'
        filepath: Path where the model should be saved
    """
    # Get the hyperparameters from the model
    model = gp_result['model']
    
    # Extract lengthscale constraint lower bound
    if hasattr(model.covar_module.base_kernel, 'lengthscale_constraint'):
        lengthscale_lower_bound = model.covar_module.base_kernel.lengthscale_constraint.lower_bound.item()
    elif hasattr(model.covar_module, 'data_covar_module'):  # For multitask model
        lengthscale_lower_bound = model.covar_module.data_covar_module.lengthscale_constraint.lower_bound.item()
    else:
        lengthscale_lower_bound = 0.2  # Default fallback
    
    # Prepare the state dictionary with all necessary parameters
    state_dict = {
        'model_state_dict': model.state_dict(),
        'likelihood_state_dict': gp_result['likelihood'].state_dict(),
        'model_type': type(model).__name__,
        'num_tasks': getattr(model, 'num_tasks', 1),
        'lengthscale_lower_bound': lengthscale_lower_bound,
        'scaler_x': gp_result['scaler_x'],
        'scaler_y': gp_result['scaler_y'],
        'depth_range': gp_result['depth_range']
    }
    
    # Save the state dictionary
    torch.save(state_dict, filepath)
    print(f"GP model saved to {filepath}")


def load_gp_model(filepath: str):
    """
    Load a GP model and its associated components from a file using PyTorch.
    
    Args:
        filepath: Path from which the model should be loaded
    
    Returns:
        Dictionary containing the loaded GP model and related components
    """
    # Load the state dictionary
    state_dict = torch.load(filepath, map_location=DEVICE)
    
    # Retrieve stored parameters
    model_type = state_dict['model_type']
    num_tasks = state_dict['num_tasks']
    lengthscale_lower_bound = state_dict['lengthscale_lower_bound']
    
    # Create dummy tensors for initialization
    dummy_x = torch.randn(1, 1, device=DEVICE)
    dummy_y = torch.randn(1, device=DEVICE) if num_tasks == 1 else torch.randn(1, num_tasks, device=DEVICE)
    
    # Initialize model and likelihood based on type
    if model_type == 'GPModel':
        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(DEVICE)
        model = GPModel(dummy_x, dummy_y, likelihood, lengthscale_lower_bound).to(DEVICE)
    elif model_type == 'MultitaskGPModel':
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks).to(DEVICE)
        model = MultitaskGPModel(dummy_x, dummy_y, likelihood, lengthscale_lower_bound, num_tasks).to(DEVICE)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load the actual parameters
    model.load_state_dict(state_dict['model_state_dict'])
    likelihood.load_state_dict(state_dict['likelihood_state_dict'])
    
    # Create the result dictionary
    gp_result = {
        'model': model,
        'likelihood': likelihood,
        'scaler_x': state_dict['scaler_x'],
        'scaler_y': state_dict['scaler_y'],
        'depth_range': state_dict['depth_range']
    }
    
    print(f"GP model loaded from {filepath}")
    return gp_result