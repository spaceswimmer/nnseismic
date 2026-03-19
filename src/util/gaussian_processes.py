import torch
import gpytorch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union

DTYPE = torch.bfloat16
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GPModel(gpytorch.models.ExactGP):
    """Simple GP model with RBF kernel."""
    
    def __init__(self, train_x: torch.Tensor, train_y: torch.Tensor, likelihood: gpytorch.likelihoods.Likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    
    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def normalize_data(data: torch.Tensor) -> Tuple[torch.Tensor, float, float]:
    """Normalize data to mean=0, std=1. Returns (normalized_data, mean, std)."""
    mean = data.mean().item()
    std = data.std().item()
    if std == 0:
        std = 1.0  # Avoid division by zero
    normalized = (data - mean) / std
    return normalized, mean, std


def train_gp_model(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    training_iter: int = 100,
    learning_rate: float = 0.1
) -> Tuple[GPModel, gpytorch.likelihoods.GaussianLikelihood, dict]:
    """
    Train a GP model on the given data with normalization.
    
    Parameters
    ----------
    train_x : torch.Tensor
        Input features (depth), shape (n, 1)
    train_y : torch.Tensor
        Target values, shape (n,)
    training_iter : int
        Number of training iterations
    learning_rate : float
        Learning rate for Adam optimizer
    
    Returns
    -------
    model, likelihood, norm_params
    """
    
    # Move data to GPU and convert to float32
    train_x = train_x.to(DEVICE).float()
    train_y = train_y.to(DEVICE).float()

    # Normalize both inputs and outputs
    train_x_norm, x_mean, x_std = normalize_data(train_x)
    train_y_norm, y_mean, y_std = normalize_data(train_y)
    
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = GPModel(train_x_norm, train_y_norm, likelihood).to(DEVICE)
    
    model.train()
    likelihood.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    
    for i in range(training_iter):
        optimizer.zero_grad()
        output = model(train_x_norm)
        loss = -mll(output, train_y_norm)
        loss.backward()
        optimizer.step()
    
    norm_params = {'x_mean': x_mean, 'x_std': x_std, 'y_mean': y_mean, 'y_std': y_std}
    return model, likelihood, norm_params


def fit_gp_model(
    df: pd.DataFrame,
    property_cols: Union[str, List[str]],
    depth_col: str = 'DEPTH',
    training_iter: int = 100,
    max_points: Optional[int] = None,
    random_state: int = 42
) -> Dict[str, dict]:
    """
    Fit GP models for one or more properties.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with depth and property columns
    property_cols : str or list of str
        Column names to fit GP models for (e.g., ['PL_GG', 'VP', 'VS'])
    depth_col : str
        Column name for depth values
    training_iter : int
        Number of training iterations
    max_points : int, optional
        Maximum number of points to use (random subsampling if exceeded)
    random_state : int
        Random seed for reproducible subsampling
    
    Returns
    -------
    dict : {property_name: {'model': model, 'likelihood': likelihood, 'norm_params': dict, 'depth_range': (min, max)}}
    """
    if isinstance(property_cols, str):
        property_cols = [property_cols]
    
    results = {}
    
    for prop in property_cols:
        if prop not in df.columns:
            print(f"Warning: {prop} not found in DataFrame")
            continue
        
        # Extract valid data
        valid_mask = df[prop].notna() & df[depth_col].notna()
        data = df.loc[valid_mask, [depth_col, prop]]
        
        if len(data) < 10:
            print(f"Warning: Only {len(data)} points for {prop}")
            continue
        
        # Subsample if needed
        if max_points is not None and len(data) > max_points:
            data = data.sample(n=max_points, random_state=random_state)
            print(f"{prop}: subsampled from {len(df.loc[valid_mask])} to {max_points} points")
        
        # Convert to tensors
        train_x = torch.tensor(data[depth_col].values, dtype=DTYPE, device=DEVICE).unsqueeze(-1)
        train_y = torch.tensor(data[prop].values, dtype=DTYPE, device=DEVICE)
        
        # Train
        model, likelihood, norm_params = train_gp_model(train_x, train_y, training_iter)
        
        # Store results
        results[prop] = {
            'model': model,
            'likelihood': likelihood,
            'norm_params': norm_params,
            'depth_range': (data[depth_col].min(), data[depth_col].max()),
            'train_x': train_x,
            'train_y': train_y
        }
        print(f"{prop}: trained on {len(data)} points, depth range {results[prop]['depth_range']}")
    
    return results


def fit_gp_model_all_wells(
    las_dfs_dict: Dict[str, pd.DataFrame],
    property_cols: Union[str, List[str]],
    depth_col: str = 'DEPTH',
    training_iter: int = 100,
    max_points: Optional[int] = None,
    random_state: int = 42
) -> Dict[str, dict]:
    """
    Fit GP models on combined data from all wells.
    
    Parameters
    ----------
    las_dfs_dict : dict
        Dictionary of {well_name: DataFrame}
    property_cols : str or list of str
        Column names to fit GP models for
    depth_col : str
        Column name for depth values
    training_iter : int
        Number of training iterations
    max_points : int, optional
        Maximum number of points to use (random subsampling if exceeded)
    random_state : int
        Random seed for reproducible subsampling
    
    Returns
    -------
    dict : {property_name: {'model': model, 'likelihood': likelihood, 'norm_params': dict, 'depth_range': (min, max)}}
    """
    if isinstance(property_cols, str):
        property_cols = [property_cols]
    
    # Combine all wells into single DataFrame
    all_dfs = []
    for well_name, df in las_dfs_dict.items():
        cols_to_keep = [c for c in [depth_col] + property_cols if c in df.columns]
        all_dfs.append(df[cols_to_keep])
    
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    return fit_gp_model(combined_df, property_cols, depth_col, training_iter, max_points, random_state)


def predict_gp_model(
    gp_result: dict,
    x_new: Union[np.ndarray, torch.Tensor]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict with GP model, handling normalization.
    
    Parameters
    ----------
    gp_result : dict
        Result from fit_gp_model()
    x_new : array-like
        New depth values to predict at
    
    Returns
    -------
    mean, std : numpy arrays
    """
    model = gp_result['model']
    likelihood = gp_result['likelihood']
    norm_params = gp_result['norm_params']
    
    # Convert and normalize input
    if isinstance(x_new, np.ndarray):
        x_new = torch.tensor(x_new, dtype=torch.float32)
    x_new = x_new.to(DEVICE).unsqueeze(-1).float()
    
    # Normalize using training stats
    x_new_norm = (x_new - norm_params['x_mean']) / norm_params['x_std']
    
    model.eval()
    likelihood.eval()
    
    with torch.no_grad():
        preds = likelihood(model(x_new_norm))
        mean_norm = preds.mean
        std_norm = preds.stddev
    
    # Denormalize outputs
    mean = mean_norm * norm_params['y_std'] + norm_params['y_mean']
    std = std_norm * norm_params['y_std']  # Scale std but don't shift
    
    return mean.cpu().numpy(), std.cpu().numpy()
