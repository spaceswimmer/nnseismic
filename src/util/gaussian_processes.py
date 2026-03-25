import torch
import gpytorch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional, Tuple, Union

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GPModel(gpytorch.models.ExactGP):
    """Simple GP model with RBF kernel."""
    
    def __init__(self, train_x, train_y, likelihood, lengthscale):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                lengthscale_constraint=gpytorch.constraints.GreaterThan(lengthscale)
            )
        )
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, lengthscale, num_tasks):
        super().__init__(train_x, train_y, likelihood)
        self.num_tasks = num_tasks
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.LinearMean(input_size=1, bias=True), num_tasks=num_tasks
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(
                lengthscale_constraint=gpytorch.constraints.GreaterThan(lengthscale)
            ),
            num_tasks=num_tasks,
            rank=1
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


def fit_gp_model(
    df: pd.DataFrame,
    property_col: Union[str, List[str]],
    depth_col: str = 'DEPTH',
    training_iter: int = 100,
    max_points: Optional[int] = None,
    random_state: int = 42,
    lengthscale = 0.2
) -> Optional[dict]:
    """
    Fit a GP model to depth-property data.
    
    Args:
        df: DataFrame containing the data
        property_col: Column name(s) for the target property/properties
        depth_col: Column name for depth values
        training_iter: Number of training iterations
        max_points: Maximum number of points to use (subsampling)
        random_state: Random seed for reproducibility
        lengthscale: Minimum lengthscale constraint for RBF kernel
    
    Returns:
        Dict with: 'model', 'likelihood', 'scaler_x', 'scaler_y', 
                   'depth_range', 'lengthscale', 'num_tasks'
        Returns None if insufficient data.
    """
    # Normalize property_col to list for consistent handling
    if isinstance(property_col, str):
        property_cols = [property_col]
        num_tasks = 1
    else:
        property_cols = list(property_col)
        num_tasks = len(property_cols)
    
    cols_to_select = [depth_col] + property_cols
    
    # Drop rows with any NaN values in selected columns
    data = df[cols_to_select].dropna()
    
    if len(data) < 10:
        print(f"Warning: Only {len(data)} points for {property_cols}")
        return None
    
    # Subsample if needed
    if max_points is not None and len(data) > max_points:
        data = data.sample(n=max_points, random_state=random_state)
        print(f"{property_cols}: subsampled to {max_points} points")
    
    X = data[depth_col].values.reshape(-1, 1)
    
    # Get y values - always 2D for scaler compatibility
    if num_tasks == 1:
        y = data[property_cols[0]].values.reshape(-1, 1)
    else:
        y = data[property_cols].values
    
    # Use StandardScaler
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    X_norm = scaler_x.fit_transform(X)
    y_norm = scaler_y.fit_transform(y)
    
    # Convert to tensors
    train_x = torch.tensor(X_norm, dtype=torch.float32, device=DEVICE)
    
    # For single-task GP, train_y must be 1D; for multi-task, 2D
    if num_tasks == 1:
        train_y = torch.tensor(y_norm.flatten(), dtype=torch.float32, device=DEVICE)
    else:
        train_y = torch.tensor(y_norm, dtype=torch.float32, device=DEVICE)
    
    # Train GP with appropriate likelihood and model
    if num_tasks == 1:
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = GPModel(train_x, train_y, likelihood, lengthscale).to(DEVICE)
    else:
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks)
        model = MultitaskGPModel(train_x, train_y, likelihood, lengthscale, num_tasks).to(DEVICE)
    
    model.train()
    likelihood.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    
    # Format property name for progress display
    prop_display = property_cols[0] if num_tasks == 1 else str(property_cols)
    
    for i in tqdm(range(training_iter), desc=f"Training GP for {prop_display}"):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        if i % 20 == 0:
            print(f"Iteration {i}, Loss: {loss.item():.4f}")
        optimizer.step()
    
    print(f"{prop_display}: trained on {len(data)} points")
    
    return {
        'model': model,
        'likelihood': likelihood,
        'scaler_x': scaler_x,
        'scaler_y': scaler_y,
        'depth_range': (X.min(), X.max()),
        'lengthscale': lengthscale,
        'num_tasks': num_tasks
    }


def fit_gp_model_all_wells(
    las_dfs_dict: Dict[str, pd.DataFrame],
    property_col: Union[str, List[str]],
    depth_col: str = 'DEPTH',
    **kwargs
) -> Optional[dict]:
    """
    Fit a GP model on combined data from all wells.
    
    Args:
        las_dfs_dict: Dictionary mapping well names to DataFrames
        property_col: Column name(s) for the target property/properties
        depth_col: Column name for depth values
        **kwargs: Additional arguments passed to fit_gp_model
    
    Returns:
        Result dict from fit_gp_model, or None if no data found.
    """
    all_data = []
    
    # Normalize to list for consistent handling
    if isinstance(property_col, str):
        cols_to_check = [depth_col, property_col]
    else:
        cols_to_check = [depth_col] + list(property_col)
    
    for well_name, df in las_dfs_dict.items():
        if all(col in df.columns for col in cols_to_check):
            all_data.append(df[cols_to_check])
    
    if not all_data:
        print(f"No data found for {property_col}")
        return None
    
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"Using {len(combined_df)} total points for {property_col}")
    
    return fit_gp_model(combined_df, property_col, depth_col, **kwargs)


def predict_gp_model(
    gp_result: dict,
    x_new: np.ndarray,
    log_transform: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """Predict with GP model, handling normalization."""
    model = gp_result['model']
    likelihood = gp_result['likelihood']
    scaler_x = gp_result['scaler_x']
    scaler_y = gp_result['scaler_y']
    num_tasks = gp_result['num_tasks']
    
    # Normalize input
    x_new = np.asarray(x_new).reshape(-1, 1)
    x_norm = scaler_x.transform(x_new)
    x_tensor = torch.tensor(x_norm, dtype=torch.float32, device=DEVICE)
    
    model.eval()
    likelihood.eval()
    
    with torch.no_grad():
        preds = likelihood(model(x_tensor))
        mean_norm = preds.mean.cpu().numpy()
        std_norm = preds.stddev.cpu().numpy()
        del preds, x_tensor  # Clean up GPU tensors
    
    # Handle single vs multi-task output shapes for inverse transform
    if num_tasks == 1:
        mean = scaler_y.inverse_transform(mean_norm.reshape(-1, 1)).squeeze()
        std = std_norm * scaler_y.scale_[0]
    else:
        mean = scaler_y.inverse_transform(mean_norm)
        std = std_norm * scaler_y.scale_
    
    if log_transform:
        variance = std ** 2
        mean_original = np.exp(mean + variance / 2)
        std_original = np.sqrt((np.exp(variance) - 1) * np.exp(2 * mean + variance))
        return mean_original, std_original
    
    return mean, std

def sample_gp_model(
    gp_result: dict,
    x_new: np.ndarray,
    log_transform: bool = False,
    n_samples: int = 1
) -> np.ndarray:
    model = gp_result['model']
    likelihood = gp_result['likelihood']
    scaler_x = gp_result['scaler_x']
    scaler_y = gp_result['scaler_y']
    num_tasks = gp_result['num_tasks']
    device = gp_result.get('device', 'cpu')
    
    x_new = np.asarray(x_new).reshape(-1, 1)
    x_norm = scaler_x.transform(x_new)
    x_tensor = torch.tensor(x_norm, dtype=torch.float32, device=device)
    
    model.eval()
    likelihood.eval()
    
    with torch.no_grad():
        preds = likelihood(model(x_tensor))
        samples_norm = preds.sample(torch.Size([n_samples])).cpu().numpy()
        del preds, x_tensor
    
    # Rest of the function remains the same...
    if num_tasks == 1:
        samples_reshaped = samples_norm.reshape(-1, 1)
        samples = scaler_y.inverse_transform(samples_reshaped).reshape(n_samples, -1)
    else:
        n_points = samples_norm.shape[1]
        samples_reshaped = samples_norm.reshape(-1, num_tasks)
        samples = scaler_y.inverse_transform(samples_reshaped).reshape(n_samples, n_points, num_tasks)
    
    if log_transform:
        samples = np.exp(samples)
    
    return samples

def sample_gp_grid(
    gp_grid: dict,
    z_new: np.ndarray,
    n_samples: int = 1,
    interpolation: str = 'linear',
    linear_extrap_funcs: Optional[List[callable]] = None
) -> np.ndarray:
    """
    Fast sampling from pre-computed GP grid with interpolation.
    
    Args:
        gp_grid: Dictionary from load_gp_grid_params()
        z_new: Array of depth points OR list of arrays (one per task for multi-task GP)
        n_samples: Number of samples to generate
        interpolation: Interpolation method ('linear', 'nearest', 'cubic')
        linear_extrap_funcs: Optional list of functions for values outside grid range.
    
    Returns:
        For single-task: array of shape (n_samples, *z_new.shape)
        For multi-task with single z_new: array of shape (n_samples, *z_new.shape, num_tasks)
        For multi-task with list of z_new: array of shape (n_samples, z_new[i].shape, num_tasks)
    """
    from scipy.interpolate import interp1d
    
    mean = gp_grid['mean']
    L = gp_grid['L']
    depth_grid = gp_grid['depth_grid']
    num_tasks = gp_grid['num_tasks']
    log_transform = gp_grid['log_transform']
    
    depth_min = depth_grid.min()
    depth_max = depth_grid.max()
    
    # Handle z_new input format - flatten all arrays for processing
    if isinstance(z_new, (list, tuple)):
        z_arrays = [np.asarray(z).flatten() for z in z_new]
        z_shapes = [np.asarray(z).shape for z in z_new]
        separate_z_per_task = True
        if num_tasks == 1:
            raise ValueError("Single-task GP only accepts a single z_new array")
        if len(z_arrays) != num_tasks:
            raise ValueError(f"Expected {num_tasks} z arrays, got {len(z_arrays)}")
    else:
        z_arr_flat = np.asarray(z_new).flatten()
        z_arrays = [z_arr_flat]
        z_shapes = [np.asarray(z_new).shape]
        separate_z_per_task = False
    
    # Generate correlated samples on the full grid
    n_grid = len(depth_grid)
    z = np.random.standard_normal((n_samples, L.shape[0]))
    
    if num_tasks == 1:
        samples_grid = mean + (L @ z.T).T
        if log_transform:
            samples_grid = np.exp(samples_grid)
        
        z_arr = z_arrays[0]
        samples = np.zeros((n_samples, len(z_arr)))
        
        # Interpolate within bounds
        in_bounds = (z_arr >= depth_min) & (z_arr <= depth_max)
        if np.any(in_bounds):
            for i in range(n_samples):
                interp_func = interp1d(depth_grid, samples_grid[i], kind=interpolation, bounds_error=True)
                samples[i, in_bounds] = interp_func(z_arr[in_bounds])
        
        # Use linear functions outside bounds
        if linear_extrap_funcs is not None:
            out_bounds = ~in_bounds
            if np.any(out_bounds):
                samples[:, out_bounds] = linear_extrap_funcs[0](z_arr[out_bounds])
        
        # Reshape back to original shape
        samples = samples.reshape((n_samples, *z_shapes[0]))
        
    else:
        # Multi-task
        samples_grid_flat = mean.flatten() + (L @ z.T).T
        samples_grid = samples_grid_flat.reshape(n_samples, n_grid, num_tasks)
        
        if log_transform:
            samples_grid = np.exp(samples_grid)
        
        if separate_z_per_task:
            samples = [np.zeros((n_samples, len(z_arr))) for z_arr in z_arrays]
            
            for t, z_arr in enumerate(z_arrays):
                in_bounds = (z_arr >= depth_min) & (z_arr <= depth_max)
                
                if np.any(in_bounds):
                    for i in range(n_samples):
                        interp_func = interp1d(depth_grid, samples_grid[i, :, t], kind=interpolation, bounds_error=True)
                        samples[t][i, in_bounds] = interp_func(z_arr[in_bounds])
                
                if linear_extrap_funcs is not None:
                    out_bounds = ~in_bounds
                    if np.any(out_bounds):
                        samples[t][:, out_bounds] = linear_extrap_funcs[t](z_arr[out_bounds])
            
            # Reshape each task back to original shape
            samples_reshaped = [s.reshape((n_samples, *shape)) for s, shape in zip(samples, z_shapes)]
            
            # Stack to shape (n_samples, max_size, num_tasks) - for compatibility
            max_size = max(s.size // n_samples for s in samples)
            samples_stacked = np.zeros((n_samples, max_size, num_tasks))
            for t in range(num_tasks):
                flat = samples_reshaped[t].reshape(n_samples, -1)
                samples_stacked[:, :flat.shape[1], t] = flat
            samples = samples_stacked
            
        else:
            z_arr = z_arrays[0]
            samples = np.zeros((n_samples, len(z_arr), num_tasks))
            
            in_bounds = (z_arr >= depth_min) & (z_arr <= depth_max)
            
            for t in range(num_tasks):
                if np.any(in_bounds):
                    for i in range(n_samples):
                        interp_func = interp1d(depth_grid, samples_grid[i, :, t], kind=interpolation, bounds_error=True)
                        samples[i, in_bounds, t] = interp_func(z_arr[in_bounds])
                
                if linear_extrap_funcs is not None:
                    out_bounds = ~in_bounds
                    if np.any(out_bounds):
                        samples[:, out_bounds, t] = linear_extrap_funcs[t](z_arr[out_bounds])
            
            # Reshape back to original shape
            samples = samples.reshape((n_samples, *z_shapes[0], num_tasks))
    
    return samples


def predict_gp_grid(
    gp_grid: dict,
    x_new: np.ndarray,
    interpolation: str = 'linear'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fast prediction (mean and std) from pre-computed GP grid with interpolation.
    
    Args:
        gp_grid: Dictionary from load_gp_grid_params()
        x_new: 1D array of depth points to predict at
        interpolation: Interpolation method ('linear', 'nearest', 'cubic')
    
    Returns:
        mean: Predicted mean at x_new points
        std: Predicted standard deviation at x_new points
    """
    from scipy.interpolate import interp1d
    
    mean = gp_grid['mean']
    L = gp_grid['L']
    depth_grid = gp_grid['depth_grid']
    num_tasks = gp_grid['num_tasks']
    log_transform = gp_grid['log_transform']
    
    # Validate depth range
    if x_new.min() < depth_grid.min() or x_new.max() > depth_grid.max():
        raise ValueError(
            f"Requested depths [{x_new.min():.1f}, {x_new.max():.1f}] "
            f"outside pre-computed range [{depth_grid.min():.1f}, {depth_grid.max():.1f}]"
        )
    
    # Extract diagonal of covariance for std
    var_grid = np.sum(L ** 2, axis=1)  # More stable than diag(L @ L.T)
    std_grid = np.sqrt(var_grid)
    
    if num_tasks == 1:
        # Interpolate mean and std
        mean_interp = interp1d(depth_grid, mean, kind=interpolation)(x_new)
        std_interp = interp1d(depth_grid, std_grid, kind=interpolation)(x_new)
        
        if log_transform:
            # Adjust for log-normal distribution
            variance = std_interp ** 2
            mean_original = np.exp(mean_interp + variance / 2)
            std_original = np.sqrt((np.exp(variance) - 1) * np.exp(2 * mean_interp + variance))
            return mean_original, std_original
        
        return mean_interp, std_interp
    
    else:
        # Multi-task: reshape and interpolate each task
        n_grid = len(depth_grid)
        mean_grid = mean.reshape(n_grid, num_tasks)
        std_grid = std_grid.reshape(n_grid, num_tasks)
        
        mean_interp = np.zeros((len(x_new), num_tasks))
        std_interp = np.zeros((len(x_new), num_tasks))
        
        for t in range(num_tasks):
            mean_interp[:, t] = interp1d(depth_grid, mean_grid[:, t], kind=interpolation)(x_new)
            std_interp[:, t] = interp1d(depth_grid, std_grid[:, t], kind=interpolation)(x_new)
        
        if log_transform:
            variance = std_interp ** 2
            mean_original = np.exp(mean_interp + variance / 2)
            std_original = np.sqrt((np.exp(variance) - 1) * np.exp(2 * mean_interp + variance))
            return mean_original, std_original
        
        return mean_interp, std_interp