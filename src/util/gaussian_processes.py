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
            gpytorch.means.ConstantMean(), num_tasks=num_tasks
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
    """
    Predict with GP model, handling normalization.
    
    Args:
        gp_result: Result dict from fit_gp_model
        x_new: Array of depth values to predict at
        log_transform: If True, apply inverse log transform to predictions
    
    Returns:
        Tuple of (mean, std) predictions at the specified depths
    """
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
    
    # Handle single vs multi-task output shapes for inverse transform
    if num_tasks == 1:
        # Single task: mean_norm is 1D, reshape for scaler
        mean = scaler_y.inverse_transform(mean_norm.reshape(-1, 1)).squeeze()
        std = std_norm * scaler_y.scale_[0]
    else:
        # Multi-task: mean_norm is 2D (n_samples, num_tasks)
        mean = scaler_y.inverse_transform(mean_norm)
        std = std_norm * scaler_y.scale_
    
    if log_transform:
        variance = std ** 2
        mean_original = np.exp(mean + variance / 2)
        std_original = np.sqrt((np.exp(variance) - 1) * np.exp(2 * mean + variance))
        return mean_original, std_original
    
    return mean, std