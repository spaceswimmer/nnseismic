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
    
    Returns dict with: 'model', 'likelihood', 'scaler_x', 'scaler_y', 'depth_range'
    """
    # Extract valid data
    if isinstance(property_col, str):
        cols_to_select = [depth_col, property_col]
        num_tasks = 1
    else:
        cols_to_select = [depth_col] + property_col
        num_tasks = len(property_col)
    
    data = df[cols_to_select].dropna()
    
    if len(data) < 10:
        print(f"Warning: Only {len(data)} points for {property_col}")
        return None
    
    # Subsample if needed
    if max_points is not None and len(data) > max_points:
        data = data.sample(n=max_points, random_state=random_state)
        print(f"{property_col}: subsampled to {max_points} points")
    
    X = data[depth_col].values.reshape(-1, 1)
    y = data[property_col].values
    
    # Use StandardScaler
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    X_norm = scaler_x.fit_transform(X)
    y_norm = scaler_y.fit_transform(y)
    
    # Convert to tensors
    train_x = torch.tensor(X_norm, dtype=torch.float32, device=DEVICE)
    train_y = torch.tensor(y_norm, dtype=torch.float32, device=DEVICE)
    
    # Train GP
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks) if num_tasks > 1 else gpytorch.likelihoods.GaussianLikelihood()

    if num_tasks == 1:
        model = GPModel(train_x, train_y, likelihood, lengthscale).to(DEVICE)
    else:
        model = MultitaskGPModel(train_x, train_y, likelihood, lengthscale, num_tasks).to(DEVICE)
    
    model.train()
    likelihood.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    
    for i in tqdm(range(training_iter), desc=f"Training GP for {property_col}"):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        if i % 5 == 0:  # Print every 10 iterations
            print(f"Iteration {i}, Loss: {loss.item():.4f}")
        optimizer.step()
    
    print(f"{property_col}: trained on {len(data)} points")
    
    return {
        'model': model,
        'likelihood': likelihood,
        'scaler_x': scaler_x,
        'scaler_y': scaler_y,
        'depth_range': (X.min(), X.max())
    }


def fit_gp_model_all_wells(
    las_dfs_dict: Dict[str, pd.DataFrame],
    property_col: Union[str, List[str]],
    depth_col: str = 'DEPTH',
    **kwargs
) -> Optional[dict]:
    """Fit a GP model on combined data from all wells."""
    all_data = []
    
    for well_name, df in las_dfs_dict.items():
        if isinstance(property_col, str):
            cols_to_check = [depth_col, property_col]
        else:
            cols_to_check = [depth_col] + property_col
            
        if all(col in df.columns for col in cols_to_check):
            all_data.append(df[cols_to_check])
    
    if not all_data:
        print(f"No data found for {property_col}")
        return None
    
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"Using {len(combined_df)} total points for {property_col}")
    
    return fit_gp_model(combined_df, property_col, depth_col, **kwargs)


def predict_gp_model(gp_result: dict, x_new: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Predict with GP model, handling normalization."""
    model = gp_result['model']
    likelihood = gp_result['likelihood']
    scaler_x = gp_result['scaler_x']
    scaler_y = gp_result['scaler_y']
    
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
    
    # Denormalize outputs
    mean = scaler_y.inverse_transform(mean_norm).squeeze()
    std = std_norm * scaler_y.scale_
    
    return mean, std