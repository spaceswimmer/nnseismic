from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
import numpy as np
import pandas as pd

def fit_gp_model_sklearn(df, property_col, random_state=42):
    """
    Fit a Gaussian Process model to depth-property data.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with depth as index
    property_col : str
        Column name (e.g., 'Vp', 'Vs', 'PL_GG')
    random_state : int
        Random seed for reproducibility
    
    Returns:
    --------
    dict with keys:
        'model' : fitted GPR model
        'depth_range' : (min_depth, max_depth)
        'r2_score' : model fit quality
    """
    # Extract data, drop NaNs
    data = df[[property_col]].dropna()
    X = data.index.values.reshape(-1, 1)  # depth as 2D array
    y = data[property_col].values
    
    if len(X) < 10:
        print(f"Warning: Only {len(X)} data points for {property_col}")
        return None
    
    # Define kernel: RBF (smooth trend) + WhiteKernel (noise)
    kernel = ConstantKernel(1.0) * RBF(length_scale=100.0) + WhiteKernel(noise_level=1.0)
    
    # Fit GPR
    gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=random_state)
    gpr.fit(X, y)
    
    # Calculate R² score on training data
    y_pred = gpr.predict(X)
    r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - y.mean())**2)
    
    depth_min, depth_max = X.min(), X.max()
    
    print(f"{property_col}: R² = {r2:.3f}, kernel = {gpr.kernel_}")
    
    return {
        'model': gpr,
        'depth_range': (depth_min, depth_max),
        'r2_score': r2
    }

def fit_gp_model_all_wells_sklearn(las_dfs_dict, property_col, max_points=2000, random_state=42):
    """
    Fit a GP model on subsampled data from all wells.
    """
    all_data = []
    
    for well_name, df in las_dfs_dict.items():
        if property_col in df.columns:
            subset = df[[property_col]].dropna()
            all_data.append(subset)
    
    if not all_data:
        print(f"No data found for {property_col}")
        return None
    
    combined_df = pd.concat(all_data)
    
    
    print(f"Using {len(combined_df)} points for {property_col}")
    
    return fit_gp_model(combined_df, property_col, random_state)