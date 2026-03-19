import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from IPython.display import display
import ipywidgets as widgets

def plot_well_logs(df, columns, well_name=None, figsize=None, depth_range=None):
    """
    Plot well log curves from a DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with depth as index and log curves as columns
    columns : list
        List of column names to plot
    well_name : str, optional
        Name of the well for the title
    figsize : tuple, optional
        Figure size (width, height). Auto-calculated if None
    depth_range : tuple, optional
        (min_depth, max_depth) to limit the plot range
    
    Special columns:
    ----------------
    LITH : Lithology marker (discrete values 0-4)
    NAS : NAS marker (discrete values 0-5, 9)
    """
    # Marker columns that should be plotted as discrete/categorical
    marker_columns = ['LITH', 'NAS']
    
    # Color maps for marker columns
    lith_colors = {0: "#2FFC07FF", 1: '#FFD700', 2: '#8B4513', 3: "#B88C13", 4: "#000000"}
    nas_colors = {
        0: '#FFFFFF',    # white
        1: '#0000FF',    # blue
        2: '#8B4513',    # brown
        3: '#FFFF00',    # yellow
        4: '#90EE90',    # light-green
        5: '#006400',    # dark-green
        9: '#800080'     # purple
    }
    
    # Filter depth range if specified
    if depth_range is not None:
        df_plot = df.loc[depth_range[0]:depth_range[1]]
    else:
        df_plot = df
    
    depth = df_plot.index.values
    n_plots = len(columns)
    
    # Create width ratios: marker columns are narrower
    width_ratios = [0.2 if col in marker_columns else 1.0 for col in columns]
    
    # Auto-calculate figure size
    if figsize is None:
        # Optional: reduce total width since marker columns are smaller
        total_ratio = sum(width_ratios)
        width = 2.5 * total_ratio  # instead of 2.5 * n_plots
        height = 12
        figsize = (width, height)
    
    
    # Create subplots with varying widths
    fig, axes = plt.subplots(1, n_plots, figsize=figsize, sharey=True,
                              gridspec_kw={'width_ratios': width_ratios})
    if n_plots == 1:
        axes = [axes]
    
    fig.suptitle(f'Well: {well_name}' if well_name else 'Well Logs', fontsize=14, fontweight='bold')
    
    for ax, col in zip(axes, columns):
        if col not in df_plot.columns:
            ax.text(0.5, 0.5, f'{col}\nnot found', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(col)
            continue
        
        data = df_plot[col].values
        
        if col in marker_columns:
            # Plot marker columns as filled discrete intervals
            ax.set_title(col, fontweight='bold')
            
            # Choose color map based on column
            color_map = lith_colors if col == 'LITH' else nas_colors
            
            # Plot as step function with colored fills
            unique_vals = np.unique(data[~np.isnan(data)])
            
            for val in unique_vals:
                if val in color_map:
                    mask = data == val
                    # Fill areas where this value occurs
                    ax.fill_betweenx(depth, 0, 1, where=mask, 
                                    color=color_map[val], alpha=0.8, 
                                    label=f'{int(val)}')
            
            ax.set_xlim(0, 1)
            ax.set_xticks([])
            ax.legend(loc='upper right', fontsize=8, title=col)
            
        else:
            # Plot regular log curves as lines
            ax.plot(data, depth, 'b-', linewidth=0.8)
            ax.set_title(col, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Auto-scale with some padding
            valid_data = data[~np.isnan(data)]
            if len(valid_data) > 0:
                vmin, vmax = np.nanmin(valid_data), np.nanmax(valid_data)
                padding = 0.1 * (vmax - vmin) if vmax > vmin else 1
                ax.set_xlim(vmin - padding, vmax + padding)
    
    # Set common y-axis
    axes[0].set_ylabel('Depth (m)', fontsize=12)
    axes[0].invert_yaxis()  # Depth increases downward
    
    plt.tight_layout()
    return fig, axes

def plot_3d_array_interactive(array, axis='z'):
    """
    Plot a 3D numpy array interactively in a Jupyter notebook.
    
    Parameters:
    array (numpy.ndarray): 3D numpy array with shape (x, y, z)
    axis (str): The axis to slice along ('x', 'y', or 'z'). Default is 'z'.
               This determines which dimension will be cycled through.
    """
    if len(array.shape) != 3:
        raise ValueError("Array must be 3-dimensional")
        
    if axis not in ['x', 'y', 'z']:
        raise ValueError("axis must be 'x', 'y', or 'z'")
    
    # Get dimensions
    x_dim, y_dim, z_dim = array.shape
    
    # Determine which axis to slice along
    if axis == 'x':
        slice_idx_max = x_dim
        plot_dims = (1, 2)  # Show y-z plane
        title_axis = 'X'
    elif axis == 'y':
        slice_idx_max = y_dim
        plot_dims = (0, 2)  # Show x-z plane
        title_axis = 'Y'
    else:  # axis == 'z'
        slice_idx_max = z_dim
        plot_dims = (0, 1)  # Show x-y plane
        title_axis = 'Z'
    
    def update_plot(idx):
        plt.figure(figsize=(8, 6))
        if axis == 'x':
            slice_data = array[idx, :, :]
        elif axis == 'y':
            slice_data = array[:, idx, :]
        else:  # axis == 'z'
            slice_data = array[:, :, idx]
            
        plt.imshow(slice_data, aspect='auto', cmap='viridis')
        plt.title(f'{title_axis}-slice at index {idx}')
        plt.colorbar()
        plt.show()
    
    # Create interactive slider widget
    slider = widgets.IntSlider(
        value=0,
        min=0,
        max=slice_idx_max - 1,
        step=1,
        description=f'{title_axis}-index:',
        continuous_update=True
    )
    
    # Create interactive widget
    interactive_plot = widgets.interactive(update_plot, idx=slider)
    
    # Display the interactive plot
    display(interactive_plot)


def plot_3d_array_with_slider(array, axis='z'):
    """
    Alternative implementation using matplotlib sliders for interactivity.
    This approach works well outside of Jupyter as well.
    
    Parameters:
    array (numpy.ndarray): 3D numpy array with shape (x, y, z)
    axis (str): The axis to slice along ('x', 'y', or 'z'). Default is 'z'.
               This determines which dimension will be cycled through.
    """
    if len(array.shape) != 3:
        raise ValueError("Array must be 3-dimensional")
        
    if axis not in ['x', 'y', 'z']:
        raise ValueError("axis must be 'x', 'y', or 'z'")
    
    # Get dimensions
    x_dim, y_dim, z_dim = array.shape
    
    # Determine which axis to slice along
    if axis == 'x':
        slice_idx_max = x_dim
        plot_dims = (1, 2)  # Show y-z plane
        title_axis = 'X'
    elif axis == 'y':
        slice_idx_max = y_dim
        plot_dims = (0, 2)  # Show x-z plane
        title_axis = 'Y'
    else:  # axis == 'z'
        slice_idx_max = z_dim
        plot_dims = (0, 1)  # Show x-y plane
        title_axis = 'Z'
    
    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.subplots_adjust(bottom=0.25)
    
    # Initial slice
    initial_idx = 0
    if axis == 'x':
        initial_slice = array[initial_idx, :, :]
    elif axis == 'y':
        initial_slice = array[:, initial_idx, :]
    else:  # axis == 'z'
        initial_slice = array[:, :, initial_idx]
    
    im = ax.imshow(initial_slice, aspect='auto', cmap='viridis')
    ax.set_title(f'{title_axis}-slice at index {initial_idx}')
    fig.colorbar(im, ax=ax)
    
    # Create slider axes
    ax_slider = plt.axes([0.2, 0.1, 0.5, 0.03])
    slider = Slider(ax_slider, f'{title_axis}-index', 0, slice_idx_max - 1, valinit=initial_idx, valfmt='%d')
    
    # Define update function
    def update(val):
        idx = int(slider.val)
        ax.clear()
        if axis == 'x':
            slice_data = array[idx, :, :]
        elif axis == 'y':
            slice_data = array[:, idx, :]
        else:  # axis == 'z'
            slice_data = array[:, :, idx]
            
        im = ax.imshow(slice_data, aspect='auto', cmap='viridis')
        ax.set_title(f'{title_axis}-slice at index {idx}')
        fig.canvas.draw()
    
    # Connect slider to update function
    slider.on_changed(update)
    
    plt.show()
    
    return fig, slider

def plot_gp_model(gp_result, property_col, ax=None):
    """
    Plot GP model with original data.
    
    Parameters:
    -----------
    gp_result : dict
        Output from fit_gp_model()
    property_col : str
        Property name (for labels)
    ax : matplotlib axis object
        If None, creates new figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    model = gp_result['model']
    
    # Denormalize the stored training data
    X_orig = model.X_train_.flatten()
    y_orig = model.y_train_ * model._y_train_std + model._y_train_mean
    
    # Generate smooth curve from GP
    x_smooth = np.linspace(X_orig.min(), X_orig.max(), 200).reshape(-1, 1)
    y_mean, y_std = model.predict(x_smooth, return_std=True)
    
    # Plot
    ax.scatter(X_orig, y_orig, alpha=0.6, label='Original Data', s=20)
    ax.plot(x_smooth, y_mean, 'r-', label='GP Mean', linewidth=2)
    ax.fill_between(x_smooth.flatten(), 
                   y_mean - 2*y_std, 
                   y_mean + 2*y_std, 
                   alpha=0.3, label='±2σ')
    
    ax.set_xlabel('Depth (m)')
    ax.set_ylabel(property_col)
    ax.set_title(f'GP Model: {property_col}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax