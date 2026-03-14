import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from IPython.display import display
import ipywidgets as widgets


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