"""
3D Seismic Cube Viewer using Plotly
Interactive visualization of seismic volumes with orthogonal slices
"""

import numpy as np
import plotly.graph_objects as go

# ============================================
# CONFIGURATION - Change this filepath
# ============================================
SEISMIC_FILEPATH = "../data/seismicCubes_cumsum_fullstack_2026.33698898.npy"

# Subsampling for performance (set to 1 for full resolution, higher for faster rendering)
SUBSAMPLE_FACTOR = 2

# Colorscale for seismic data
COLORSCALE = "gray"

# Opacity for the slices (0-1)
SLICE_OPACITY = 0.9

# Depth scaling factor (to squish the vertical dimension)
# 1.0 = no scaling, 0.25 = depth axis will be 25% of original height
DEPTH_SCALE = 0.9

CBAR_TITLE = 'Amplitude'
CUBE_NAME = '3D Seismic Cube'


def load_seismic_data(filepath, subsample=1):
    """Load and subsample seismic data"""
    print(f"Loading seismic data from: {filepath}")
    data = np.load(filepath)

    if subsample > 1:
        data = data[::subsample, ::subsample, ::subsample]
        print(f"Subsampled to shape: {data.shape}")
    else:
        print(f"Data shape: {data.shape}")

    return data


def normalize_data(data):
    """Normalize data to 0-1 range for better visualization"""
    p2, p98 = np.percentile(data, [2, 98])
    data_clipped = np.clip(data, p2, p98)
    data_normalized = (data_clipped - data_clipped.min()) / (
        data_clipped.max() - data_clipped.min()
    )
    return data_normalized, p2, p98


def create_3d_cube_visualization(seismic_data):
    """Create interactive 3D visualization with three orthogonal slices

    Data orientation: seismic_data[inline, xline, depth]
    """

    data_norm, p2, p98 = normalize_data(seismic_data)
    n_inline, n_xline, n_depth = data_norm.shape

    print(f"Creating 3D visualization...")
    print(f"Cube dimensions: Inline={n_inline}, Xline={n_xline}, Depth={n_depth}")

    fig = go.Figure()

    inline_pos = 0#n_inline // 2
    xline_pos = 0#n_xline // 2
    depth_pos = n_depth - n_depth // 4

    # Slice 1: Inline slice (fixed inline, shows xline-depth plane)
    inline_slice = data_norm[inline_pos, :, :]
    inline_y, inline_z_unscaled = np.meshgrid(
        np.arange(n_xline), np.arange(n_depth), indexing="ij"
    )
    inline_x = np.full_like(inline_y, inline_pos)
    inline_z = inline_z_unscaled * DEPTH_SCALE

    fig.add_trace(
        go.Surface(
            x=inline_x,
            y=inline_y,
            z=inline_z,
            surfacecolor=inline_slice,
            colorscale=COLORSCALE,
            opacity=SLICE_OPACITY,
            showscale=False,
            name=f"Inline {inline_pos}",
        )
    )

    # Slice 2: Xline slice (fixed xline, shows inline-depth plane)
    xline_slice = data_norm[:, xline_pos, :]
    xline_x, xline_z_unscaled = np.meshgrid(
        np.arange(n_inline), np.arange(n_depth), indexing="ij"
    )
    xline_y = np.full_like(xline_x, xline_pos)
    xline_z = xline_z_unscaled * DEPTH_SCALE

    fig.add_trace(
        go.Surface(
            x=xline_x,
            y=xline_y,
            z=xline_z,
            surfacecolor=xline_slice,
            colorscale=COLORSCALE,
            opacity=SLICE_OPACITY,
            showscale=False,
            name=f"Xline {xline_pos}",
        )
    )

    # Slice 3: Depth slice (fixed depth, shows inline-xline plane)
    depth_slice = data_norm[:, :, depth_pos]
    depth_x, depth_y = np.meshgrid(
        np.arange(n_inline), np.arange(n_xline), indexing="ij"
    )
    depth_z = np.full_like(depth_x, depth_pos * DEPTH_SCALE)

    fig.add_trace(
        go.Surface(
            x=depth_x,
            y=depth_y,
            z=depth_z,
            surfacecolor=depth_slice,
            colorscale=COLORSCALE,
            opacity=SLICE_OPACITY,
            showscale=True,
            name=f"Depth {depth_pos}",
            colorbar=dict(title=CBAR_TITLE, x=1.02, len=0.8),
        )
    )

    max_depth_scaled = n_depth * DEPTH_SCALE

    fig.update_layout(
        title=dict(
            text=CUBE_NAME + f"<br><sub>Inline: {n_inline}, Xline: {n_xline}, Depth: {n_depth}</sub>",
            x=0.5,
            xanchor="center",
        ),
        scene=dict(
            xaxis=dict(title="Inline", backgroundcolor="rgb(230, 230, 230)"),
            yaxis=dict(title="Xline", backgroundcolor="rgb(230, 230, 230)"),
            zaxis=dict(
                title="Depth",
                backgroundcolor="rgb(230, 230, 230)",
                autorange="reversed",
                range=[max_depth_scaled, 0],
            ),
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=DEPTH_SCALE),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
        ),
        width=1200,
        height=900,
        margin=dict(l=0, r=0, t=80, b=0),
    )

    return fig


def main():
    """Main function to run the viewer"""
    print("=" * 60)
    print("3D Seismic Cube Viewer")
    print("=" * 60)

    seismic = load_seismic_data(SEISMIC_FILEPATH, subsample=SUBSAMPLE_FACTOR)
    fig = create_3d_cube_visualization(seismic)

    print("\nDisplaying interactive 3D visualization...")
    print("Controls:")
    print("  - Left-click + drag: Rotate view")
    print("  - Right-click + drag: Zoom")
    print("  - Middle-click + drag: Pan")
    print("  - Hover over slice: See coordinates and amplitude")
    print("  - Click legend items: Toggle slice visibility")
    print("=" * 60)

    fig.show()

    save_html = input("\nSave as HTML? (y/n): ").lower().strip()
    if save_html == "y":
        output_path = SEISMIC_FILEPATH.replace(".npy", "_3d_viewer.html")
        fig.write_html(output_path)
        print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
