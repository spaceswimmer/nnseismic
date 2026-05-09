import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.ticker import FuncFormatter
from pathlib import Path


def read_surface_dat(filepath, il_col=4, xl_col=5, t0_col=6, name_col=7):
    data = np.loadtxt(
        filepath,
        comments="#",
        usecols=(il_col - 1, xl_col - 1, t0_col - 1, name_col - 1),
        encoding="latin1",
        dtype=str,
    )
    il = data[:, 0].astype(int)
    xl = data[:, 1].astype(int)
    t0 = data[:, 2].astype(float)
    name = data[:, 3]
    return il, xl, t0, name


def read_surface_folder(folder_path, il_col=4, xl_col=5, t0_col=6, name_col=7):
    folder = Path(folder_path)
    dat_files = sorted(folder.glob("*.dat"))
    surfaces = []
    for fpath in dat_files:
        il, xl, t0, names = read_surface_dat(fpath, il_col, xl_col, t0_col, name_col)
        horizon_name = names[0]
        surfaces.append({"il": il, "xl": xl, "t0": t0, "name": horizon_name})
        print(f"  loaded {fpath.name}: {len(il)} points, horizon={horizon_name}")
    return surfaces


def plot_seismic_with_surface(
    traces,
    surfaces,
    il_range,
    xl_range,
    dt=2,
    sample_offset=0,
    t0_range=None,
):
    n_il, n_xl, n_samples = traces.shape
    il_vals = np.arange(il_range[0], il_range[0] + n_il)

    surfaces = sorted(surfaces, key=lambda s: np.mean(s["t0"]))

    surface_grids = []
    for surf in surfaces:
        grid = np.full((n_il, n_xl), np.nan)
        for il, xl, t0 in zip(surf["il"], surf["xl"], surf["t0"]):
            if il_range[0] <= il < il_range[1] and xl_range[0] <= xl < xl_range[1]:
                il_idx = il - il_range[0]
                xl_idx = xl - xl_range[0]
                sample_idx = int(t0 / dt) - sample_offset
                grid[il_idx, xl_idx] = sample_idx
        surface_grids.append(grid)

    fig, ax = plt.subplots(figsize=(10, 14))
    plt.subplots_adjust(left=0.25, bottom=0.25, right=0.7)

    def t0_formatter(x, pos):
        return f"{x * dt + sample_offset:.0f}"

    ax.yaxis.set_major_formatter(FuncFormatter(t0_formatter))

    initial_idx = 0
    slice_data = traces[initial_idx, :, :].T

    im = ax.imshow(
        slice_data,
        aspect="auto",
        cmap="gray",
        vmin=-np.max(np.abs(slice_data)),
        vmax=np.max(np.abs(slice_data)),
    )

    colors = plt.cm.tab20b(np.linspace(0, 1, len(surfaces)))
    for grid, surf, color in zip(surface_grids, surfaces, colors):
        surface_line = grid[initial_idx, :]
        valid_xl = ~np.isnan(surface_line)
        if np.any(valid_xl):
            ax.plot(
                np.where(valid_xl)[0],
                surface_line[valid_xl],
                color=color,
                linewidth=2,
                label=surf["name"],
            )

    ax.set_title(f"Inline {il_vals[initial_idx]}")
    ax.set_xlabel("Crossline")
    ax.set_ylabel("T0, ms")
    if t0_range is not None:
        t0_min_samp = t0_range[0] / dt - sample_offset
        t0_max_samp = t0_range[1] / dt - sample_offset
        ax.set_ylim(t0_max_samp, t0_min_samp)

    ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
    slider = Slider(ax_slider, "Inline", 0, n_il - 1, valinit=initial_idx, valfmt="%d")

    def update(val):
        idx = int(slider.val)
        slice_data = traces[idx, :, :].T
        im.set_data(slice_data)

        for line in ax.lines[:]:
            line.remove()

        for grid, surf, color in zip(surface_grids, surfaces, colors):
            surface_line = grid[idx, :]
            valid_xl = ~np.isnan(surface_line)
            if np.any(valid_xl):
                ax.plot(
                    np.where(valid_xl)[0],
                    surface_line[valid_xl],
                    color=color,
                    linewidth=2,
                    label=surf["name"],
                )

        ax.set_title(f"Inline {il_vals[idx]}")
        fig.canvas.draw_idle()

    slider.on_changed(update)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    plt.show()

    return fig, slider


def main():
    traces_file = "../data/vankor_il-5110-5510_xl-1100-1500.npy"
    surface_folder = (
        "/mnt/Documents/MSU/Diploma/Data/х Фактические данные/"
        "Куб, интерпретация, ГИС/Интерпретация/"
    )

    print("loading traces")
    traces = np.load(traces_file)
    traces = traces[:, :, :2500]

    print("reading surfaces from folder...")
    surfaces = read_surface_folder(surface_folder)

    dt = 2
    sample_offset = 0
    plot_seismic_with_surface(
        traces,
        surfaces,
        il_range=(5110, 5510),
        xl_range=(1100, 1500),
        dt=dt,
        sample_offset=sample_offset,
        t0_range=(600, 3700),
    )


if __name__ == "__main__":
    main()
