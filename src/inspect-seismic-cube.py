import numpy as np
import util.filehandler as fh
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def read_surface_dat(filepath, il_col=4, xl_col=5, t0_col=6):
    data = np.loadtxt(
        filepath,
        comments="#",
        usecols=(il_col - 1, xl_col - 1, t0_col - 1),
        encoding="latin1",
    )
    il = data[:, 0].astype(int)
    xl = data[:, 1].astype(int)
    t0 = data[:, 2]
    return il, xl, t0


def plot_seismic_with_surface(
    traces,
    surface_il,
    surface_xl,
    surface_t0,
    il_range,
    xl_range,
    dt=2,
    sample_offset=800,
):
    n_il, n_xl, n_samples = traces.shape
    il_vals = np.arange(il_range[0], il_range[0] + n_il)

    surface_grid = np.full((n_il, n_xl), np.nan)
    for i, (il, xl, t0) in enumerate(zip(surface_il, surface_xl, surface_t0)):
        if il_range[0] <= il < il_range[1] and xl_range[0] <= xl < xl_range[1]:
            il_idx = il - il_range[0]
            xl_idx = xl - xl_range[0]
            sample_idx = int(t0 / dt) - sample_offset
            surface_grid[il_idx, xl_idx] = sample_idx

    fig, ax = plt.subplots(figsize=(12, 8))
    plt.subplots_adjust(left=0.25, bottom=0.25)

    initial_idx = 0
    slice_data = traces[initial_idx, :, :].T

    im = ax.imshow(
        slice_data,
        aspect="auto",
        cmap="gray",
        vmin=-np.max(np.abs(slice_data)),
        vmax=np.max(np.abs(slice_data)),
    )

    surface_line = surface_grid[initial_idx, :]
    valid_xl = ~np.isnan(surface_line)
    if np.any(valid_xl):
        ax.plot(
            np.where(valid_xl)[0],
            surface_line[valid_xl],
            "r-",
            linewidth=2,
            label="Surface T0",
        )

    ax.set_title(f"Inline {il_vals[initial_idx]}")
    ax.set_xlabel("Crossline")
    ax.set_ylabel("Sample")
    cbar = fig.colorbar(im, ax=ax)

    ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
    slider = Slider(ax_slider, "Inline", 0, n_il - 1, valinit=initial_idx, valfmt="%d")

    def update(val):
        idx = int(slider.val)
        slice_data = traces[idx, :, :].T
        im.set_data(slice_data)

        for line in ax.lines[:]:
            line.remove()

        surface_line = surface_grid[idx, :]
        valid_xl = ~np.isnan(surface_line)
        if np.any(valid_xl):
            ax.plot(
                np.where(valid_xl)[0],
                surface_line[valid_xl],
                "r-",
                linewidth=2,
                label="Surface T0",
            )

        ax.set_title(f"Inline {il_vals[idx]}")
        fig.canvas.draw_idle()

    slider.on_changed(update)
    ax.legend()
    plt.show()

    return fig, slider


def main():
    segy_file = "../data/Vankorskaya_s_p_5_03-04_Migrirovannyiy_PreStack.sgy"
    surface_file = "/mnt/Documents/MSU/Diploma/Data/х Фактические данные/Куб, интерпретация, ГИС/Интерпретация/Vankorskaya_s_p_5_03-04_T0_(korrelyatsiya_OG)_Dannyie_T0_po_seysmicheskim_kubam_3D_Krovlya_plasta_Nh-III-IV_v_nizhney_chasti_nizhnehetskoy_svityi.dat"

    print("reading segy")
    il_range = (5110, 5510)
    xl_range = (1100, 1500)
    traces, il_segy, xl_segy = fh.read_sgy_selective(segy_file, il_range, xl_range)
    # np.save("../data/vankor_il-5110-5510_xl-1100-1500.npy", traces)
    # np.save("../data/vankor-ilines.npy", il_segy)
    # np.save("../data/vankor-xlines.npy", xl_segy)
    traces = traces[:, :, 800:1300]

    print("reading surface")
    surface_il, surface_xl, surface_t0 = read_surface_dat(
        surface_file, il_col=4, xl_col=5, t0_col=6
    )

    dt = 2
    sample_offset = 800
    plot_seismic_with_surface(
        traces,
        surface_il,
        surface_xl,
        surface_t0,
        il_range,
        xl_range,
        dt=dt,
        sample_offset=sample_offset,
    )


if __name__ == "__main__":
    main()
