# nnseismic

**nnseismic** is a repository containing code for deep learning on seismic data. It implements neural network models for predicting Relative Geological Time (RGT) from seismic cubes. Visualizations of some results are available in [`data/images`](data/images) (interactive 3D HTML viewers and static images).

## Repository Structure

- **[`src/RGTnet`](src/RGTnet)** — full implementation of the neural network from the paper [«3-D Seismic RGT...»](https://doi.org/10.1109/TGRS.2021.3126028). Includes the 3D architecture, training pipeline, and inference.
- **[`src/DNN`](src/DNN)** — a lightweight version of RGTnet, developed by me. Faster and simpler while maintaining acceptable prediction quality.
- **[`synthoseis/`](synthoseis)** — a submodule, fork of [sede-open/synthoseis](https://github.com/sede-open/synthoseis), hosted at [spaceswimmer/synthoseis](https://github.com/spaceswimmer/synthoseis). Used for generating synthetic seismic data.
- **[`data/`](data)** — seismic cubes, synthetic data, saved models, and images.
- **[`config/`](config)** — configuration files for running modeling.

## Cloning

```bash
git clone --recurse-submodules git@github.com:spaceswimmer/nnseismic.git
```

If the repository was already cloned without submodules:

```bash
git submodule update --init --recursive
```

## Running

The project uses [`uv`](https://docs.astral.sh/uv/) for dependency management (see [`pyproject.toml`](pyproject.toml) and [`uv.lock`](uv.lock)). Python ≥ 3.9 is required.

```bash
# Install dependencies
uv sync

# Launch Jupyter notebooks
uv run jupyter notebook src/

# Run python scripts
uv run python file.py
```

## Useful Links

- Main repository: [github.com/spaceswimmer/nnseismic](https://github.com/spaceswimmer/nnseismic)
- Synthoseis fork: [github.com/spaceswimmer/synthoseis](https://github.com/spaceswimmer/synthoseis)
- Original synthoseis: [github.com/sede-open/synthoseis](https://github.com/sede-open/synthoseis)
