import argparse
import os
import re
import numpy as np
import torch
from lw_spacenet import UNet3D
from dataset import SeismicDataset, sort_list_IDs
from glob import glob


def parse_position_from_filename(filename):
    """Parse position (i_j_k) from filename like '64_128_0.dat'."""
    basename = os.path.splitext(filename)[0]
    match = re.match(r"^(\d+)_(\d+)_(\d+)$", basename)
    if match:
        return tuple(int(x) for x in match.groups())
    return None


def predict(
    model_path, data_dir, output_dir, shape=(128, 128, 128, 1), preserve_position=False
):
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = UNet3D(in_channels=shape[3], out_channels=1, init_features=16)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device).bfloat16()
    model.eval()

    seis_dir = os.path.join(data_dir, "seis")
    if not os.path.exists(seis_dir):
        raise ValueError(f"Seismic data directory {seis_dir} does not exist")

    file_list = os.listdir(seis_dir)
    list_IDs = sort_list_IDs(file_list)

    dataset = SeismicDataset(
        root_dir=data_dir, list_IDs=list_IDs, shape=shape, only_load_input=True
    )

    for idx in range(len(dataset)):
        ID = list_IDs[idx]
        seismic_data = dataset[idx]
        seismic_data = seismic_data.unsqueeze(0).to(device).bfloat16()

        with torch.no_grad():
            prediction = model(seismic_data)

        prediction = prediction.squeeze().cpu().float().numpy()
        prediction = prediction.transpose((2, 1, 0))

        if preserve_position:
            pos = parse_position_from_filename(ID)
            if pos:
                output_name = f"{pos[0]}_{pos[1]}_{pos[2]}_rgt.dat"
            else:
                output_name = os.path.splitext(ID)[0] + "_rgt.dat"
        else:
            output_name = os.path.splitext(ID)[0] + "_rgt.dat"

        output_path = os.path.join(output_dir, output_name)
        prediction.astype(np.float32).tofile(output_path)
        print(f"Processed {ID} -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict RGT from seismic data")
    parser.add_argument(
        "--model", type=str, required=True, help="Path to model checkpoint (.pth)"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to data folder (should contain 'seis' subfolder)",
    )
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument(
        "--shape",
        type=int,
        nargs=4,
        default=[128, 128, 128, 1],
        help="Shape of input chunk (n1, n2, n3, n_channels)",
    )
    parser.add_argument(
        "--preserve-position",
        action="store_true",
        help="Preserve position from input filename in output filename (i_j_k_rgt.dat)",
    )

    args = parser.parse_args()

    predict(
        args.model, args.data, args.output, tuple(args.shape), args.preserve_position
    )
