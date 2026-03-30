import numpy as np
import util.filehandler as fh
from util.plotting import plot_3d_array_with_slider

def main():
    folder = "..data/synthetic_data/run/seismic__2026.24282536_tagilst_test_6/"
    file = "seismicCubes_cumsum_RMO_24_degrees_normalized_augmented_2026.24282536.npy"
    cube = np.load(folder+file, 'r')
    plot_3d_array_with_slider(cube, axis='x')
    
if __name__ == "__main__":
    main()
