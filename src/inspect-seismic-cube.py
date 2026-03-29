import numpy as np
import util.filehandler as fh
from util.plotting import plot_3d_array_with_slider

def main():
    folder = "../data/synthetic_data/run/seismic__2026.23045076_tagilsk_test_3/"
    file = "seismicCubes_RFC_fullstack_2026.23045076.npy"
    cube = np.load(folder+file, 'r')
    plot_3d_array_with_slider(cube, axis='x')

if __name__ == "__main__":
    main()
