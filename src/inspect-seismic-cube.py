import numpy as np
import util.filehandler as fh
import util.plotting as pl
import matplotlib.pyplot as plt

def main():
    file = "/mnt/storage/nnseismic/real_data/Vankorskaya_s_p_5_03-04_Migrirovannyiy_PreStack.sgy"
    print('reading segy')
    il_range = (5110, 5510)
    xl_range = (1100, 1500)
    traces, _, _ = fh.read_sgy_selective(file, il_range, xl_range)
    pl.plot_3d_array_with_slider(traces, axis='x')
if __name__ == "__main__":
    main()
