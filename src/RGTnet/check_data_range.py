"""Check data range for seismic and RGT data."""
import argparse
import numpy as np
import os


def check_range(dataroot):
    print(f"Checking data in: {dataroot}\n")
    
    for dtype in ['seis', 'rgt']:
        data_dir = os.path.join(dataroot, dtype)
        if not os.path.exists(data_dir):
            print(f"{dtype}: directory not found")
            continue
            
        files = os.listdir(data_dir)
        if not files:
            print(f"{dtype}: no files found")
            continue
        
        min_val, max_val = float('inf'), float('-inf')
        means, stds = [], []
        
        for f in files:
            data = np.fromfile(os.path.join(data_dir, f), dtype=np.float32)
            min_val = min(min_val, data.min())
            max_val = max(max_val, data.max())
            means.append(data.mean())
            stds.append(data.std())
        
        print(f"{dtype}:")
        print(f"  range: [{min_val:.4f}, {max_val:.4f}]")
        print(f"  mean: {np.mean(means):.4f}, std: {np.mean(stds):.4f}")
        print(f"  SSIM max_val: {max(abs(min_val), abs(max_val)):.4f}")
        print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Check data range for SSIM loss')
    parser.add_argument('--dataroot', 
                        default='/mnt/storage/nnseismic/synthetic_data/train',
                        help='path to data directory')
    args = parser.parse_args()
    check_range(args.dataroot)