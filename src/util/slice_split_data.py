import argparse
import os
import shutil
from glob import glob

import numpy as np
from sklearn.model_selection import train_test_split

# Configuration Constants
CHUNK_SIZE = (128, 128, 128)
STRIDE = (64, 64, 64)
SEIS_PATTERN = "seismicCubes_cumsum_fullstack*.npy"
AGE_PATTERN = "faulted_age*.npy"
SORT_KEY = lambda p: p.rsplit('2026', 1)[-1]

def normalize_seismic(data):
    """Min-max normalization to [0, 1]."""
    mn, mx = data.min(), data.max()
    return np.zeros_like(data) if mx == mn else (data - mn) / (mx - mn)

def normalize_rgt(data):
    """Z-score normalization."""
    m, s = data.mean(), data.std()
    return np.zeros_like(data) if s == 0 else (data - m) / s

def get_chunk_slices(data_shape, chunk_size, stride):
    """Calculate all chunk slice positions for a given data shape."""
    h, w, d = data_shape
    ch, cw, cd = chunk_size
    sh, sw, sd = stride
    
    h_steps = max(1, (h - ch) // sh + 1)
    w_steps = max(1, (w - cw) // sw + 1)
    d_steps = max(1, (d - cd) // sd + 1)
    
    slices = []
    for i in range(h_steps):
        for j in range(w_steps):
            for k in range(d_steps):
                end_h = min(i * sh + ch, h)
                end_w = min(j * sw + cw, w)
                end_d = min(k * sd + cd, d)
                start_h = max(0, end_h - ch)
                start_w = max(0, end_w - cw)
                start_d = max(0, end_d - cd)
                slices.append((slice(start_h, end_h), slice(start_w, end_w), slice(start_d, end_d)))           
    return slices

def save_chunks(data, slices, save_dir, start_idx, norm_fn):
    """Save chunks using pre-calculated slices, returns next index."""
    count = start_idx
    for sl in slices:
        chunk = norm_fn(data[sl])
        chunk.tofile(os.path.join(save_dir, f"{count:05d}.dat"))
        count += 1
    return count

def clean_path(path):
    """Remove file or directory at path."""
    if os.path.exists(path):
        if os.path.isfile(path):
            os.remove(path)
        else:
            shutil.rmtree(path)

def main():
    parser = argparse.ArgumentParser(description="Slice, normalize, and split seismic data.")
    parser.add_argument("--raw_data_path", type=str, required=True, help="Path to raw .npy files")
    parser.add_argument("--output_path_root", type=str, required=True, help="Root directory for train/val output")
    args = parser.parse_args()

    base = args.output_path_root
    dirs = {
        "chunks_seis": os.path.join(base, "chunks", "seis"),
        "chunks_rgt": os.path.join(base, "chunks", "rgt"),
        "train_seis": os.path.join(base, "train", "seis"),
        "train_rgt": os.path.join(base, "train", "rgt"),
        "val_seis": os.path.join(base, "val", "seis"),
        "val_rgt": os.path.join(base, "val", "rgt"),
    }

    # Clean ALL specific directories (handles files or folders)
    for d in dirs.values():
        clean_path(d)
        os.makedirs(d, exist_ok=True)
    
    # Also clean parent directories
    for parent in set(os.path.dirname(p) for p in dirs.values()):
        if os.path.exists(parent):
            shutil.rmtree(parent)
        os.makedirs(parent, exist_ok=True)
    
    # Recreate specific subdirs after parent cleanup
    for d in dirs.values():
        clean_path(d)
        os.makedirs(d, exist_ok=True)

    # Locate and sort files
    seis_paths = sorted(glob(os.path.join(args.raw_data_path, SEIS_PATTERN)), key=SORT_KEY)
    age_paths = sorted(glob(os.path.join(args.raw_data_path, AGE_PATTERN)), key=SORT_KEY)

    if len(seis_paths) != len(age_paths):
        raise ValueError("Mismatch: Number of seismic and age files must be equal.")
    if len(seis_paths) == 0:
        raise ValueError(f"No files found. Check path: {args.raw_data_path}")

    # Process all pairs with continuous global indexing
    global_idx = 0
    for idx, (s_p, a_p) in enumerate(zip(seis_paths, age_paths)):
        print(f"Processing pair {idx+1}/{len(seis_paths)}...")
        
        seis = np.load(s_p).astype(np.float32)[:,:,250:]
        age = np.load(a_p).astype(np.float32)[:,:,250:]
        
        # Calculate slices based on seismic shape, apply to both
        slices = get_chunk_slices(seis.shape, CHUNK_SIZE, STRIDE)
        
        # Save both with same slices and continuous global index
        next_idx = save_chunks(seis, slices, dirs["chunks_seis"], global_idx, normalize_seismic)
        next_idx = save_chunks(age, slices, dirs["chunks_rgt"], global_idx, normalize_rgt)
        
        del seis, age
        chunks_this_file = next_idx - global_idx
        global_idx = next_idx
        print(f"  Created {chunks_this_file} chunks (total: {global_idx})")

    # Verify both directories have identical files
    seis_files = sorted(os.listdir(dirs["chunks_seis"]))
    rgt_files = sorted(os.listdir(dirs["chunks_rgt"]))
    
    if seis_files != rgt_files:
        raise ValueError(f"File mismatch! Seis: {len(seis_files)}, Rgt: {len(rgt_files)}")
    
    files = seis_files
    print(f"Total chunks: {len(files)}")
    
    if len(files) == 0:
        raise ValueError("No chunks created. Check data shapes and patterns.")

    # Train/Val Split
    train_idx, val_idx = train_test_split(range(len(files)), test_size=0.1, random_state=42)

    def move_files(indices, src_s, src_r, dst_s, dst_r):
        for i in indices:
            fname = files[i]
            shutil.move(os.path.join(src_s, fname), dst_s)
            shutil.move(os.path.join(src_r, fname), dst_r)

    print(f"Splitting: {len(train_idx)} train, {len(val_idx)} val chunks.")
    
    print("Moving training data...")
    move_files(train_idx, dirs["chunks_seis"], dirs["chunks_rgt"], 
               dirs["train_seis"], dirs["train_rgt"])
    
    print("Moving validation data...")
    move_files(val_idx, dirs["chunks_seis"], dirs["chunks_rgt"], 
               dirs["val_seis"], dirs["val_rgt"])

    # Cleanup
    print("Cleaning intermediate chunks...")
    shutil.rmtree(os.path.join(base, "chunks"))
    print("Done.")

if __name__ == "__main__":
    main()