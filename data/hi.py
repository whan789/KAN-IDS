import h5py
import numpy as np

def convert_npz_to_h5(npz_path, h5_path):
    print(f"Loading {npz_path} to convert...")
    with np.load(npz_path) as data:
        X = data['X']
        T_relative = data['T_relative']
        y = data['y']

    print(f"Creating HDF5 file at {h5_path}...")
    with h5py.File(h5_path, 'w') as hf:
        hf.create_dataset('X', data=X, compression="gzip")
        hf.create_dataset('T_relative', data=T_relative, compression="gzip")
        hf.create_dataset('y', data=y, compression="gzip")
    print("Conversion complete.")

convert_npz_to_h5('/data/anom/final_data/DL_data_final/final_test_data_merged.npz', 'final_test_data.h5')