import numpy as np

def create_spatial_grid(x_size, z_size, dx):
    x = np.arange(-x_size/2, x_size/2 + dx, dx)
    z = np.arange(0, z_size + dx, dx)
    X, Z = np.meshgrid(x, z)
    return X, Z, x, z

def create_roi_masks(X, Z, roi_x_positions, vessel_cz, roi_size):
    half_size = roi_size / 2
    roi_masks = []

    for roi_x in roi_x_positions:
        mask = ((X >= roi_x - half_size) & (X <= roi_x + half_size) &
                (Z >= vessel_cz - half_size) & (Z <= vessel_cz + half_size))
        roi_masks.append(mask)

        if not mask.any():
            print(f"Warning: ROI at {roi_x} mm is empty")

    return roi_masks
