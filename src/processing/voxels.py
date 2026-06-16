import numpy as np
import pandas as pd

def downsample_and_profile_voxels(
    x_coords: np.ndarray, 
    y_coords: np.ndarray, 
    z_coords: np.ndarray, 
    voxel_size: float,
    filter_outliers: bool = True,
    z_min_limit: float = -50.0,
    z_max_limit: float = 150.0
) -> pd.DataFrame:
    """
    Downsamples point cloud coordinates into a 3D voxel grid and computes elevation statistics.
    Optionally filters out elevation outliers and noise.
    
    Parameters:
        x_coords: 1D numpy array of X coordinates (floats)
        y_coords: 1D numpy array of Y coordinates (floats)
        z_coords: 1D numpy array of Z coordinates (floats)
        voxel_size: Size of the voxel cube in coordinate units
        filter_outliers: If True, applies 3-sigma and range filtering to Z coordinates
        z_min_limit: Absolute minimum elevation allowed
        z_max_limit: Absolute maximum elevation allowed
        
    Returns:
        pd.DataFrame
    """
    if len(x_coords) == 0:
        return pd.DataFrame(columns=[
            'voxel_x', 'voxel_y', 'voxel_z', 
            'point_count', 'mean_z', 'std_z', 'min_z', 'max_z'
        ])
        
    if filter_outliers:
        # Step 1: Physical range constraint
        in_range = (z_coords >= z_min_limit) & (z_coords <= z_max_limit)
        x_coords = x_coords[in_range]
        y_coords = y_coords[in_range]
        z_coords = z_coords[in_range]
        
        # Step 2: Statistical 3-sigma filter
        if len(z_coords) > 1:
            mean = np.mean(z_coords)
            std = np.std(z_coords)
            if std > 0.01:
                in_sigma = np.abs(z_coords - mean) <= 3.0 * std
                x_coords = x_coords[in_sigma]
                y_coords = y_coords[in_sigma]
                z_coords = z_coords[in_sigma]

    if len(x_coords) == 0:
        return pd.DataFrame(columns=[
            'voxel_x', 'voxel_y', 'voxel_z', 
            'point_count', 'mean_z', 'std_z', 'min_z', 'max_z'
        ])
    
    # Calculate voxel indices using floor division
    vx = np.floor(x_coords / voxel_size).astype(np.int64)
    vy = np.floor(y_coords / voxel_size).astype(np.int64)
    vz = np.floor(z_coords / voxel_size).astype(np.int64)
    
    # Create DataFrame for efficient grouping
    df = pd.DataFrame({
        'voxel_x': vx,
        'voxel_y': vy,
        'voxel_z': vz,
        'z': z_coords
    })
    
    # Group by voxel grid coordinates and aggregate
    grouped = df.groupby(['voxel_x', 'voxel_y', 'voxel_z']).agg(
        point_count=('z', 'count'),
        mean_z=('z', 'mean'),
        std_z=('z', 'std'),
        min_z=('z', 'min'),
        max_z=('z', 'max')
    ).reset_index()
    
    # Standard deviation is NaN for voxels with 1 point; fill with 0.0
    grouped['std_z'] = grouped['std_z'].fillna(0.0)
    
    return grouped
