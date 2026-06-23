import numpy as np
import pandas as pd
from src.processing.voxels import downsample_and_profile_voxels

def test_empty_coordinates():
    """Verifies that passing empty coordinate arrays returns an empty DataFrame with the correct columns."""
    df = downsample_and_profile_voxels(
        np.array([]), np.array([]), np.array([]),
        voxel_size=2.0
    )
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0
    expected_cols = ['voxel_x', 'voxel_y', 'voxel_z', 'point_count', 'mean_z', 'std_z', 'min_z', 'max_z']
    assert list(df.columns) == expected_cols

def test_voxel_downsampling_and_aggregates():
    """Verifies that voxel indices are correctly calculated and elevations are aggregated."""
    x = np.array([2.5, 3.5, 10.5])
    y = np.array([2.5, 3.5, 10.5])
    z = np.array([1.0, 3.0, 5.0])
    
    # voxel_size = 5.0
    # indices:
    # 2.5/5.0 = 0.5 -> floor is 0
    # 10.5/5.0 = 2.1 -> floor is 2
    # So we expect two voxels: (0, 0, 0) containing 2.5, 3.5 (elevations 1.0, 3.0) and (2, 2, 1) containing 10.5 (elevation 5.0)
    df = downsample_and_profile_voxels(x, y, z, voxel_size=5.0, filter_outliers=False)
    
    assert len(df) == 2
    
    # Sort to ensure predictable checks
    df = df.sort_values(by=['voxel_x', 'voxel_y', 'voxel_z']).reset_index(drop=True)
    
    # First Voxel
    assert df.loc[0, 'voxel_x'] == 0
    assert df.loc[0, 'voxel_y'] == 0
    assert df.loc[0, 'voxel_z'] == 0
    assert df.loc[0, 'point_count'] == 2
    assert df.loc[0, 'mean_z'] == 2.0  # (1.0 + 3.0)/2
    assert df.loc[0, 'min_z'] == 1.0
    assert df.loc[0, 'max_z'] == 3.0
    assert df.loc[0, 'std_z'] == np.std([1.0, 3.0], ddof=1) # pandas uses ddof=1 by default
    
    # Second Voxel
    assert df.loc[1, 'voxel_x'] == 2
    assert df.loc[1, 'voxel_y'] == 2
    assert df.loc[1, 'voxel_z'] == 1
    assert df.loc[1, 'point_count'] == 1
    assert df.loc[1, 'mean_z'] == 5.0
    assert df.loc[1, 'std_z'] == 0.0  # Single point NaN filled with 0.0

def test_outlier_filtering(sample_point_data):
    """Verifies that the absolute range and 3-sigma statistical outlier filters successfully clean noise."""
    x = sample_point_data["x"]
    y = sample_point_data["y"]
    z = sample_point_data["z"]
    
    # 1. Run without outlier filtering
    df_nofilter = downsample_and_profile_voxels(x, y, z, voxel_size=2.0, filter_outliers=False)
    total_raw_points = df_nofilter['point_count'].sum()
    assert total_raw_points == len(z)  # All points preserved
    
    # Verify that extreme values (-100, 200) are present
    assert df_nofilter['min_z'].min() == -100.0
    assert df_nofilter['max_z'].max() == 200.0
    
    # 2. Run with outlier filtering active
    df_filter = downsample_and_profile_voxels(x, y, z, voxel_size=2.0, filter_outliers=True)
    total_filtered_points = df_filter['point_count'].sum()
    
    # Filtered count should equal the expected clean points count (outliers pruned)
    assert total_filtered_points == sample_point_data["clean_count"]
    
    # Outliers should be pruned
    assert df_filter['min_z'].min() > -50.0
    assert df_filter['max_z'].max() < 150.0
    # The statistical outlier of 20.0m is more than 3std from mean ~11.0m, so it should be pruned too
    assert df_filter['max_z'].max() < 15.0
