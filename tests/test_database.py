import os
import duckdb
import pandas as pd
import pytest
from src.storage.database import DatabaseManager

def test_database_initialization_and_metadata(mock_db_path):
    """Verifies table creation and metadata storage (get/set) operations."""
    with DatabaseManager(mock_db_path) as db:
        # Check tables are initialized
        tables = db.conn.execute("SHOW TABLES").fetchall()
        table_names = [r[0] for r in tables]
        assert "processed_voxels" in table_names
        assert "pipeline_metadata" in table_names
        
        # Test metadata keys
        db.set_metadata("test_key", "test_val")
        assert db.get_metadata("test_key") == "test_val"
        
        # Test default fallback for missing key
        assert db.get_metadata("missing_key", "default_val") == "default_val"

def test_reset_storage_and_bulk_insert(mock_db_path):
    """Verifies that reset wipes data, and bulk_insert loads dataframes correctly."""
    df = pd.DataFrame({
        'voxel_x': [1, 2],
        'voxel_y': [10, 20],
        'voxel_z': [100, 200],
        'point_count': [5, 15],
        'mean_z': [100.5, 200.5],
        'std_z': [0.1, 0.2],
        'min_z': [100.0, 200.0],
        'max_z': [101.0, 201.0]
    })
    
    with DatabaseManager(mock_db_path) as db:
        # Insert records
        db.bulk_insert_voxels(df)
        
        # Verify row counts
        count = db.conn.execute("SELECT COUNT(*) FROM processed_voxels").fetchone()[0]
        assert count == 2
        
        # Reset storage
        db.reset_storage()
        
        # Count should be 0
        count_after = db.conn.execute("SELECT COUNT(*) FROM processed_voxels").fetchone()[0]
        assert count_after == 0

def test_index_creation_and_stats(mock_db_path):
    """Verifies index is successfully created and database stats aggregation matches expected values."""
    df = pd.DataFrame({
        'voxel_x': [5, 10],
        'voxel_y': [50, 100],
        'voxel_z': [500, 1000],
        'point_count': [10, 20],
        'mean_z': [500.5, 1000.5],
        'std_z': [0.5, 0.6],
        'min_z': [500.0, 1000.0],
        'max_z': [501.0, 1001.0]
    })
    
    with DatabaseManager(mock_db_path) as db:
        db.bulk_insert_voxels(df)
        
        # Create indexes
        db.create_indexes()
        
        # Query duckdb catalogs to confirm index exists
        indices = db.conn.execute("SELECT index_name FROM duckdb_indexes()").fetchall()
        index_names = [r[0] for r in indices]
        assert "idx_voxel_coords" in index_names
        
        # Validate stats output
        stats = db.get_stats()
        assert stats["total_voxels"] == 2
        assert stats["total_points"] == 30 # 10 + 20
        assert stats["x_range"] == (5, 10)
        assert stats["y_range"] == (50, 100)
        assert stats["z_range"] == (500, 1000)
