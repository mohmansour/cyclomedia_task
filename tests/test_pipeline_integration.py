import os
import json
import pytest
from unittest.mock import MagicMock, patch
import numpy as np

from src.pipeline.dag import Pipeline
from src.pipeline.tasks import FetchMetadataTask, ProcessChunksTask, SaveToStorageTask
from src.storage.database import DatabaseManager

# Create mock points structure
class MockPoints:
    def __init__(self, size=10):
        self.x = np.random.uniform(188000, 188100, size)
        self.y = np.random.uniform(1878900, 1879000, size)
        self.z = np.random.normal(11.0, 0.5, size)
    
    def __len__(self):
        return len(self.x)

@pytest.fixture
def mock_copc_reader():
    """Mocks the CopcReader class and its instances."""
    # Create the mock reader instance
    mock_reader = MagicMock()
    
    # Mock header metadata
    mock_reader.header.mins = [188000.0, 1878900.0, 0.0]
    mock_reader.header.maxs = [188200.0, 1879100.0, 50.0]
    mock_reader.header.point_count = 1000
    
    # Mock the query output
    mock_reader.query.return_value = MockPoints(10)
    
    # Mock context manager open() call
    with patch('src.pipeline.tasks.CopcReader.open') as mock_open:
        mock_open.return_value.__enter__.return_value = mock_reader
        yield mock_open, mock_reader

def test_pipeline_successful_e2e_run(mock_db_path, temp_parquet_dir, mock_copc_reader):
    """Verifies that the end-to-end pipeline executes successfully and populates DuckDB when all chunks download."""
    mock_open, mock_reader = mock_copc_reader
    
    # Set up tasks with small grid size (2x2 = 4 chunks) to run fast
    fetch_metadata = FetchMetadataTask(
        copc_url="https://s3.example.com/test.copc.laz",
        grid_size=2,
        voxel_size=2.0,
        db_path=mock_db_path,
        temp_dir=temp_parquet_dir,
        max_workers=2,
        filter_outliers=True,
        max_chunks=None
    )
    process_chunks = ProcessChunksTask()
    save_to_storage = SaveToStorageTask()
    
    pipeline = Pipeline()
    pipeline.add_task(fetch_metadata)
    pipeline.add_task(process_chunks)
    pipeline.add_task(save_to_storage)
    
    process_chunks.dependencies = ["FetchMetadata"]
    save_to_storage.dependencies = ["ProcessChunks"]
    
    # Run pipeline
    context = pipeline.run()
    
    # Verify execution output states
    assert "FetchMetadata" in context
    assert "ProcessChunks" in context
    assert "SaveToStorage" in context
    
    # Check that DuckDB was correctly populated
    with DatabaseManager(mock_db_path) as db:
        stats = db.get_stats()
        # 4 chunks * 10 points/chunk = 40 points total before downsampling/outliers
        assert stats["total_voxels"] > 0
        assert stats["total_points"] > 0
        
        # Verify metadata
        assert db.get_metadata("grid_size") == "2"
        assert db.get_metadata("voxel_size") == "2.0"
        assert db.get_metadata("failed_chunks") == "[]"

def test_pipeline_resilience_to_chunk_download_failures(mock_db_path, temp_parquet_dir, mock_copc_reader):
    """Verifies that individual chunk download failures do not abort the pipeline, and failures are recorded in metadata."""
    mock_open, mock_reader = mock_copc_reader
    
    # Define a custom query logic that raises an error for a specific chunk (e.g. chunk id '0_1')
    # Since query is called within process threads, we can inspect the query_box limits
    # and raise an error for a specific spatial partition.
    def query_side_effect(query_box):
        # Determine if this corresponds to a specific chunk partition
        # Grid bounds are X: [188000.0, 188200.0], Y: [1878900.0, 1879100.0]. Grid size is 2.
        # X range is divided into: [188000, 188100] (index 0) and [188100, 188200] (index 1)
        # Y range is divided into: [1878900, 1879000] (index 0) and [1879000, 1879100] (index 1)
        # So chunk 0_1 is X range [188000, 188100] and Y range [1879000, 1879100]
        mins = query_box.mins
        maxs = query_box.maxs
        
        # Check if query falls in chunk '0_1' (X is < 188100, Y is > 1879000)
        if mins[0] < 188100 and mins[1] >= 1879000:
            raise RuntimeError("Transient S3 Download Error on Chunk 0_1")

        return MockPoints(10)
        
    mock_reader.query.side_effect = query_side_effect
    
    # Initialize tasks
    fetch_metadata = FetchMetadataTask(
        copc_url="https://s3.example.com/test.copc.laz",
        grid_size=2,
        voxel_size=2.0,
        db_path=mock_db_path,
        temp_dir=temp_parquet_dir,
        max_workers=2,
        filter_outliers=True,
        max_chunks=None
    )
    process_chunks = ProcessChunksTask()
    save_to_storage = SaveToStorageTask()
    
    pipeline = Pipeline()
    pipeline.add_task(fetch_metadata)
    pipeline.add_task(process_chunks)
    pipeline.add_task(save_to_storage)
    
    process_chunks.dependencies = ["FetchMetadata"]
    save_to_storage.dependencies = ["ProcessChunks"]
    
    # Run pipeline
    context = pipeline.run()
    
    # Pipeline should succeed despite the failure in chunk 0_1!
    assert context["SaveToStorage"]["status"] == "success"
    
    # Check that failed chunk was logged in database metadata
    with DatabaseManager(mock_db_path) as db:
        failed_chunks_str = db.get_metadata("failed_chunks")
        failed_chunks = json.loads(failed_chunks_str)
        
        # Chunk 0_1 should be in the list of failed chunks
        assert "0_1" in failed_chunks
        
        # Other chunks (e.g. 0_0, 1_0, 1_1) should have loaded successfully
        stats = db.get_stats()
        assert stats["total_voxels"] > 0
        assert stats["total_points"] > 0

def test_pipeline_resume_failed(mock_db_path, temp_parquet_dir, mock_copc_reader):
    """Verifies that running in resume mode only downloads/processes previously failed chunks and appends them to storage."""
    mock_open, mock_reader = mock_copc_reader
    
    # 1. Initialize database state with some existing mock voxels and metadata showing '0_1' failed
    with DatabaseManager(mock_db_path) as db:
        db.conn.execute("""
            INSERT INTO processed_voxels VALUES
            (100, 200, 300, 50, 11.5, 0.2, 11.0, 12.0)
        """)
        db.set_metadata("voxel_size", "2.0")
        db.set_metadata("grid_size", "2")
        db.set_metadata("copc_url", "https://s3.example.com/test.copc.laz")
        db.set_metadata("point_count", "1000")
        db.set_metadata("filter_outliers", "True")
        db.set_metadata("failed_chunks", json.dumps(["0_1"]))
        
        # Verify initial stats: 1 voxel, 50 points
        initial_stats = db.get_stats()
        assert initial_stats["total_voxels"] == 1
        assert initial_stats["total_points"] == 50

    # 2. Run pipeline with resume_failed = True
    fetch_metadata = FetchMetadataTask(
        copc_url="https://s3.example.com/test.copc.laz",
        grid_size=2,
        voxel_size=2.0,
        db_path=mock_db_path,
        temp_dir=temp_parquet_dir,
        max_workers=2,
        filter_outliers=True,
        max_chunks=None,
        resume_failed=True
    )
    process_chunks = ProcessChunksTask()
    save_to_storage = SaveToStorageTask()
    
    pipeline = Pipeline()
    pipeline.add_task(fetch_metadata)
    pipeline.add_task(process_chunks)
    pipeline.add_task(save_to_storage)
    
    process_chunks.dependencies = ["FetchMetadata"]
    save_to_storage.dependencies = ["ProcessChunks"]
    
    context = pipeline.run()
    
    assert context["SaveToStorage"]["status"] == "success"
    
    # Check that ONLY chunk '0_1' was processed
    meta_result = context["FetchMetadata"]
    assert len(meta_result["chunks"]) == 1
    assert meta_result["chunks"][0]["id"] == "0_1"
    
    # 3. Check database to ensure data was appended (not overwritten)
    with DatabaseManager(mock_db_path) as db:
        stats = db.get_stats()
        # Voxel count should be initial (1) + whatever chunk 0_1 created (>0)
        assert stats["total_voxels"] > 1
        assert stats["total_points"] > 50
        
        # failed_chunks list should now be empty (since 0_1 processed successfully)
        failed_chunks_str = db.get_metadata("failed_chunks")
        assert json.loads(failed_chunks_str) == []

