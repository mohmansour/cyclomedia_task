import os
import pytest
from src.storage.database import DatabaseManager

def test_api_health(test_client, mock_db_path):
    """Verifies the health check endpoint response matches DB existence."""
    # Ensure the DB file is initialized and exists
    with DatabaseManager(mock_db_path) as db:
        pass
        
    res = test_client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "healthy"

    
    # Temporarily remove DB and check health degradation
    if os.path.exists(mock_db_path):
        os.remove(mock_db_path)
    res = test_client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "degraded"

def test_api_stats_db_missing(test_client, mock_db_path):
    """Verifies stats endpoint returns 404 if the database is missing."""
    if os.path.exists(mock_db_path):
        os.remove(mock_db_path)
    res = test_client.get("/stats")
    assert res.status_code == 404
    assert "Database file not found" in res.json()["detail"]

def test_api_stats_and_queries(test_client, mock_db_path):
    """Verifies stats, densest, and voxel bounding box query endpoints return valid data."""
    # Populate the database with test data
    with DatabaseManager(mock_db_path) as db:
        db.conn.execute("""
            INSERT INTO processed_voxels VALUES
            (10, 20, 30, 100, 60.0, 1.5, 58.0, 62.0),
            (11, 21, 31, 500, 62.0, 0.5, 61.0, 63.0),
            (12, 22, 32, 50, 64.0, 2.5, 60.0, 68.0)
        """)
        db.set_metadata("voxel_size", "2.0")
        db.set_metadata("grid_size", "8")
        db.set_metadata("copc_url", "https://s3.example.com/test.copc.laz")
        db.set_metadata("point_count", "100000")
        db.set_metadata("filter_outliers", "True")
        db.set_metadata("failed_chunks", "[]")

    # 1. Test /stats
    res = test_client.get("/stats")
    assert res.status_code == 200
    data = res.json()
    assert data["database_stats"]["total_voxels"] == 3
    assert data["database_stats"]["total_points"] == 650
    assert data["pipeline_config"]["voxel_size_meters"] == 2.0
    assert data["pipeline_config"]["copc_url"] == "https://s3.example.com/test.copc.laz"
    
    # 2. Test /densest (Highest point count first)
    res = test_client.get("/densest?limit=2")
    assert res.status_code == 200
    densest = res.json()
    assert len(densest) == 2
    # The one with 500 points (voxel_x=11) should be first
    assert densest[0]["voxel_x"] == 11
    assert densest[0]["point_count"] == 500
    assert densest[1]["voxel_x"] == 10
    assert densest[1]["point_count"] == 100

    # 3. Test /voxels querying using physical coordinates (which scale to indices by dividing by voxel_size=2.0)
    # voxel_x indices: 10, 11, 12 -> physical coordinates: 20m, 22m, 24m
    # min_x=21.0, max_x=23.0 -> index range: ceil(21.0/2) = 10 (floor is 10), so indices in range: 10, 11 (20/2=10, 22/2=11)
    # Actually, the scale formula in API is: 
    #   if min_x is not None: min_x = int(np.floor(min_x / voxel_size)) -> min_x = int(np.floor(21.0 / 2.0)) = 10
    #   if max_x is not None: max_x = int(np.floor(max_x / voxel_size)) -> max_x = int(np.floor(23.0 / 2.0)) = 11
    # Thus, query translates to: WHERE voxel_x >= 10 AND voxel_x <= 11
    res = test_client.get("/voxels?min_x=21.0&max_x=23.0")
    assert res.status_code == 200
    voxels = res.json()
    assert len(voxels) == 2
    assert voxels[0]["voxel_x"] == 10
    assert voxels[1]["voxel_x"] == 11

    # 4. Test /voxels querying using raw voxel coordinates directly
    res = test_client.get("/voxels?min_x=11&max_x=12&is_voxel_coords=true")
    assert res.status_code == 200
    voxels = res.json()
    assert len(voxels) == 2
    assert voxels[0]["voxel_x"] == 11
    assert voxels[1]["voxel_x"] == 12
