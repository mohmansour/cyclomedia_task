import os
import logging
import numpy as np
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import RedirectResponse
from typing import Optional, List, Dict, Any

from src.storage.database import DatabaseManager

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FastAPIService")

# Determine DB Path from env or default
DB_PATH = os.getenv("DB_PATH", "point_cloud_pipeline.db")

app = FastAPI(
    title="Cyclomedia Point Cloud Voxel API",
    description="Microservice to serve, query, and analyze processed 3D point cloud voxels from DuckDB.",
    version="1.0.0"
)

@app.get("/", include_in_schema=False)
def index():
    # Redirect to Swagger UI docs
    return RedirectResponse(url="/docs")

@app.get("/health")
def health():
    """Simple API health check."""
    db_exists = os.path.exists(DB_PATH)
    return {
        "status": "healthy" if db_exists else "degraded",
        "database_connected": db_exists,
        "database_path": DB_PATH
    }

@app.get("/stats")
def get_stats():
    """
    Returns global statistics and configuration metadata about the processed point cloud dataset.
    """
    if not os.path.exists(DB_PATH):
        raise HTTPException(status_code=404, detail="Database file not found. Run the ingestion pipeline first.")
        
    with DatabaseManager(DB_PATH, read_only=True) as db:
        stats = db.get_stats()
        
        # Load additional metadata
        import json
        metadata = {
            "voxel_size_meters": float(db.get_metadata("voxel_size", "2.0")),
            "grid_size": int(db.get_metadata("grid_size", "8")),
            "copc_url": db.get_metadata("copc_url", ""),
            "original_point_count": int(db.get_metadata("point_count", "0")),
            "outliers_filtered": db.get_metadata("filter_outliers", "True") == "True",
            "failed_chunks": json.loads(db.get_metadata("failed_chunks", "[]"))
        }
        
    return {
        "database_stats": stats,
        "pipeline_config": metadata
    }

@app.get("/densest", response_model=List[Dict[str, Any]])
def get_densest(limit: int = Query(default=10, ge=1, le=100)):
    """
    Retrieves the densest voxels (highest point counts).
    """
    if not os.path.exists(DB_PATH):
        raise HTTPException(status_code=404, detail="Database file not found.")
        
    with DatabaseManager(DB_PATH, read_only=True) as db:
        res = db.conn.execute("""
            SELECT voxel_x, voxel_y, voxel_z, point_count, mean_z, std_z, min_z, max_z
            FROM processed_voxels
            ORDER BY point_count DESC
            LIMIT ?
        """, (limit,)).fetchall()
        
    columns = ["voxel_x", "voxel_y", "voxel_z", "point_count", "mean_z", "std_z", "min_z", "max_z"]
    return [dict(zip(columns, row)) for row in res]

@app.get("/voxels", response_model=List[Dict[str, Any]])
def query_voxels(
    min_x: Optional[float] = None,
    max_x: Optional[float] = None,
    min_y: Optional[float] = None,
    max_y: Optional[float] = None,
    min_z: Optional[float] = None,
    max_z: Optional[float] = None,
    is_voxel_coords: bool = Query(
        default=False, 
        description="If true, coordinates are treated as raw voxel indices. If false, they are treated as physical coordinates (meters) and scaled automatically."
    ),
    limit: int = Query(default=1000, ge=1, le=50000)
):
    """
    Queries voxels within a 3D bounding box.
    """
    if not os.path.exists(DB_PATH):
        raise HTTPException(status_code=404, detail="Database file not found.")
        
    # Scale physical coordinates to voxel indices if necessary
    if not is_voxel_coords:
        with DatabaseManager(DB_PATH, read_only=True) as db:
            voxel_size = float(db.get_metadata("voxel_size", "2.0"))
            
        if min_x is not None: min_x = int(np.floor(min_x / voxel_size))
        if max_x is not None: max_x = int(np.floor(max_x / voxel_size))
        if min_y is not None: min_y = int(np.floor(min_y / voxel_size))
        if max_y is not None: max_y = int(np.floor(max_y / voxel_size))
        if min_z is not None: min_z = int(np.floor(min_z / voxel_size))
        if max_z is not None: max_z = int(np.floor(max_z / voxel_size))
    else:
        # Convert floats to ints for voxel indices
        if min_x is not None: min_x = int(min_x)
        if max_x is not None: max_x = int(max_x)
        if min_y is not None: min_y = int(min_y)
        if max_y is not None: max_y = int(max_y)
        if min_z is not None: min_z = int(min_z)
        if max_z is not None: max_z = int(max_z)

    query_parts = []
    params = []
    
    if min_x is not None:
        query_parts.append("voxel_x >= ?")
        params.append(min_x)
    if max_x is not None:
        query_parts.append("voxel_x <= ?")
        params.append(max_x)
    if min_y is not None:
        query_parts.append("voxel_y >= ?")
        params.append(min_y)
    if max_y is not None:
        query_parts.append("voxel_y <= ?")
        params.append(max_y)
    if min_z is not None:
        query_parts.append("voxel_z >= ?")
        params.append(min_z)
    if max_z is not None:
        query_parts.append("voxel_z <= ?")
        params.append(max_z)
        
    where_clause = " AND ".join(query_parts) if query_parts else "1=1"
    
    with DatabaseManager(DB_PATH, read_only=True) as db:
        query = f"""
            SELECT voxel_x, voxel_y, voxel_z, point_count, mean_z, std_z, min_z, max_z
            FROM processed_voxels
            WHERE {where_clause}
            ORDER BY voxel_x, voxel_y, voxel_z
            LIMIT ?
        """
        # Append limit parameter securely
        query_params = params + [limit]
        logger.info(f"Executing Query: {query} with params {query_params}")
        res = db.conn.execute(query, query_params).fetchall()
        
    columns = ["voxel_x", "voxel_y", "voxel_z", "point_count", "mean_z", "std_z", "min_z", "max_z"]
    return [dict(zip(columns, row)) for row in res]
