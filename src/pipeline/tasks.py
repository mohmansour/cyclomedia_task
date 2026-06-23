import os
import shutil
import logging
import laspy
from laspy.copc import CopcReader
import pandas as pd
import numpy as np
from typing import Dict, Any

from src.pipeline.dag import Task
from src.processing.voxels import downsample_and_profile_voxels
from src.storage.database import DatabaseManager

logger = logging.getLogger("PipelineTasks")

class FetchMetadataTask(Task):
    """
    Task 1: Connects to the remote COPC file, fetches global metadata, 
    and divides the 2D bounding box into a grid of spatial chunks.
    """
    def __init__(self, copc_url: str, grid_size: int, voxel_size: float, db_path: str, temp_dir: str, max_workers: int = 4, filter_outliers: bool = True, max_chunks: int = None, resume_failed: bool = False):
        super().__init__(name="FetchMetadata", retries=3, retry_delay=5.0)
        self.copc_url = copc_url
        self.grid_size = grid_size
        self.voxel_size = voxel_size
        self.db_path = db_path
        self.temp_dir = temp_dir
        self.max_workers = max_workers
        self.filter_outliers = filter_outliers
        self.max_chunks = max_chunks
        self.resume_failed = resume_failed

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        logger.info(f"Connecting to COPC dataset: {self.copc_url}")
        
        with CopcReader.open(self.copc_url) as reader:
            mins = reader.header.mins
            maxs = reader.header.maxs
            point_count = reader.header.point_count
            
            logger.info(f"Loaded COPC header metadata.")
            logger.info(f"Point count: {point_count:,}")
            logger.info(f"Bounding Box Mins: {mins}")
            logger.info(f"Bounding Box Maxs: {maxs}")
            
        # Extract X & Y bounds to divide spatially
        xmin, ymin, zmin = mins
        xmax, ymax, zmax = maxs
        
        x_step = (xmax - xmin) / self.grid_size
        y_step = (ymax - ymin) / self.grid_size
        
        chunks = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                chunk_xmin = xmin + i * x_step
                chunk_xmax = xmin + (i + 1) * x_step
                chunk_ymin = ymin + j * y_step
                chunk_ymax = ymin + (j + 1) * y_step
                
                chunks.append({
                    "id": f"{i}_{j}",
                    "xmin": chunk_xmin,
                    "xmax": chunk_xmax,
                    "ymin": chunk_ymin,
                    "ymax": chunk_ymax
                })
                
        logger.info(f"Created {len(chunks)} spatial chunks grid ({self.grid_size}x{self.grid_size}).")
        
        # If resume_failed is active, load failed chunks from database and filter
        if self.resume_failed:
            import json
            if not os.path.exists(self.db_path):
                raise FileNotFoundError(f"Database file '{self.db_path}' not found. Cannot resume ingestion.")
                
            with DatabaseManager(self.db_path, read_only=True) as db:
                failed_chunks_str = db.get_metadata("failed_chunks", "[]")
                failed_chunk_ids = json.loads(failed_chunks_str)
                
            if not failed_chunk_ids:
                logger.info("No failed chunks found in database metadata. Ingestion is already complete.")
                # We return an empty chunk list, subsequent task will know
                chunks = []
            else:
                chunks = [c for c in chunks if c["id"] in failed_chunk_ids]
                logger.info(f"Resume Mode: Filtered to {len(chunks)} previously failed chunks: {failed_chunk_ids}")
        
        # Prepare intermediate temp directory
        if os.path.exists(self.temp_dir):
            logger.info(f"Clearing existing temporary directory: {self.temp_dir}")
            shutil.rmtree(self.temp_dir)
        os.makedirs(self.temp_dir, exist_ok=True)
        
        return {
            "copc_url": self.copc_url,
            "voxel_size": self.voxel_size,
            "grid_size": self.grid_size,
            "db_path": self.db_path,
            "temp_dir": self.temp_dir,
            "chunks": chunks,
            "point_count": point_count,
            "mins": mins,
            "maxs": maxs,
            "max_workers": self.max_workers,
            "filter_outliers": self.filter_outliers,
            "max_chunks": self.max_chunks,
            "resume_failed": self.resume_failed
        }



def _process_single_chunk(chunk: Dict[str, Any], copc_url: str, voxel_size: float, temp_dir: str, filter_outliers: bool) -> tuple:
    """
    Helper function run within ThreadPoolExecutor to download and process one chunk.
    Includes chunk-level retry logic with exponential backoff.
    """
    import time
    chunk_id = chunk["id"]
    query_box = laspy.Bounds(
        mins=[chunk["xmin"], chunk["ymin"]],
        maxs=[chunk["xmax"], chunk["ymax"]]
    )
    
    max_retries = 3
    base_delay = 1.0
    points = None
    last_err = None
    
    for attempt in range(1, max_retries + 1):
        try:
            # Open reader inside thread so each HTTP request is self-contained and parallelized
            with CopcReader.open(copc_url) as reader:
                points = reader.query(query_box)
            break
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                delay = base_delay * (2 ** (attempt - 1))
                logger.warning(
                    f"Chunk {chunk_id} download failed (Attempt {attempt}/{max_retries}). "
                    f"Retrying in {delay:.1f}s... Error: {str(e)}"
                )
                time.sleep(delay)
            else:
                logger.error(f"Chunk {chunk_id} download failed after {max_retries} attempts.")
                raise last_err
        
    num_points = len(points)
    if num_points > 0:
        x_arr = np.array(points.x)
        y_arr = np.array(points.y)
        z_arr = np.array(points.z)
        
        voxels_df = downsample_and_profile_voxels(
            x_arr, y_arr, z_arr, 
            voxel_size, 
            filter_outliers=filter_outliers
        )
        num_voxels = len(voxels_df)
        
        out_path = os.path.join(temp_dir, f"chunk_{chunk_id}.parquet")
        voxels_df.to_parquet(out_path, index=False)
        return out_path, num_points, num_voxels
    return None, 0, 0



class ProcessChunksTask(Task):
    """
    Task 2: Iterates over the grid of spatial chunks. Streams points in parallel
    for each chunk, downsamples them into voxels, and saves the output 
    as an intermediate Parquet file. Keeps memory footprint low and bounded.
    """
    def __init__(self):
        super().__init__(name="ProcessChunks", retries=3, retry_delay=10.0)

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        meta = context["FetchMetadata"]
        copc_url = meta["copc_url"]
        chunks = meta["chunks"]
        voxel_size = meta["voxel_size"]
        temp_dir = meta["temp_dir"]
        max_workers = meta["max_workers"]
        filter_outliers = meta["filter_outliers"]
        max_chunks = meta.get("max_chunks")
        
        if max_chunks is not None:
            logger.info(f"Limiting chunk processing to the first {max_chunks} chunks.")
            chunks = chunks[:max_chunks]
        
        # Clear temp_dir at the START of every attempt, not just at pipeline boot.
        # This prevents stale Parquet files written during a previous failed attempt
        # from mixing with files from this attempt and producing duplicate rows in DuckDB.
        if os.path.exists(temp_dir):
            logger.info(f"Clearing stale temp directory before processing: {temp_dir}")
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)
            
        logger.info(f"Starting parallel chunk processing. Total chunks: {len(chunks)}, Workers: {max_workers}")
        
        processed_files = []
        total_points_read = 0
        total_voxels_generated = 0
        failed_chunks = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = {
                executor.submit(
                    _process_single_chunk, 
                    chunk, copc_url, voxel_size, temp_dir, filter_outliers
                ): chunk
                for chunk in chunks
            }
            
            for idx, future in enumerate(as_completed(futures)):
                chunk = futures[future]
                chunk_id = chunk["id"]
                try:
                    out_path, pts, voxs = future.result()
                    if out_path:
                        processed_files.append(out_path)
                        total_points_read += pts
                        total_voxels_generated += voxs
                        logger.info(
                            f"[{idx+1}/{len(chunks)}] Chunk {chunk_id} processed: "
                            f"{pts:,} points -> {voxs:,} voxels."
                        )
                    else:
                        logger.info(f"[{idx+1}/{len(chunks)}] Chunk {chunk_id} processed (Empty).")
                except Exception as e:
                    # Log and skip the failed chunk rather than aborting the entire run.
                    # A transient S3 error on one chunk should not discard the work
                    # already completed by all other threads.
                    failed_chunks.append(chunk_id)
                    logger.warning(
                        f"[{idx+1}/{len(chunks)}] Chunk {chunk_id} SKIPPED after error: {str(e)}",
                        exc_info=True
                    )
        
        if failed_chunks:
            logger.warning(f"{len(failed_chunks)} chunk(s) were skipped due to errors: {failed_chunks}")
                    
        logger.info(
            f"Chunk processing complete. "
            f"Total points read: {total_points_read:,}. "
            f"Total voxels generated: {total_voxels_generated:,}."
        )
        
        if not processed_files and chunks:
            raise RuntimeError(f"All {len(chunks)} chunks failed to process during ingestion. Aborting pipeline.")
            
        return {
            "processed_files": processed_files,
            "total_points_read": total_points_read,
            "total_voxels_generated": total_voxels_generated,
            "failed_chunks": failed_chunks
        }


class SaveToStorageTask(Task):
    """
    Task 3: Bulk-loads all processed voxel Parquet files into a local DuckDB 
    database, creates spatial coordinate indexes, and cleans up temp files.
    """
    def __init__(self):
        super().__init__(name="SaveToStorage", retries=2, retry_delay=5.0)

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        meta = context["FetchMetadata"]
        process_result = context["ProcessChunks"]
        
        db_path = meta["db_path"]
        temp_dir = meta["temp_dir"]
        processed_files = process_result["processed_files"]
        
        if not processed_files:
            logger.warning("No processed files found to load into database.")
            return {"status": "skipped", "reason": "no data"}
            
        logger.info(f"Connecting to DuckDB storage at {db_path}...")
        
        with DatabaseManager(db_path) as db:
            import json
            
            resume_failed = meta.get("resume_failed", False)
            if resume_failed:
                logger.info("Resume Mode active. Appending data without resetting database table...")
                
                # Fetch previously failed list
                prev_failed_str = db.get_metadata("failed_chunks", "[]")
                try:
                    prev_failed_list = json.loads(prev_failed_str)
                except Exception:
                    prev_failed_list = []
                
                processed_chunk_ids = {c["id"] for c in meta["chunks"]}
                # Remove successfully recovered chunks from failed list
                updated_failed_list = [cid for cid in prev_failed_list if cid not in processed_chunk_ids]
                # Re-add any that failed again in this attempt
                for cid in process_result.get("failed_chunks", []):
                    if cid not in updated_failed_list:
                        updated_failed_list.append(cid)
            else:
                logger.info("Standard Mode active. Resetting database storage...")
                db.reset_storage()
                updated_failed_list = process_result.get("failed_chunks", [])
            
            # Use DuckDB native parquet reader to bulk load files
            # Note: We escape backslashes in path for Windows compatibility
            escaped_temp_path = os.path.join(temp_dir, "*.parquet").replace("\\", "/")
            logger.info(f"Bulk-loading data from Parquet files: {escaped_temp_path}")
            
            # Execute native bulk insert query
            db.conn.execute(f"INSERT INTO processed_voxels SELECT * FROM read_parquet('{escaped_temp_path}')")
            
            # Save metadata configuration
            db.set_metadata("voxel_size", meta["voxel_size"])
            db.set_metadata("grid_size", meta["grid_size"])
            db.set_metadata("copc_url", meta["copc_url"])
            db.set_metadata("point_count", meta["point_count"])
            db.set_metadata("filter_outliers", meta["filter_outliers"])
            db.set_metadata("failed_chunks", json.dumps(updated_failed_list))
            
            # Create indexing for fast query performance
            db.create_indexes()
            
            # Fetch stats
            stats = db.get_stats()
            logger.info("Verification stats retrieved from DuckDB:")
            logger.info(f"  Total stored voxels: {stats['total_voxels']:,}")
            logger.info(f"  Total stored points: {stats['total_points']:,}")
            logger.info(f"  Voxel X range: {stats['x_range']}")
            logger.info(f"  Voxel Y range: {stats['y_range']}")
            logger.info(f"  Voxel Z range: {stats['z_range']}")

            
        # Clean up temporary Parquet files
        try:
            logger.info(f"Cleaning up temporary directory: {temp_dir}")
            shutil.rmtree(temp_dir)
        except Exception as e:
            logger.warning(f"Could not delete temp directory {temp_dir}: {str(e)}")
            
        return {
            "status": "success",
            "db_stats": stats
        }
