import os
import argparse
import logging
import sys
import time
import duckdb

# Add the project root to sys.path to resolve src package imports correctly
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.pipeline.dag import Pipeline
from src.pipeline.tasks import FetchMetadataTask, ProcessChunksTask, SaveToStorageTask

def setup_logging():
    """
    Configures structured console logging with millisecond timestamps and clear formatting.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Create console handler with formatting
    ch = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)-8s] [%(name)-15s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)

def run_verification_queries(db_path: str):
    """
    Runs simple SQL verification queries against the generated DuckDB database.
    """
    print("\n" + "="*50)
    print("DATABASE VERIFICATION STATS (DUCKDB)")
    print("="*50)
    
    if not os.path.exists(db_path):
        print(f"Error: Database file '{db_path}' not found.")
        return
        
    conn = duckdb.connect(db_path)
    
    # 1. Total processed voxel count and point count sum
    stats = conn.execute("""
        SELECT 
            COUNT(*) as voxel_count,
            SUM(point_count) as total_points,
            AVG(point_count) as avg_points_per_voxel
        FROM processed_voxels
    """).fetchone()
    print(f"Total Processed Voxels:      {stats[0]:,}")
    print(f"Total Points Represented:    {stats[1]:,}")
    print(f"Average Points per Voxel:    {stats[2]:.2f}")
    
    # 2. Voxel coordinates extents
    extents = conn.execute("""
        SELECT 
            MIN(voxel_x), MAX(voxel_x),
            MIN(voxel_y), MAX(voxel_y),
            MIN(voxel_z), MAX(voxel_z)
        FROM processed_voxels
    """).fetchone()
    print(f"Voxel Index Extents:")
    print(f"  Voxel X range: [{extents[0]}, {extents[1]}]")
    print(f"  Voxel Y range: [{extents[2]}, {extents[3]}]")
    print(f"  Voxel Z range: [{extents[4]}, {extents[5]}]")
    
    # 3. Dense region profile (Top 5 voxels by density)
    print("\nTop 5 Densest Voxels (Highest point count):")
    densest = conn.execute("""
        SELECT voxel_x, voxel_y, voxel_z, point_count, round(mean_z, 2) as mean_z, round(std_z, 2) as std_z
        FROM processed_voxels
        ORDER BY point_count DESC
        LIMIT 5
    """).fetchall()
    print(f"  {'Voxel (X, Y, Z)':<25} | {'Point Count':<12} | {'Mean Elev.':<10} | {'Std Dev':<8}")
    print(f"  {'-'*25}-+-{'-'*12}-+-{'-'*10}-+-{'-'*8}")
    for row in densest:
        coord_str = f"({row[0]}, {row[1]}, {row[2]})"
        print(f"  {coord_str:<25} | {row[3]:<12,} | {row[4]:<10} | {row[5]:<8}")

    # 4. Elevation distribution statistics
    elev_stats = conn.execute("""
        SELECT 
            MIN(min_z) as min_elev,
            MAX(max_z) as max_elev,
            AVG(mean_z) as avg_elev
        FROM processed_voxels
    """).fetchone()
    print("\nElevation Profile (meters):")
    print(f"  Minimum Elevation: {elev_stats[0]:.2f}")
    print(f"  Maximum Elevation: {elev_stats[1]:.2f}")
    print(f"  Average Elevation: {elev_stats[2]:.2f}")
    print("="*50 + "\n")
    
    conn.close()

def main():
    parser = argparse.ArgumentParser(description="Data Engineer Point Cloud Pipeline")
    parser.add_argument(
        "--url", 
        type=str, 
        default="https://s3.amazonaws.com/hobu-lidar/sofi.copc.laz",
        help="S3 URL to the remote COPC LAZ file"
    )
    parser.add_argument(
        "--grid-size", 
        type=int, 
        default=8,
        help="Grid dimension for 2D spatial splitting (e.g. 8 yields an 8x8 grid of 64 chunks)"
    )
    parser.add_argument(
        "--voxel-size", 
        type=float, 
        default=2.0,
        help="Voxel cell dimension in coordinate units (meters) for downsampling"
    )
    parser.add_argument(
        "--db-path", 
        type=str, 
        default="point_cloud_pipeline.db",
        help="Path to save the output DuckDB file"
    )
    parser.add_argument(
        "--temp-dir", 
        type=str, 
        default="temp_parquet",
        help="Temporary directory for streaming chunk parquet outputs"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Skip pipeline execution and run verification on an existing DB"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel worker threads for streaming chunks"
    )
    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="Disable data quality outlier filtering"
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Limit the number of chunks processed for quick testing/reproduction"
    )
    
    args = parser.parse_args()
    
    if args.verify_only:
        run_verification_queries(args.db_path)
        return
        
    setup_logging()
    
    logger = logging.getLogger("PipelineRunner")
    logger.info("Initializing Data Engineer Point Cloud Pipeline...")
    
    # Instantiate the tasks
    fetch_metadata = FetchMetadataTask(
        copc_url=args.url,
        grid_size=args.grid_size,
        voxel_size=args.voxel_size,
        db_path=args.db_path,
        temp_dir=args.temp_dir,
        max_workers=args.workers,
        filter_outliers=not args.no_filter,
        max_chunks=args.max_chunks
    )
    process_chunks = ProcessChunksTask()
    save_to_storage = SaveToStorageTask()
    
    # Build the DAG Pipeline
    pipeline = Pipeline()
    pipeline.add_task(fetch_metadata)
    pipeline.add_task(process_chunks)
    pipeline.add_task(save_to_storage)
    
    # Establish dependencies
    # ProcessChunks depends on FetchMetadata
    process_chunks.dependencies = ["FetchMetadata"]
    # SaveToStorage depends on ProcessChunks
    save_to_storage.dependencies = ["ProcessChunks"]
    
    start_time = time.time()
    try:
        pipeline.run()
        duration = time.time() - start_time
        logger.info(f"Pipeline executed successfully in {duration:.2f} seconds.")
        
        # Run verification queries at the end
        run_verification_queries(args.db_path)
        
    except Exception as e:
        logger.critical(f"Pipeline execution failed: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
