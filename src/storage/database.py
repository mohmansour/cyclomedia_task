import os
import logging
import duckdb
import pandas as pd

logger = logging.getLogger("DatabaseManager")

class DatabaseManager:
    """
    Manages connections and transactions for the DuckDB storage layer.
    Supports bulk-loading data from pandas DataFrames.
    """
    def __init__(self, db_path: str, read_only: bool = False):
        self.db_path = db_path
        self.read_only = read_only
        self.conn = None

    def connect(self):
        if not self.conn:
            logger.info(f"Connecting to DuckDB database at '{self.db_path}' (read_only={self.read_only})...")
            self.conn = duckdb.connect(self.db_path, read_only=self.read_only)
            if not self.read_only:
                self._create_table()

    def disconnect(self):
        if self.conn:
            logger.info("Disconnecting from DuckDB...")
            self.conn.close()
            self.conn = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

    def _create_table(self):
        """
        Creates the processed_voxels and pipeline_metadata tables if they do not exist.
        """
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS processed_voxels (
                voxel_x BIGINT,
                voxel_y BIGINT,
                voxel_z BIGINT,
                point_count BIGINT,
                mean_z DOUBLE,
                std_z DOUBLE,
                min_z DOUBLE,
                max_z DOUBLE
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS pipeline_metadata (
                key VARCHAR UNIQUE,
                value VARCHAR
            )
        """)
        logger.debug("Tables 'processed_voxels' and 'pipeline_metadata' initialized.")

    def set_metadata(self, key: str, value: str):
        """
        Inserts or updates a metadata key-value pair.
        """
        self.conn.execute("""
            INSERT OR REPLACE INTO pipeline_metadata (key, value) 
            VALUES (?, ?)
        """, (key, str(value)))

    def get_metadata(self, key: str, default: str = None) -> str:
        """
        Retrieves a metadata value by key.
        """
        try:
            res = self.conn.execute("SELECT value FROM pipeline_metadata WHERE key = ?", (key,)).fetchone()
            return res[0] if res else default
        except Exception:
            return default

    def create_indexes(self):
        """
        Creates spatial/coordinate indexes on the table to speed up downstream queries.
        Drops any pre-existing index first to guarantee a clean rebuild on re-runs.
        """
        logger.info("Creating spatial indexes on 'processed_voxels'...")
        # Drop first: ensures the index is always rebuilt from the current data,
        # not inherited from a previous run where the table was repopulated.
        self.conn.execute("DROP INDEX IF EXISTS idx_voxel_coords")
        self.conn.execute("""
            CREATE INDEX idx_voxel_coords 
            ON processed_voxels (voxel_x, voxel_y, voxel_z)
        """)
        logger.info("Indexes created successfully.")

    def reset_storage(self):
        """
        Drops and fully recreates the processed_voxels table.
        
        Preferred over DELETE FROM because:
        - O(1) reset (no row-by-row scan).
        - Cascades the DROP to all attached indexes, so create_indexes()
          always runs against a clean, index-free table on re-runs.
        """
        logger.warning("Resetting storage: dropping and recreating 'processed_voxels'...")
        self.conn.execute("DROP TABLE IF EXISTS processed_voxels")
        self._create_table()

    def bulk_insert_voxels(self, df: pd.DataFrame):
        """
        Bulk inserts a pandas DataFrame of voxels into DuckDB.
        """
        if df.empty:
            return
        
        logger.debug(f"Bulk-inserting {len(df)} voxel records into DuckDB...")
        # DuckDB can register and query pandas dataframes in the local scope directly
        self.conn.execute("INSERT INTO processed_voxels SELECT * FROM df")

    def get_stats(self) -> dict:
        """
        Retrieves basic stats from the database for verification.
        """
        res = self.conn.execute("""
            SELECT 
                COUNT(*), 
                SUM(point_count),
                MIN(voxel_x), MAX(voxel_x),
                MIN(voxel_y), MAX(voxel_y),
                MIN(voxel_z), MAX(voxel_z)
            FROM processed_voxels
        """).fetchone()
        
        return {
            "total_voxels": res[0],
            "total_points": res[1],
            "x_range": (res[2], res[3]),
            "y_range": (res[4], res[5]),
            "z_range": (res[6], res[7])
        }
