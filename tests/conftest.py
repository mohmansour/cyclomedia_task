import os
import shutil
import tempfile
import numpy as np
import pytest
from fastapi.testclient import TestClient

@pytest.fixture
def mock_db_path():
    """Provides a temporary file path for testing DuckDB without overwriting the production DB."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    try:
        os.remove(path)
    except OSError:
        pass
    yield path

    # Clean up after test
    if os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass

@pytest.fixture
def temp_parquet_dir():
    """Provides a temporary directory for Parquet chunk output buffers."""
    path = tempfile.mkdtemp()
    yield path
    # Clean up after test
    if os.path.exists(path):
        shutil.rmtree(path)

@pytest.fixture
def sample_point_data():
    """Generates a dictionary of X, Y, Z coordinate arrays with outliers and noise."""
    np.random.seed(42)
    # Generate 100 clean points around center
    x = np.random.uniform(188000, 188100, 100)
    y = np.random.uniform(1878900, 1879000, 100)
    # Most points between 10.0m and 12.0m elevation
    z = np.random.normal(11.0, 0.5, 100)
    
    # Add explicit range outliers (e.g. -100m, 200m)
    x = np.append(x, [188050.0, 188050.0])
    y = np.append(y, [1878950.0, 1878950.0])
    z = np.append(z, [-100.0, 200.0])
    
    # Add statistical outlier (e.g. 20.0m - more than 3-sigma from mean 11.0m)
    x = np.append(x, [188050.0])
    y = np.append(y, [1878950.0])
    z = np.append(z, [20.0])
    
    return {
        "x": x,
        "y": y,
        "z": z,
        "clean_count": 100 # Expected clean points after range and 3-sigma filtering
    }

@pytest.fixture
def test_client(mock_db_path, monkeypatch):
    """Provides a FastAPI TestClient configured to use the mock DB path."""
    import src.service.api as api
    monkeypatch.setattr(api, "DB_PATH", mock_db_path)
    from src.service.api import app
    with TestClient(app) as client:
        yield client

