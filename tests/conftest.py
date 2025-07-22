"""
Pytest configuration and shared fixtures for dwd-radolan-utils tests.
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, MagicMock
import tempfile
import shutil


@pytest.fixture
def sample_radar_data():
    """Create sample radar data for testing."""
    # Create a 3D array (time, lat, lon) with some realistic patterns
    time_steps = 24
    lat_size = 100
    lon_size = 100
    
    data = np.random.rand(time_steps, lat_size, lon_size) * 50  # Rain values 0-50mm
    # Add some NaN values to simulate missing data
    data[data < 5] = np.nan
    return data


@pytest.fixture
def sample_time_data():
    """Create sample time data for testing."""
    time_data = []
    base_time = datetime(2024, 1, 1, 0, 0, 0)
    for i in range(24):
        time_data.append(base_time.replace(hour=i))
    return np.array(time_data, dtype='datetime64')


@pytest.fixture
def sample_grid():
    """Create a sample grid for testing."""
    grid_size = 100
    grid = np.random.rand(grid_size, grid_size)
    # Create some patterns - higher values in center
    center = grid_size // 2
    y, x = np.ogrid[:grid_size, :grid_size]
    mask = (x - center)**2 + (y - center)**2 <= (grid_size//4)**2
    grid[mask] *= 2
    return grid


@pytest.fixture
def sample_boolean_grid():
    """Create a sample boolean grid for testing."""
    grid_size = 100
    grid = np.zeros((grid_size, grid_size), dtype=bool)
    # Create a rectangular region of True values
    grid[25:75, 25:75] = True
    return grid


@pytest.fixture
def sample_min_max_dict():
    """Create a sample min/max dictionary for testing."""
    return {
        "min_x": 10,
        "max_x": 90,
        "min_y": 10, 
        "max_y": 90
    }


@pytest.fixture
def sample_wgs84_coordinates():
    """Sample WGS84 coordinates for testing."""
    return [(7.158556, 51.255604)]  # Kluse discharge station


@pytest.fixture
def mock_dem_data():
    """Create mock DEM data for testing."""
    size = 50
    dem = np.random.rand(size, size) * 1000  # Elevation 0-1000m
    # Add some structure - lower values near edges
    dem[0, :] = dem[0, :] * 0.1
    dem[-1, :] = dem[-1, :] * 0.1
    dem[:, 0] = dem[:, 0] * 0.1
    dem[:, -1] = dem[:, -1] * 0.1
    return dem


@pytest.fixture
def temp_directory():
    """Create a temporary directory for testing file operations."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_radar_files(temp_directory):
    """Create mock radar data files for testing."""
    files = []
    base_date = datetime(2024, 1, 1)
    
    for i in range(3):
        # Create sample radar data file
        date_str = base_date.replace(day=i+1).strftime("%Y%m%d")
        radar_file = temp_directory / f"{date_str}-{date_str}.npz"
        time_file = temp_directory / f"{date_str}-{date_str}_time.npz"
        
        # Sample data with consistent dimensions (80x80 to match min_max_dict)
        sample_data = np.random.rand(24, 80, 80) * 30
        sample_times = np.array([
            base_date.replace(day=i+1, hour=h) for h in range(24)
        ], dtype='datetime64')
        
        np.savez_compressed(radar_file, data=sample_data)
        np.savez_compressed(time_file, sample_times)
        
        files.append((radar_file, time_file))
    
    return files


@pytest.fixture
def mock_radolan_response():
    """Mock HTTP response for RADOLAN data downloads."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.content = b"mock radolan data content"
    return mock_response


@pytest.fixture
def sample_dataframe():
    """Create a sample pandas DataFrame for testing."""
    dates = pd.date_range('2024-01-01', periods=24, freq='H')
    data = {
        'timestamp': dates,
        'ts_0': np.random.rand(24) * 10,
        'ts_1': np.random.rand(24) * 15,
        'ts_2': np.random.rand(24) * 8
    }
    return pd.DataFrame(data)


@pytest.fixture
def mock_pysheds_grid():
    """Create a mock pysheds Grid object for testing."""
    mock_grid = Mock()
    mock_grid.crs = "EPSG:32633"  # UTM Zone 33N
    mock_grid.extent = (400000, 500000, 5600000, 5700000)  # Sample UTM bounds
    mock_grid.affine = None  # Simplified for testing
    mock_grid.shape = (100, 100)
    
    # Mock methods
    mock_grid.snap_to_mask.return_value = (450000, 5650000)  # Sample coordinates
    mock_grid.catchment.return_value = np.ones((100, 100), dtype=bool)
    mock_grid.distance_to_outlet.return_value = np.random.rand(100, 100) * 1000
    mock_grid.nearest_cell.return_value = (50, 50)
    
    return mock_grid


@pytest.fixture 
def mock_pysheds_raster():
    """Create a mock pysheds Raster object for testing."""
    mock_raster = Mock()
    mock_raster.shape = (100, 100)
    mock_raster.nodata = np.float32(-9999)
    
    # Create sample raster data
    data = np.random.rand(100, 100).astype(np.float32) * 500  # Sample elevation data
    data[0:10, 0:10] = mock_raster.nodata  # Some nodata values
    
    # Make the mock behave like a numpy array for indexing and comparisons
    mock_raster.__getitem__ = lambda self, key: data[key]
    mock_raster.__setitem__ = lambda self, key, value: data.__setitem__(key, value)
    mock_raster.__lt__ = lambda self, other: data < other
    mock_raster.__gt__ = lambda self, other: data > other
    mock_raster.__ge__ = lambda self, other: data >= other
    mock_raster.__le__ = lambda self, other: data <= other
    
    return mock_raster 