"""
Tests for the geo_utils module.
"""

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from dwd_radolan_utils.geo_utils import (
    convert_radolan_to_wgs84,
    cut_out_shapes,
    get_wgs84_grid,
    turn_df_to_xarray,
)


class TestTurnDfToXarray:
    """Test cases for turn_df_to_xarray function."""

    def test_turn_df_to_xarray_basic(self):
        """Test basic DataFrame to xarray conversion."""
        # Create sample DataFrame
        data = {
            "x": [1, 2, 3, 1, 2, 3],
            "y": [1, 1, 1, 2, 2, 2],
            "time": pd.date_range("2024-01-01", periods=6),
            "RW": [10.5, 15.2, 8.7, 12.1, 9.8, 14.3],
        }
        df = pd.DataFrame(data)

        # Convert to xarray
        result = turn_df_to_xarray(df)

        # Check type and structure
        assert isinstance(result, xr.Dataset)
        assert "RW" in result.data_vars
        assert "x" in result.coords
        assert "y" in result.coords
        assert "time" in result.coords

        # Check attributes
        assert "standard_name" in result["x"].attrs
        assert "standard_name" in result["y"].attrs
        assert "standard_name" in result["time"].attrs
        assert "standard_name" in result["RW"].attrs

        assert result["x"].attrs["units"] == "km"
        assert result["y"].attrs["units"] == "km"
        assert result["RW"].attrs["unit"] == "mm h-1"

    def test_turn_df_to_xarray_attributes(self):
        """Test that correct attributes are added to the xarray."""
        data = {
            "x": [1, 2],
            "y": [1, 2],
            "time": pd.date_range("2024-01-01", periods=2),
            "RW": [10.0, 20.0],
        }
        df = pd.DataFrame(data)

        result = turn_df_to_xarray(df)

        # Check coordinate attributes
        assert result["x"].attrs["standard_name"] == "projection_x_coordinate"
        assert result["x"].attrs["long_name"] == "x coordinate of projection"
        assert result["y"].attrs["standard_name"] == "projection_y_coordinate"
        assert result["y"].attrs["long_name"] == "y coordinate of projection"

        # Check data variable attributes
        assert result["RW"].attrs["valid_min"] == 0
        assert result["RW"].attrs["valid_max"] == 100
        assert result["RW"].attrs["standard_name"] == "rainfall_rate"


class TestConvertRadolanToWgs84:
    """Test cases for convert_radolan_to_wgs84 function."""

    @patch("dwd_radolan_utils.geo_utils.pyproj.Transformer")
    def test_convert_radolan_to_wgs84_basic(self, mock_transformer_class):
        """Test basic coordinate conversion."""
        # Setup mock transformer
        mock_transformer = Mock()
        mock_transformer.transform.return_value = (7.0, 51.0)  # lon, lat
        mock_transformer_class.from_crs.return_value = mock_transformer

        # Test data
        x = np.array([100, 200])
        y = np.array([300, 400])

        # Call function
        result_x, result_y = convert_radolan_to_wgs84(x, y)

        # Verify
        assert result_x == 7.0
        assert result_y == 51.0
        mock_transformer.transform.assert_called_once_with(x, y)

    @patch("dwd_radolan_utils.geo_utils.pyproj.Transformer")
    def test_convert_radolan_to_wgs84_arrays(self, mock_transformer_class):
        """Test coordinate conversion with arrays."""
        # Setup mock transformer
        mock_transformer = Mock()
        mock_transformer.transform.return_value = (
            np.array([7.0, 7.1]),
            np.array([51.0, 51.1]),
        )
        mock_transformer_class.from_crs.return_value = mock_transformer

        # Test data
        x = np.array([100, 200])
        y = np.array([300, 400])

        # Call function
        result_x, result_y = convert_radolan_to_wgs84(x, y)

        # Verify
        assert isinstance(result_x, np.ndarray)
        assert isinstance(result_y, np.ndarray)
        assert len(result_x) == 2
        assert len(result_y) == 2


class TestGetWgs84Grid:
    """Test cases for get_wgs84_grid function."""

    @patch("dwd_radolan_utils.geo_utils.convert_radolan_to_wgs84")
    def test_get_wgs84_grid_shape(self, mock_convert):
        """Test that the WGS84 grid has correct shape."""
        # Mock the conversion function
        mock_convert.return_value = (
            np.arange(900),  # Mock longitude values
            np.arange(900),  # Mock latitude values
        )

        result = get_wgs84_grid()

        # Check shape
        assert result.shape == (900, 900, 2)

        # Check that it's a numpy array
        assert isinstance(result, np.ndarray)

    @patch("dwd_radolan_utils.geo_utils.convert_radolan_to_wgs84")
    def test_get_wgs84_grid_content(self, mock_convert):
        """Test the structure of the WGS84 grid content."""
        # Mock realistic coordinate values (should return 1D arrays like the real function)
        mock_convert.return_value = (
            np.linspace(5.0, 15.0, 900),  # Longitude range for Germany
            np.linspace(47.0, 55.0, 900),  # Latitude range for Germany
        )

        result = get_wgs84_grid()

        # Check that the third dimension contains lat, lon pairs
        assert result.shape[2] == 2

        # Check coordinate ranges are reasonable for Germany
        lats = result[:, :, 0]
        lons = result[:, :, 1]

        # Should have varying coordinates
        assert np.var(lats) > 0
        assert np.var(lons) > 0

    def test_get_wgs84_grid_coordinate_ranges(self):
        """Test that coordinate conversion is called with correct ranges."""
        with patch("dwd_radolan_utils.geo_utils.convert_radolan_to_wgs84") as mock_convert:
            # Mock return values
            mock_convert.return_value = (np.arange(900), np.arange(900))

            get_wgs84_grid()

            # Verify the function was called
            mock_convert.assert_called_once()

            # Check the input ranges to the conversion function
            args = mock_convert.call_args[0]
            x_coords, y_coords = args

            # Verify reasonable coordinate ranges for RADOLAN
            assert len(x_coords) == 900
            assert len(y_coords) == 900

    def test_get_wgs84_grid_orientation_latitude(self):
        """Test that latitude increases from south to north (bottom to top of array)."""
        grid = get_wgs84_grid()
        
        # Extract latitude values (first element of last dimension)
        lats = grid[:, :, 0]
        
        # Check that latitude increases from bottom to top (south to north)
        # Bottom row should have lower latitude than top row
        bottom_lat = lats[-1, 450]  # Middle of bottom row
        top_lat = lats[0, 450]      # Middle of top row
        
        assert bottom_lat < top_lat, f"Latitude should increase from south to north. Bottom: {bottom_lat}, Top: {top_lat}"
        
        # Check that latitude generally increases going up
        # Sample a few points along a column
        sample_indices = [800, 600, 400, 200, 50]  # From bottom to top
        sample_lats = [lats[i, 450] for i in sample_indices]
        
        # Each latitude should be higher than the previous (going north)
        for i in range(1, len(sample_lats)):
            assert sample_lats[i] > sample_lats[i-1], f"Latitude should increase going north. Index {sample_indices[i-1]}: {sample_lats[i-1]}, Index {sample_indices[i]}: {sample_lats[i]}"

    def test_get_wgs84_grid_orientation_longitude(self):
        """Test that longitude increases from west to east (left to right of array)."""
        grid = get_wgs84_grid()
        
        # Extract longitude values (second element of last dimension)
        lons = grid[:, :, 1]
        
        # Check that longitude increases from left to right (west to east)
        # Left column should have lower longitude than right column
        left_lon = lons[450, 0]    # Middle of left column
        right_lon = lons[450, -1]  # Middle of right column
        
        assert left_lon < right_lon, f"Longitude should increase from west to east. Left: {left_lon}, Right: {right_lon}"
        
        # Check that longitude generally increases going right
        # Sample a few points along a row
        sample_indices = [50, 200, 400, 600, 850]  # From left to right
        sample_lons = [lons[450, i] for i in sample_indices]
        
        # Each longitude should be higher than the previous (going east)
        for i in range(1, len(sample_lons)):
            assert sample_lons[i] > sample_lons[i-1], f"Longitude should increase going east. Index {sample_indices[i-1]}: {sample_lons[i-1]}, Index {sample_indices[i]}: {sample_lons[i]}"

    def test_get_wgs84_grid_corner_coordinates(self):
        """Test that corner coordinates are reasonable for Germany's coverage."""
        grid = get_wgs84_grid()
        
        # Extract corner coordinates
        # Note: Grid orientation should be [latitude, longitude]
        southwest_corner = grid[-1, 0, :]    # Bottom-left: [lat, lon]
        southeast_corner = grid[-1, -1, :]   # Bottom-right: [lat, lon]
        northwest_corner = grid[0, 0, :]     # Top-left: [lat, lon]
        northeast_corner = grid[0, -1, :]    # Top-right: [lat, lon]
        
        # Germany approximately covers:
        # Latitude: 47°N to 55°N
        # Longitude: 5°E to 15°E
        # But Radolan grid extends beyond Germany borders
        
        # Test coordinate ranges (allowing some buffer beyond Germany)
        assert 45.0 <= southwest_corner[0] <= 50.0, f"SW latitude should be reasonable: {southwest_corner[0]}"
        assert 52.0 <= northwest_corner[0] <= 57.0, f"NW latitude should be reasonable: {northwest_corner[0]}"
        assert 45.0 <= southeast_corner[0] <= 50.0, f"SE latitude should be reasonable: {southeast_corner[0]}"
        assert 52.0 <= northeast_corner[0] <= 57.0, f"NE latitude should be reasonable: {northeast_corner[0]}"
        
        assert 2.0 <= southwest_corner[1] <= 8.0, f"SW longitude should be reasonable: {southwest_corner[1]}"
        assert 2.0 <= northwest_corner[1] <= 8.0, f"NW longitude should be reasonable: {northwest_corner[1]}"
        assert 13.0 <= southeast_corner[1] <= 18.0, f"SE longitude should be reasonable: {southeast_corner[1]}"
        assert 13.0 <= northeast_corner[1] <= 18.0, f"NE longitude should be reasonable: {northeast_corner[1]}"
        
        # Test corner relationships
        assert northwest_corner[0] > southwest_corner[0], "North should be higher latitude than south"
        assert northeast_corner[0] > southeast_corner[0], "North should be higher latitude than south"
        assert southeast_corner[1] > southwest_corner[1], "East should be higher longitude than west"
        assert northeast_corner[1] > northwest_corner[1], "East should be higher longitude than west"

    def test_get_wgs84_grid_monotonic_increase(self):
        """Test that coordinates increase monotonically in both directions."""
        grid = get_wgs84_grid()
        
        # Test monotonic increase in latitude (north direction)
        # Pick a column and check that latitude increases going up (decreasing row index)
        middle_col = 450
        lat_column = grid[:, middle_col, 0]
        
        # Since array index 0 is north and 899 is south, latitude should decrease with increasing index
        for i in range(len(lat_column) - 1):
            assert lat_column[i] >= lat_column[i + 1], f"Latitude should decrease with increasing row index. Row {i}: {lat_column[i]}, Row {i+1}: {lat_column[i+1]}"
        
        # Test monotonic increase in longitude (east direction)
        # Pick a row and check that longitude increases going right
        middle_row = 450
        lon_row = grid[middle_row, :, 1]
        
        for i in range(len(lon_row) - 1):
            assert lon_row[i] <= lon_row[i + 1], f"Longitude should increase with increasing column index. Col {i}: {lon_row[i]}, Col {i+1}: {lon_row[i+1]}"

    def test_get_wgs84_grid_center_coordinate(self):
        """Test that the center coordinate is reasonable for Germany."""
        grid = get_wgs84_grid()
        
        # Get center coordinate
        center_lat = grid[450, 450, 0]
        center_lon = grid[450, 450, 1]
        
        # Center of Germany is approximately 51°N, 10°E
        # Allow some tolerance since Radolan grid may not be perfectly centered on Germany
        assert 49.0 <= center_lat <= 53.0, f"Center latitude should be reasonable for Germany: {center_lat}"
        assert 8.0 <= center_lon <= 12.0, f"Center longitude should be reasonable for Germany: {center_lon}"


class TestCutOutShapes:
    """Test cases for cut_out_shapes function."""

    def test_cut_out_shapes_2d(self):
        """Test cutting out shapes from 2D array."""
        # Create test array
        array = np.arange(100).reshape(10, 10)

        # Cut out a sub-region
        result = cut_out_shapes(x=array, min_dim_1=2, max_dim_1=8, min_dim_2=3, max_dim_2=7)

        # Check shape
        assert result.shape == (6, 4)  # (8-2, 7-3)

        # Check content
        expected = array[2:8, 3:7]
        np.testing.assert_array_equal(result, expected)

    def test_cut_out_shapes_3d(self):
        """Test cutting out shapes from 3D array."""
        # Create test array
        array = np.arange(1000).reshape(10, 10, 10)

        # Cut out a sub-region
        result = cut_out_shapes(x=array, min_dim_1=1, max_dim_1=9, min_dim_2=2, max_dim_2=8)

        # Check shape
        assert result.shape == (10, 8, 6)  # (10, 9-1, 8-2)

        # Check content
        expected = array[:, 1:9, 2:8]
        np.testing.assert_array_equal(result, expected)

    def test_cut_out_shapes_full_range(self):
        """Test cutting with full range (no actual cutting)."""
        array = np.arange(100).reshape(10, 10)

        result = cut_out_shapes(x=array, min_dim_1=0, max_dim_1=10, min_dim_2=0, max_dim_2=10)

        # Should be identical to original
        np.testing.assert_array_equal(result, array)

    def test_cut_out_shapes_edge_cases(self):
        """Test edge cases for cutting."""
        array = np.arange(100).reshape(10, 10)

        # Single row/column
        result = cut_out_shapes(x=array, min_dim_1=5, max_dim_1=6, min_dim_2=3, max_dim_2=7)

        assert result.shape == (1, 4)

        # Single element
        result = cut_out_shapes(x=array, min_dim_1=5, max_dim_1=6, min_dim_2=3, max_dim_2=4)

        assert result.shape == (1, 1)

    def test_cut_out_shapes_invalid_dimensions(self):
        """Test error handling for invalid dimensions."""
        array = np.arange(100)  # 1D array

        with pytest.raises(ValueError, match="Input array must have either 2 or 3 dimensions"):
            cut_out_shapes(x=array, min_dim_1=0, max_dim_1=10, min_dim_2=0, max_dim_2=10)

    def test_cut_out_shapes_boundary_validation(self):
        """Test that boundaries work correctly."""
        array = np.random.rand(20, 30, 40)

        # Test various boundary combinations
        result = cut_out_shapes(x=array, min_dim_1=5, max_dim_1=15, min_dim_2=10, max_dim_2=25)

        assert result.shape == (20, 10, 15)  # (original_dim_0, 15-5, 25-10)

        # Verify content
        expected = array[:, 5:15, 10:25]
        np.testing.assert_array_equal(result, expected)
