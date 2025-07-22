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
        # Mock realistic coordinate values
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
