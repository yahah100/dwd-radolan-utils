"""
Tests for the extraction module.
"""

from datetime import datetime
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from dwd_radolan_utils.extraction import (
    aggregate_ts,
    compute_arg_min_max_dict,
    compute_arg_min_max_dict_bool,
    compute_arg_min_max_dict_nan,
    extract_time_series_from_radar,
    read_radar_data,
    save_ts_array,
)


class TestReadRadarData:
    """Test cases for read_radar_data function."""

    def test_read_radar_data_basic(self, mock_radar_files, sample_min_max_dict):
        """Test basic radar data reading functionality."""
        path = mock_radar_files[0][0].parent

        with patch("dwd_radolan_utils.extraction.cut_out_shapes") as mock_cut:
            mock_cut.side_effect = lambda x, **kwargs: x  # Return input unchanged

            radar_data, time_data = read_radar_data(path=path, min_max_dict=sample_min_max_dict)

            assert isinstance(radar_data, np.ndarray)
            assert isinstance(time_data, np.ndarray)
            assert radar_data.shape[0] == time_data.shape[0]

    def test_read_radar_data_with_date_filter(self, mock_radar_files, sample_min_max_dict):
        """Test radar data reading with date filtering."""
        path = mock_radar_files[0][0].parent
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 2)

        with patch("dwd_radolan_utils.extraction.cut_out_shapes") as mock_cut:
            mock_cut.side_effect = lambda x, **kwargs: x

            radar_data, time_data = read_radar_data(
                path=path,
                min_max_dict=sample_min_max_dict,
                start_date=start_date,
                end_date=end_date,
            )

            assert len(radar_data) > 0
            assert len(time_data) > 0

    def test_read_radar_data_no_files_found(self, temp_directory, sample_min_max_dict):
        """Test error handling when no files are found."""
        with pytest.raises(Exception, match="No files found"):
            read_radar_data(
                path=temp_directory,
                min_max_dict=sample_min_max_dict,
                start_date=datetime(2023, 1, 1),
                end_date=datetime(2023, 1, 2),
            )


class TestComputeArgMinMaxDict:
    """Test cases for compute_arg_min_max_dict functions."""

    def test_compute_arg_min_max_dict_nan_2d(self):
        """Test NaN boundary computation for 2D grid."""
        grid = np.full((10, 10), np.nan)
        grid[3:7, 2:8] = 1.0  # Valid data region

        result = compute_arg_min_max_dict_nan(grid)

        assert result["min_x"] == 3
        assert result["max_x"] == 7
        assert result["min_y"] == 2
        assert result["max_y"] == 8

    def test_compute_arg_min_max_dict_nan_3d(self):
        """Test NaN boundary computation for 3D grid."""
        grid = np.full((5, 10, 10), np.nan)
        grid[:, 3:7, 2:8] = 1.0  # Valid data region

        result = compute_arg_min_max_dict_nan(grid)

        assert result["min_x"] == 3
        assert result["max_x"] == 7
        assert result["min_y"] == 2
        assert result["max_y"] == 8

    def test_compute_arg_min_max_dict_nan_all_nan(self):
        """Test error handling for all-NaN grid."""
        grid = np.full((10, 10), np.nan)

        with pytest.raises(ValueError, match="All values in grid are NaN"):
            compute_arg_min_max_dict_nan(grid)

    def test_compute_arg_min_max_dict_bool_2d(self, sample_boolean_grid):
        """Test boolean boundary computation for 2D grid."""
        result = compute_arg_min_max_dict_bool(sample_boolean_grid)

        assert result["min_x"] == 25
        assert result["max_x"] == 75
        assert result["min_y"] == 25
        assert result["max_y"] == 75

    def test_compute_arg_min_max_dict_bool_all_false(self):
        """Test error handling for all-False boolean grid."""
        grid = np.zeros((10, 10), dtype=bool)

        with pytest.raises(ValueError, match="All values in boolean grid are False"):
            compute_arg_min_max_dict_bool(grid)

    def test_compute_arg_min_max_dict_dispatcher(self):
        """Test the main dispatcher function."""
        # Test NaN grid
        nan_grid = np.full((10, 10), np.nan)
        nan_grid[3:7, 2:8] = 1.0
        result = compute_arg_min_max_dict(nan_grid)
        assert result["min_x"] == 3

        # Test boolean grid
        bool_grid = np.zeros((10, 10), dtype=bool)
        bool_grid[3:7, 2:8] = True
        result = compute_arg_min_max_dict(bool_grid)
        assert result["min_x"] == 3

        # Test invalid grid
        invalid_grid = np.random.rand(10, 10) * 100  # No NaN, not boolean
        with pytest.raises(ValueError, match="Grid has unsupported data type"):
            compute_arg_min_max_dict(invalid_grid)


class TestAggregateTs:
    """Test cases for aggregate_ts function."""

    def test_aggregate_ts_mean(self, sample_grid, sample_radar_data):
        """Test time series aggregation with mean method."""
        result = aggregate_ts(sample_grid, sample_radar_data, method="mean")

        assert isinstance(result, np.ndarray)
        assert result.shape == (sample_radar_data.shape[0],)
        assert not np.any(np.isnan(result))  # Should not have NaN after processing

    def test_aggregate_ts_sum(self, sample_grid, sample_radar_data):
        """Test time series aggregation with sum method."""
        result = aggregate_ts(sample_grid, sample_radar_data, method="sum")

        assert isinstance(result, np.ndarray)
        assert result.shape == (sample_radar_data.shape[0],)

    def test_aggregate_ts_max(self, sample_grid, sample_radar_data):
        """Test time series aggregation with max method."""
        result = aggregate_ts(sample_grid, sample_radar_data, method="max")

        assert isinstance(result, np.ndarray)
        assert result.shape == (sample_radar_data.shape[0],)

    def test_aggregate_ts_min(self, sample_grid, sample_radar_data):
        """Test time series aggregation with min method."""
        result = aggregate_ts(sample_grid, sample_radar_data, method="min")

        assert isinstance(result, np.ndarray)
        assert result.shape == (sample_radar_data.shape[0],)

    def test_aggregate_ts_invalid_method(self, sample_grid, sample_radar_data):
        """Test error handling for invalid aggregation method."""
        with pytest.raises(ValueError, match="Method invalid not supported"):
            aggregate_ts(sample_grid, sample_radar_data, method="invalid")  # type: ignore

    def test_aggregate_ts_boolean_grid(self, sample_boolean_grid, sample_radar_data):
        """Test aggregation with boolean grid input."""
        result = aggregate_ts(sample_boolean_grid, sample_radar_data, method="mean")

        assert isinstance(result, np.ndarray)
        assert result.shape == (sample_radar_data.shape[0],)


class TestExtractTimeSeriesFromRadar:
    """Test cases for extract_time_series_from_radar function."""

    @patch("dwd_radolan_utils.extraction.read_radar_data")
    @patch("dwd_radolan_utils.extraction.cut_out_shapes")
    def test_extract_time_series_basic(
        self,
        mock_cut,
        mock_read,
        sample_boolean_grid,
        sample_radar_data,
        sample_time_data,
        temp_directory,
    ):
        """Test basic time series extraction."""
        # Setup mocks
        mock_read.return_value = (sample_radar_data, sample_time_data)
        mock_cut.side_effect = lambda x, **kwargs: x

        # Test the function
        ts_array, timestamps = extract_time_series_from_radar(grid=sample_boolean_grid, path=temp_directory, save=False)

        assert isinstance(ts_array, np.ndarray)
        assert isinstance(timestamps, np.ndarray)
        assert ts_array.shape[0] == len(timestamps)
        assert ts_array.shape[1] == sample_boolean_grid.shape[0]

    @patch("dwd_radolan_utils.extraction.read_radar_data")
    @patch("dwd_radolan_utils.extraction.cut_out_shapes")
    @patch("dwd_radolan_utils.extraction.save_ts_array")
    def test_extract_time_series_with_save(
        self,
        mock_save,
        mock_cut,
        mock_read,
        sample_boolean_grid,
        sample_radar_data,
        sample_time_data,
        temp_directory,
    ):
        """Test time series extraction with saving."""
        # Setup mocks
        mock_read.return_value = (sample_radar_data, sample_time_data)
        mock_cut.side_effect = lambda x, **kwargs: x

        # Test the function
        ts_array, timestamps = extract_time_series_from_radar(
            grid=sample_boolean_grid,
            path=temp_directory,
            save=True,
            save_path=temp_directory / "test_output.csv",
        )

        # Verify save was called
        mock_save.assert_called_once()


class TestSaveTsArray:
    """Test cases for save_ts_array function."""

    def test_save_ts_array_csv(self, sample_time_data, temp_directory):
        """Test saving time series array to CSV."""
        ts_array = np.random.rand(len(sample_time_data), 3)
        save_path = temp_directory / "test_output.csv"
        column_names = ["ts_0", "ts_1", "ts_2"]

        save_ts_array(
            ts_array=ts_array,
            timestamps=sample_time_data,
            path=save_path,
            column_names=column_names,
            file_format="csv",
        )

        assert save_path.exists()

        # Verify the saved data
        df = pd.read_csv(save_path, index_col=0, parse_dates=True)
        assert list(df.columns) == column_names
        assert len(df) == len(sample_time_data)

    def test_save_ts_array_parquet(self, sample_time_data, temp_directory):
        """Test saving time series array to Parquet."""
        ts_array = np.random.rand(len(sample_time_data), 3)
        save_path = temp_directory / "test_output.parquet"

        save_ts_array(
            ts_array=ts_array,
            timestamps=sample_time_data,
            path=save_path,
            file_format="parquet",
        )

        assert save_path.exists()

        # Verify the saved data
        df = pd.read_parquet(save_path)
        assert len(df) == len(sample_time_data)

    def test_save_ts_array_default_columns(self, sample_time_data, temp_directory):
        """Test saving with default column names."""
        ts_array = np.random.rand(len(sample_time_data), 3)
        save_path = temp_directory / "test_output.csv"

        save_ts_array(ts_array=ts_array, timestamps=sample_time_data, path=save_path)

        assert save_path.exists()

        # Check default column names
        df = pd.read_csv(save_path, index_col=0)
        expected_columns = ["ts_0", "ts_1", "ts_2"]
        assert list(df.columns) == expected_columns

    def test_save_ts_array_invalid_format(self, sample_time_data, temp_directory):
        """Test error handling for invalid file format."""
        ts_array = np.random.rand(len(sample_time_data), 3)
        save_path = temp_directory / "test_output.invalid"

        with pytest.raises(ValueError, match="File format invalid not supported"):
            save_ts_array(
                ts_array=ts_array,
                timestamps=sample_time_data,
                path=save_path,
                file_format="invalid",  # type: ignore
            )
