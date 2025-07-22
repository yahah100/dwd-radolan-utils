"""
Integration tests for dwd-radolan-utils.

These tests verify that different modules work together correctly.
"""
import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, Mock

from dwd_radolan_utils.extraction import extract_time_series_from_radar
from dwd_radolan_utils.geo_utils import cut_out_shapes
from dwd_radolan_utils.download import save_to_npz_files


class TestIntegrationExtractionAndGeoUtils:
    """Integration tests between extraction and geo_utils modules."""
    
    def test_extraction_with_geo_utils_integration(self, temp_directory):
        """Test that extraction properly uses geo_utils functions."""
        # Create mock radar data files
        base_date = datetime(2024, 1, 1)
        for i in range(2):
            date_str = base_date.replace(day=i+1).strftime("%Y%m%d")
            radar_file = temp_directory / f"{date_str}-{date_str}.npz"
            time_file = temp_directory / f"{date_str}-{date_str}_time.npz"
            
            # Create sample data that can be cut
            sample_data = np.random.rand(12, 100, 100) * 30
            sample_times = np.array([
                base_date.replace(day=i+1, hour=h) for h in range(12)
            ], dtype='datetime64')
            
            np.savez_compressed(radar_file, data=sample_data)
            np.savez_compressed(time_file, sample_times)
        
        # Create a sample grid with clear boundaries
        grid = np.zeros((50, 50), dtype=bool)
        grid[10:40, 10:40] = True  # Create a clear rectangular region
        
        # Test the integration
        with patch('dwd_radolan_utils.extraction.compute_arg_min_max_dict') as mock_compute_bounds:
            # Mock bounds that will work with our test data
            mock_compute_bounds.return_value = {
                'min_x': 10, 'max_x': 90, 'min_y': 10, 'max_y': 90
            }
            
            ts_array, timestamps = extract_time_series_from_radar(
                grid=grid,
                path=temp_directory,
                save=False
            )
            
            # Verify the integration worked
            assert isinstance(ts_array, np.ndarray)
            assert isinstance(timestamps, np.ndarray)
            assert ts_array.shape[0] == len(timestamps)
            assert ts_array.shape[1] == grid.shape[0]
            
            # Verify that cut_out_shapes was effectively used
            # (indirectly through the successful processing)
            mock_compute_bounds.assert_called_once()


class TestIntegrationDownloadAndExtraction:
    """Integration tests between download and extraction modules."""
    
    def test_download_save_and_extraction_integration(self, temp_directory):
        """Test that downloaded data can be properly extracted."""
        # Create sample data similar to what download would produce
        radar_data = np.random.rand(48, 100, 100) * 25  # 48 hours of data
        time_list = [datetime(2024, 1, 1, h) for h in range(24)] + \
                   [datetime(2024, 1, 2, h) for h in range(24)]
        
        # Use the save function from download module
        save_to_npz_files(radar_data, time_list, temp_directory)
        
        # Create a grid for extraction
        grid = np.ones((20, 20), dtype=bool)  # Simple boolean mask
        
        # Test that extraction can read the saved data
        with patch('dwd_radolan_utils.extraction.compute_arg_min_max_dict') as mock_compute_bounds:
            mock_compute_bounds.return_value = {
                'min_x': 0, 'max_x': 100, 'min_y': 0, 'max_y': 100
            }
            
            ts_array, timestamps = extract_time_series_from_radar(
                grid=grid,
                path=temp_directory,
                save=False
            )
            
            # Verify successful integration
            assert len(timestamps) == 48  # Should read all time steps
            assert ts_array.shape == (48, 20)  # Time steps x grid dimensions
            
            # Verify timestamps are correctly parsed
            assert timestamps[0] == np.datetime64('2024-01-01T00:00:00')
            assert timestamps[-1] == np.datetime64('2024-01-02T23:00:00')


class TestIntegrationFullWorkflow:
    """Integration test for the complete workflow."""
    
    @patch('dwd_radolan_utils.catchment_area.load_inflated_dem')
    @patch('dwd_radolan_utils.catchment_area.compute_accumulation')
    @patch('dwd_radolan_utils.catchment_area.compute_catchment_area')
    def test_full_catchment_to_extraction_workflow(self, mock_compute_catchment, 
                                                  mock_compute_acc, mock_load_dem, 
                                                  temp_directory):
        """Test the complete workflow from catchment computation to time series extraction."""
        # Mock catchment area computation
        mock_dem = Mock()
        mock_grid = Mock()
        mock_load_dem.return_value = (mock_dem, mock_grid)
        
        mock_acc = Mock()
        mock_fdir = Mock()
        mock_compute_acc.return_value = (mock_acc, mock_fdir)
        
        # Create a realistic distance grid
        dist_grid = np.random.rand(50, 50) * 1000  # Distance values in meters
        mock_clipped_catch = Mock()
        mock_snap_coords = (450000, 5650000)
        mock_compute_catchment.return_value = (mock_clipped_catch, dist_grid, mock_snap_coords)
        
        # Mock the grid conversion (this would normally be complex)
        with patch('dwd_radolan_utils.catchment_area.convert_grid_to_radolan_grid') as mock_convert:
            # Create a converted grid that mimics RADOLAN grid structure
            converted_grid = np.zeros((1, 900, 900))  # 1 catchment, RADOLAN dimensions
            # Add some realistic distance values in a small region
            converted_grid[0, 400:450, 400:450] = np.random.rand(50, 50) * 1000
            mock_convert.return_value = converted_grid
            
            # Import and test the catchment computation
            from dwd_radolan_utils.catchment_area import compute_catchement_for_location
            
            coordinates = (7.158556, 51.255604)  # Kluse station
            dist, grid = compute_catchement_for_location(coordinates, downsample_factor=50)
            
            # Verify catchment computation worked
            assert dist == dist_grid
            assert grid == mock_grid
            
            # Now test the extraction with the converted grid
            # Create mock radar data that matches the expected structure
            radar_data = np.random.rand(24, 80, 80) * 20  # 24 hours, smaller region
            time_list = [datetime(2024, 1, 1, h) for h in range(24)]
            
            # Save the radar data
            save_to_npz_files(radar_data, time_list, temp_directory)
            
            # Test extraction with the converted grid
            with patch('dwd_radolan_utils.extraction.compute_arg_min_max_dict') as mock_compute_bounds:
                # Use bounds that match our test data
                mock_compute_bounds.return_value = {
                    'min_x': 0, 'max_x': 80, 'min_y': 0, 'max_y': 80
                }
                
                # Extract the grid for extraction (first catchment)
                extraction_grid = converted_grid[0]
                
                ts_array, timestamps = extract_time_series_from_radar(
                    grid=extraction_grid,
                    path=temp_directory,
                    save=False
                )
                
                # Verify the complete workflow
                assert isinstance(ts_array, np.ndarray)
                assert len(timestamps) == 24
                assert ts_array.shape == (24, 900)  # Time x RADOLAN grid dimension
                
                # Verify all modules were called
                mock_load_dem.assert_called_once()
                mock_compute_acc.assert_called_once()
                mock_compute_catchment.assert_called_once()
                mock_convert.assert_called_once()


class TestIntegrationErrorHandling:
    """Integration tests for error handling across modules."""
    
    def test_extraction_handles_missing_data_gracefully(self, temp_directory):
        """Test that extraction handles missing or corrupted data files gracefully."""
        # Create a grid but no data files
        grid = np.ones((10, 10), dtype=bool)
        
        # Test extraction with no data files
        with pytest.raises(Exception, match="No files found"):
            extract_time_series_from_radar(
                grid=grid,
                path=temp_directory,
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 1, 2),
                save=False
            )
    
    def test_extraction_handles_invalid_grid_gracefully(self, temp_directory):
        """Test that extraction handles invalid grids gracefully."""
        # Create some data files
        radar_data = np.random.rand(12, 50, 50) * 20
        time_list = [datetime(2024, 1, 1, h) for h in range(12)]
        save_to_npz_files(radar_data, time_list, temp_directory)
        
        # Test with invalid grid (all NaN)
        invalid_grid = np.full((50, 50), np.nan)
        
        with pytest.raises(ValueError, match="All values in grid are NaN"):
            extract_time_series_from_radar(
                grid=invalid_grid,
                path=temp_directory,
                save=False
            )
    
    def test_file_format_consistency(self, temp_directory):
        """Test that saved files can be read consistently."""
        # Test CSV format
        ts_array = np.random.rand(24, 3)
        timestamps = np.array([datetime(2024, 1, 1, h) for h in range(24)], dtype='datetime64')
        
        from dwd_radolan_utils.extraction import save_ts_array
        
        # Save as CSV
        csv_path = temp_directory / "test.csv"
        save_ts_array(ts_array, timestamps, csv_path, file_format="csv")
        
        # Verify file exists and has correct structure
        assert csv_path.exists()
        
        import pandas as pd
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        assert len(df) == 24
        assert len(df.columns) == 3
        
        # Save as Parquet
        parquet_path = temp_directory / "test.parquet"
        save_ts_array(ts_array, timestamps, parquet_path, file_format="parquet")
        
        # Verify parquet file
        assert parquet_path.exists()
        df_parquet = pd.read_parquet(parquet_path)
        assert len(df_parquet) == 24
        assert len(df_parquet.columns) == 3 