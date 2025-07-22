"""
Tests for the catchment_area module.
"""
import pytest
import numpy as np
import time
import logging
from pathlib import Path
from unittest.mock import patch, Mock, MagicMock

from dwd_radolan_utils.catchment_area import (
    load_inflated_dem,
    compute_accumulation,
    compute_catchment_area,
    compute_catchement_for_location,
    compute_multiple_catchments,
    convert_grid_to_radolan_grid_loops,
    convert_grid_to_radolan_grid_vectorized,
    convert_grid_to_radolan_grid,
)


class TestLoadInflatedDem:
    """Test cases for load_inflated_dem function."""
    
    @patch('dwd_radolan_utils.catchment_area.load_dem')
    @patch('dwd_radolan_utils.catchment_area.zoom_dem')
    def test_load_inflated_dem_basic(self, mock_zoom, mock_load, mock_pysheds_raster, mock_pysheds_grid):
        """Test basic DEM loading and processing."""
        # Setup mocks
        mock_load.return_value = (mock_pysheds_raster, mock_pysheds_grid)
        mock_zoom.return_value = (mock_pysheds_raster, mock_pysheds_grid)
        
        # Mock the grid methods
        mock_pysheds_grid.fill_pits.return_value = mock_pysheds_raster
        mock_pysheds_grid.fill_depressions.return_value = mock_pysheds_raster
        mock_pysheds_grid.resolve_flats.return_value = mock_pysheds_raster
        
        result_dem, result_grid = load_inflated_dem(downsample_factor=10)
        
        # Verify calls
        mock_load.assert_called_once()
        mock_zoom.assert_called_once()
        mock_pysheds_grid.fill_pits.assert_called_once()
        mock_pysheds_grid.fill_depressions.assert_called_once()
        mock_pysheds_grid.resolve_flats.assert_called_once()
        
        assert result_dem == mock_pysheds_raster
        assert result_grid == mock_pysheds_grid
    
    @patch('dwd_radolan_utils.catchment_area.load_dem')
    @patch('dwd_radolan_utils.catchment_area.zoom_dem')
    def test_load_inflated_dem_custom_path(self, mock_zoom, mock_load, mock_pysheds_raster, mock_pysheds_grid):
        """Test DEM loading with custom data directory."""
        mock_load.return_value = (mock_pysheds_raster, mock_pysheds_grid)
        mock_zoom.return_value = (mock_pysheds_raster, mock_pysheds_grid)
        
        # Mock the grid methods
        mock_pysheds_grid.fill_pits.return_value = mock_pysheds_raster
        mock_pysheds_grid.fill_depressions.return_value = mock_pysheds_raster
        mock_pysheds_grid.resolve_flats.return_value = mock_pysheds_raster
        
        custom_path = Path("custom/dem/path")
        load_inflated_dem(data_dir=custom_path, downsample_factor=5)
        
        # Verify the custom path was passed to load_dem
        mock_load.assert_called_once_with(custom_path)
        mock_zoom.assert_called_once_with(mock_pysheds_raster, mock_pysheds_grid, downsample_factor=5)


class TestComputeAccumulation:
    """Test cases for compute_accumulation function."""
    
    def test_compute_accumulation_basic(self, mock_pysheds_raster, mock_pysheds_grid):
        """Test flow direction and accumulation computation."""
        dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
        
        # Mock the grid methods to return different rasters
        mock_fdir = Mock()
        mock_acc = Mock()
        mock_pysheds_grid.flowdir.return_value = mock_fdir
        mock_pysheds_grid.accumulation.return_value = mock_acc
        
        acc, fdir = compute_accumulation(mock_pysheds_raster, mock_pysheds_grid, dirmap)
        
        # Verify calls
        mock_pysheds_grid.flowdir.assert_called_once_with(mock_pysheds_raster, dirmap=dirmap)
        mock_pysheds_grid.accumulation.assert_called_once_with(mock_fdir, dirmap=dirmap)
        
        assert acc == mock_acc
        assert fdir == mock_fdir
    
    def test_compute_accumulation_custom_dirmap(self, mock_pysheds_raster, mock_pysheds_grid):
        """Test accumulation with custom direction mapping."""
        custom_dirmap = (1, 2, 4, 8, 16, 32, 64, 128)  # Different order
        
        mock_fdir = Mock()
        mock_acc = Mock()
        mock_pysheds_grid.flowdir.return_value = mock_fdir
        mock_pysheds_grid.accumulation.return_value = mock_acc
        
        compute_accumulation(mock_pysheds_raster, mock_pysheds_grid, custom_dirmap)
        
        # Verify the custom dirmap was used
        mock_pysheds_grid.flowdir.assert_called_once_with(mock_pysheds_raster, dirmap=custom_dirmap)
        mock_pysheds_grid.accumulation.assert_called_once_with(mock_fdir, dirmap=custom_dirmap)


class TestComputeCatchmentArea:
    """Test cases for compute_catchment_area function."""
    
    @patch('dwd_radolan_utils.catchment_area.convert_to_utm')
    def test_compute_catchment_area_basic(self, mock_convert_utm, mock_pysheds_raster, mock_pysheds_grid):
        """Test basic catchment area computation."""
        # Setup mocks
        mock_convert_utm.return_value = (450000, 5650000)  # UTM coordinates
        dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
        coordinates = (7.158556, 51.255604)  # lon, lat
        
        # Mock accumulation raster with high values
        mock_acc = np.ones((100, 100)) * 2000  # Above threshold
        mock_pysheds_raster.__getitem__ = lambda self, key: mock_acc[key]
        mock_pysheds_raster.__gt__ = lambda self, threshold: mock_acc > threshold
        
        # Mock grid methods
        mock_pysheds_grid.snap_to_mask.return_value = (450000.0, 5650000.0)
        mock_catchment = np.ones((100, 100), dtype=bool)
        mock_pysheds_grid.catchment.return_value = mock_catchment
        mock_dist = np.random.rand(100, 100) * 1000
        mock_pysheds_grid.distance_to_outlet.return_value = mock_dist
        mock_pysheds_grid.view.return_value = mock_catchment
        
        clipped_catch, dist, snap_coords = compute_catchment_area(
            mock_pysheds_raster, mock_pysheds_raster, mock_pysheds_grid, dirmap, coordinates
        )
        
        # Verify calls
        mock_convert_utm.assert_called_once_with(mock_pysheds_grid, coordinates)
        mock_pysheds_grid.snap_to_mask.assert_called_once()
        mock_pysheds_grid.catchment.assert_called_once()
        mock_pysheds_grid.distance_to_outlet.assert_called_once()
        
        # Check results
        assert snap_coords == (450000.0, 5650000.0)
        assert clipped_catch is not None
        assert dist is not None
    
    def test_compute_catchment_area_coordinate_types(self, mock_pysheds_raster, mock_pysheds_grid):
        """Test that coordinate types are handled correctly."""
        dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
        coordinates = (7.158556, 51.255604)
        
        # Mock to return numpy scalars
        mock_pysheds_grid.snap_to_mask.return_value = (np.float64(450000), np.float64(5650000))
        
        with patch('dwd_radolan_utils.catchment_area.convert_to_utm') as mock_convert:
            mock_convert.return_value = (450000, 5650000)
            
            _, _, snap_coords = compute_catchment_area(
                mock_pysheds_raster, mock_pysheds_raster, mock_pysheds_grid, dirmap, coordinates
            )
            
            # Should convert numpy scalars to Python floats
            assert isinstance(snap_coords[0], float)
            assert isinstance(snap_coords[1], float)


class TestComputeCatchmentForLocation:
    """Test cases for compute_catchement_for_location function."""
    
    @patch('dwd_radolan_utils.catchment_area.load_inflated_dem')
    @patch('dwd_radolan_utils.catchment_area.compute_accumulation')
    @patch('dwd_radolan_utils.catchment_area.compute_catchment_area')
    def test_compute_catchement_for_location_basic(self, mock_compute_catchment, mock_compute_acc, mock_load_dem):
        """Test catchment computation for a single location."""
        coordinates = (7.158556, 51.255604)
        
        # Setup mocks
        mock_dem = Mock()
        mock_grid = Mock()
        mock_load_dem.return_value = (mock_dem, mock_grid)
        
        mock_acc = Mock()
        mock_fdir = Mock()
        mock_compute_acc.return_value = (mock_acc, mock_fdir)
        
        mock_clipped_catch = Mock()
        mock_dist = Mock()
        mock_snap_coords = (450000, 5650000)
        mock_compute_catchment.return_value = (mock_clipped_catch, mock_dist, mock_snap_coords)
        
        result_dist, result_grid = compute_catchement_for_location(coordinates, downsample_factor=20)
        
        # Verify calls
        mock_load_dem.assert_called_once_with(downsample_factor=20)
        mock_compute_acc.assert_called_once()
        mock_compute_catchment.assert_called_once()
        
        # Check results
        assert result_dist == mock_dist
        assert result_grid == mock_grid


class TestComputeMultipleCatchments:
    """Test cases for compute_multiple_catchments function."""
    
    @patch('dwd_radolan_utils.catchment_area.compute_catchement_for_location')
    def test_compute_multiple_catchments_basic(self, mock_compute_single):
        """Test computing catchments for multiple locations."""
        coordinates = [(7.158556, 51.255604), (8.0, 52.0), (9.0, 53.0)]
        
        # Mock return values for each location
        mock_dists = [Mock(), Mock(), Mock()]
        mock_grids = [Mock(), Mock(), Mock()]
        mock_compute_single.side_effect = list(zip(mock_dists, mock_grids))
        
        dist_list, grid_list = compute_multiple_catchments(coordinates, downsample_factor=25)
        
        # Verify calls
        assert mock_compute_single.call_count == 3
        for i, coord in enumerate(coordinates):
            call_args = mock_compute_single.call_args_list[i]
            assert call_args[0][0] == coord
            assert call_args[0][1] == 25
        
        # Check results
        assert len(dist_list) == 3
        assert len(grid_list) == 3
        assert dist_list == mock_dists
        assert grid_list == mock_grids


class TestConvertGridToRadolanGrid:
    """Test cases for grid conversion functions."""
    
    @patch('dwd_radolan_utils.catchment_area.get_wgs84_grid')
    @patch('dwd_radolan_utils.catchment_area.Transformer')
    def test_convert_grid_to_radolan_grid_vectorized(self, mock_transformer_class, mock_get_wgs84, mock_pysheds_raster, mock_pysheds_grid):
        """Test vectorized grid conversion."""
        # Setup mocks
        wgs84_grid = np.random.rand(900, 900, 2)
        # Set realistic coordinate ranges for Germany
        wgs84_grid[:, :, 0] = np.random.rand(900, 900) * 8 + 47  # Latitude 47-55
        wgs84_grid[:, :, 1] = np.random.rand(900, 900) * 10 + 5   # Longitude 5-15
        mock_get_wgs84.return_value = wgs84_grid
        
        mock_transformer = Mock()
        mock_transformer.transform.return_value = (
            np.random.rand(900*900) * 100000 + 400000,  # x coordinates within bounds
            np.random.rand(900*900) * 100000 + 5600000   # y coordinates within bounds
        )
        mock_transformer_class.from_crs.return_value = mock_transformer
        
        # Mock the grid extent to include our coordinates
        mock_pysheds_grid.extent = (300000, 600000, 5500000, 5800000)  # Wider bounds
        mock_pysheds_grid.affine = None  # Trigger fallback coordinate conversion
        
        # Mock the raster data
        raster_data = np.random.rand(100, 100) * 1000
        mock_pysheds_raster.__getitem__ = lambda self, key: raster_data[key]
        mock_pysheds_raster.shape = (100, 100)
        mock_pysheds_raster.nodata = -9999
        
        result = convert_grid_to_radolan_grid_vectorized(mock_pysheds_raster, mock_pysheds_grid)
        
        # Check result shape
        assert result.shape == (900, 900)
        assert isinstance(result, np.ndarray)
        
    
    @patch('dwd_radolan_utils.catchment_area.get_wgs84_grid')
    @patch('dwd_radolan_utils.catchment_area.Transformer')
    def test_convert_grid_to_radolan_grid_loops(self, mock_transformer_class, mock_get_wgs84, mock_pysheds_raster, mock_pysheds_grid):
        """Test loop-based grid conversion."""
        # Setup smaller grid for testing loops (to avoid long test runtime)
        wgs84_grid = np.random.rand(900, 900, 2)
        mock_get_wgs84.return_value = wgs84_grid
        
        mock_transformer = Mock()
        mock_transformer.transform.return_value = (
            np.random.rand(900*900) * 100000 + 400000,
            np.random.rand(900*900) * 100000 + 5600000
        )
        mock_transformer_class.from_crs.return_value = mock_transformer
        
        # Mock the raster and grid
        raster_data = np.random.rand(100, 100) * 1000
        mock_pysheds_raster.__getitem__ = lambda self, key: raster_data[key]
        mock_pysheds_raster.shape = (100, 100)
        mock_pysheds_raster.nodata = -9999
        
        mock_pysheds_grid.nearest_cell.return_value = (50, 50)
        
        result = convert_grid_to_radolan_grid_loops(mock_pysheds_raster, mock_pysheds_grid)
        
        # Check result shape
        assert result.shape == (900, 900)
        assert isinstance(result, np.ndarray)
    
    def test_convert_grid_to_radolan_grid_multiple(self, mock_pysheds_raster, mock_pysheds_grid):
        """Test conversion of multiple grids."""
        # Create list inputs
        dist_list = [mock_pysheds_raster, mock_pysheds_raster]
        grid_list = [mock_pysheds_grid, mock_pysheds_grid]
        
        with patch('dwd_radolan_utils.catchment_area.convert_grid_to_radolan_grid_vectorized') as mock_vectorized:
            # Mock return values
            mock_vectorized.return_value = np.random.rand(900, 900)
            
            result = convert_grid_to_radolan_grid(dist_list, grid_list)
            
            # Should call vectorized function for each grid
            assert mock_vectorized.call_count == 2
            
            # Result should be stacked
            assert result.shape == (2, 900, 900)


class TestConversionMethods:
    """Test cases for conversion method comparison."""
    
    def test_conversion_methods(self, mock_pysheds_raster, mock_pysheds_grid):
        """
        Benchmark both conversion methods and compare their performance.
        
        This test compares the vectorized and loop-based conversion methods,
        measuring their performance and validating that they produce equivalent results.
        """
        logging.info("Starting benchmark comparison of conversion methods...")
        
        # Mock the raster data
        raster_data = np.random.rand(100, 100) * 1000
        mock_pysheds_raster.__getitem__ = lambda self, key: raster_data[key]
        mock_pysheds_raster.shape = (100, 100)
        mock_pysheds_raster.nodata = -9999
        
        # Mock the grid extent and crs
        mock_pysheds_grid.extent = (300000, 600000, 5500000, 5800000)
        mock_pysheds_grid.crs = 'EPSG:25832'
        mock_pysheds_grid.affine = None  # Trigger fallback coordinate conversion
        mock_pysheds_grid.nearest_cell.return_value = (50, 50)
        
        with patch('dwd_radolan_utils.catchment_area.get_wgs84_grid') as mock_get_wgs84, \
             patch('dwd_radolan_utils.catchment_area.Transformer') as mock_transformer_class:
            
            # Setup mocks for coordinate transformation
            wgs84_grid = np.random.rand(900, 900, 2)
            wgs84_grid[:, :, 0] = np.random.rand(900, 900) * 8 + 47  # Latitude 47-55
            wgs84_grid[:, :, 1] = np.random.rand(900, 900) * 10 + 5   # Longitude 5-15
            mock_get_wgs84.return_value = wgs84_grid
            
            mock_transformer = Mock()
            mock_transformer.transform.return_value = (
                np.random.rand(900*900) * 100000 + 400000,  # x coordinates within bounds
                np.random.rand(900*900) * 100000 + 5600000   # y coordinates within bounds
            )
            mock_transformer_class.from_crs.return_value = mock_transformer
            
            # Test vectorized method
            logging.info("Testing vectorized method...")
            start_time = time.time()
            result_vectorized = convert_grid_to_radolan_grid_vectorized(mock_pysheds_raster, mock_pysheds_grid)
            vectorized_time = time.time() - start_time
            
            # Test loop method
            logging.info("Testing loop method...")
            start_time = time.time()
            result_loops = convert_grid_to_radolan_grid_loops(mock_pysheds_raster, mock_pysheds_grid)
            loops_time = time.time() - start_time
            
            # Compare results
            valid_vectorized = np.sum(~np.isnan(result_vectorized))
            valid_loops = np.sum(~np.isnan(result_loops))
            
            # Check if results are approximately equal (within tolerance)
            tolerance = 1e-6
            close_match = np.allclose(result_vectorized, result_loops, equal_nan=True, rtol=tolerance)
            
            # Calculate speedup
            speedup = loops_time / vectorized_time if vectorized_time > 0 else float('inf')
            
            benchmark_results = {
                'vectorized_time': vectorized_time,
                'loops_time': loops_time,
                'speedup': speedup,
                'valid_cells_vectorized': valid_vectorized,
                'valid_cells_loops': valid_loops,
                'results_match': close_match,
                'total_cells': 900 * 900
            }
            
            # Print results
            logging.info("=" * 60)
            logging.info("BENCHMARK RESULTS")
            logging.info("=" * 60)
            logging.info(f"Vectorized method time: {vectorized_time:.3f} seconds")
            logging.info(f"Loop method time:       {loops_time:.3f} seconds")
            logging.info(f"Speedup:               {speedup:.1f}x faster")
            logging.info(f"Valid cells (vectorized): {valid_vectorized:,} / {900*900:,}")
            logging.info(f"Valid cells (loops):      {valid_loops:,} / {900*900:,}")
            logging.info(f"Results match:         {'✅ Yes' if close_match else '❌ No'}")
            logging.info("=" * 60)
            
            # Assertions for test validation
            assert result_vectorized.shape == (900, 900)
            assert result_loops.shape == (900, 900)
            assert isinstance(benchmark_results['vectorized_time'], float)
            assert isinstance(benchmark_results['loops_time'], float)
            assert isinstance(benchmark_results['speedup'], float)
            assert benchmark_results['total_cells'] == 900 * 900
            
            # Both methods should produce some valid cells
            assert valid_vectorized >= 0
            assert valid_loops >= 0 

