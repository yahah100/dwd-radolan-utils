from pathlib import Path
import logging
import numpy as np
from pysheds.sgrid import sGrid
from pysheds.sview import Raster as sRaster
from pyproj import CRS, Transformer

from dwd_radolan_utils.pysheds_helper.utils import zoom_dem, convert_to_utm, load_dem
from dwd_radolan_utils.pysheds_helper.plot_helper import plot_distance_catchment_area
from dwd_radolan_utils.geo_utils import get_wgs84_grid

def load_inflated_dem(
    data_dir=Path("data/dgm"), downsample_factor: int = 10
) -> tuple[sRaster, sGrid]:
    """Load and process a DEM (Digital Elevation Model) by filling pits, depressions, and resolving flats.
    
    Args:
        data_dir (Path, optional): Directory containing the DEM mosaic file. 
            Defaults to Path("data/dgm").
        downsample_factor (int, optional): Factor by which to downsample the DEM for faster processing. 
            Defaults to 10.
    
    Returns:
        tuple[Raster, Grid]: A tuple containing:
            - inflated_dem (Raster): The processed DEM with pits filled, depressions filled, 
              and flats resolved
            - grid (Grid): The Grid object containing the processed DEM data
    """
    dem, grid = load_dem(data_dir)

    dem, grid = zoom_dem(dem, grid, downsample_factor=downsample_factor)

    logging.info(f"Filling pits")
    pit_filled_dem = grid.fill_pits(dem)

    logging.info(f"Filling depressions")
    flooded_dem = grid.fill_depressions(pit_filled_dem)

    logging.info(f"Resolving flats")
    inflated_dem: sRaster = grid.resolve_flats(flooded_dem)
    inflated_dem[inflated_dem < 0] = inflated_dem.nodata
    return inflated_dem, grid


def compute_accumulation(
    inflated_dem: sRaster, grid: sGrid, dirmap: tuple[int, ...]
) -> tuple[sRaster, sRaster]:
    """Compute flow directions and flow accumulation from a processed DEM.
    
    Args:
        inflated_dem (Raster): The processed DEM with pits filled and flats resolved
        grid (Grid): The Grid object containing the DEM data
        dirmap (tuple[int, ...]): Direction mapping tuple defining the flow direction encoding.
            Typically (64, 128, 1, 2, 4, 8, 16, 32) for D8 flow directions
    
    Returns:
        tuple[Raster, Raster]: A tuple containing:
            - acc (Raster): Flow accumulation raster showing the number of upstream cells
            - fdir (Raster): Flow direction raster showing the direction of steepest descent
    """
    logging.info(f"Computing flow directions")
    fdir: sRaster = grid.flowdir(inflated_dem, dirmap=dirmap)
    logging.info(f"Computing flow accumulation")
    acc: sRaster = grid.accumulation(fdir, dirmap=dirmap)
    return acc, fdir


def compute_catchment_area(
    fdir: sRaster,
    acc: sRaster,
    grid: sGrid,
    dirmap: tuple[int, ...],
    coordinates: tuple[float, float],
) -> tuple[sRaster, sRaster, tuple[float, float]]:
    """Compute the catchment area and distance to outlet for a given pour point.
    
    Args:
        fdir (Raster): Flow direction raster showing the direction of steepest descent
        acc (Raster): Flow accumulation raster showing the number of upstream cells
        grid (Grid): The Grid object containing the flow data
        dirmap (tuple[int, ...]): Direction mapping tuple defining the flow direction encoding.
            Typically (64, 128, 1, 2, 4, 8, 16, 32) for D8 flow directions
        coordinates (tuple[float, float]): Target coordinates (latitude, longitude) for the 
            pour point where the catchment will be delineated
    
    Returns:
        tuple[Raster, Raster, tuple[int, int]]: A tuple containing:
            - clipped_catch (Raster): Clipped catchment boundary raster
            - dist (Raster): Distance to outlet raster showing flow distances
            - (x_snap, y_snap) (tuple[int, int]): Snapped coordinates of the pour point
    """
    target_coords = convert_to_utm(grid, coordinates)

    acc_threshold = 1_000

    # Snap pour point to high accumulation cell
    x_snap_raw, y_snap_raw = grid.snap_to_mask(acc > acc_threshold, target_coords)
    # Convert to scalar values to fix type issues
    x_snap = float(x_snap_raw) if hasattr(x_snap_raw, 'item') else float(x_snap_raw)
    y_snap = float(y_snap_raw) if hasattr(y_snap_raw, 'item') else float(y_snap_raw)
    print(f"\nSnapped coordinates: x={x_snap}, y={y_snap}")

    # Delineate the catchment
    catch = grid.catchment(
        x=x_snap, y=y_snap, fdir=fdir, dirmap=dirmap, xytype="coordinate"
    )

    print(f"Catchment shape: {catch.shape}")
    print(f"Catchment True values: {np.sum(catch)}")

    # Create clipped view of the catchment
    grid.clip_to(catch)
    clipped_catch = grid.view(catch)

    dist = grid.distance_to_outlet(
        x=x_snap, y=y_snap, fdir=fdir, dirmap=dirmap, xytype="coordinate"
    )
    dist[dist == np.inf] = np.nan
    return clipped_catch, dist, (x_snap, y_snap)


def compute_catchement_for_location(coordinates: tuple[float, float], downsample_factor: int = 10) -> tuple[sRaster, sGrid]:
    """Compute the catchment area and distance to outlet for a given location.
    
    Args:
        coordinates (tuple[float, float]): Target coordinates (latitude, longitude) for the 
            pour point where the catchment will be delineated
        downsample_factor (int, optional): Factor by which to downsample the DEM for faster processing. 
            Defaults to 10.

    Returns:
        tuple[Raster, Grid, int, int]: A tuple containing:
            - dist (Raster): Distance to outlet raster showing flow distances
            - grid (Grid): The Grid object containing the flow data
    
    This function demonstrates the complete workflow for catchment area analysis:
    1. Loads and processes a DEM (downsampled by factor of 10)
    2. Computes flow directions and accumulation
    3. Delineates catchment area for the Kluse discharge station coordinates
    4. Plots the results including distance to outlet
    
    The example uses coordinates for Kluse discharge station (51.255604, 7.158556).
    """
    inflated_dem, grid = load_inflated_dem(downsample_factor=downsample_factor)

    dirmap = (64, 128, 1, 2, 4, 8, 16, 32)

    acc, fdir = compute_accumulation(inflated_dem, grid, dirmap)


    _, dist, _ = compute_catchment_area(
        fdir, acc, grid, dirmap, coordinates
    )

    # plot_distance_catchment_area(grid, dist, x_snap, y_snap)

    return dist, grid


def compute_multiple_catchments(coordinates: list[tuple[float, float]], downsample_factor: int = 50) -> tuple[sRaster, sGrid]:
    """Compute the catchment area and distance to outlet for a list of locations.
    
    Args:
        coordinates (list[tuple[float, float]]): List of target coordinates (latitude, longitude) for the 
            pour points where the catchment will be delineated
        downsample_factor (int, optional): Factor by which to downsample the DEM for faster processing. 
            Defaults to 50.
    
    Returns:
        tuple[Raster, Grid]: A tuple containing:
            - dist (Raster): Distance to outlet raster showing flow distances
            - grid (Grid): The Grid object containing the flow data
    """

    for coordinate in coordinates:
        dist, grid = compute_catchement_for_location(coordinate, downsample_factor)
    
    return dist, grid


def convert_grid_to_radolan_grid_loops(dist: sRaster, grid: sGrid) -> np.ndarray:
    """
    Project each cell of the distance raster to the radolan grid using nested loops (slower but simpler).
    Values where no distance is available are set to np.nan.
    
    This is the original implementation with nested loops for performance comparison.

    Args:
        dist (sRaster): Distance raster showing flow distances
        grid (sGrid): The Grid object containing the flow data
    
    Returns:
        np.ndarray: A grid of distance values in the radolan grid
    """
    
    wgs_84_grid = get_wgs84_grid()
    
    new_grid = np.full((900, 900), np.nan)
    dist_crs = grid.crs
    
    wgs84_crs = CRS.from_epsg(4326)
    transformer = Transformer.from_crs(wgs84_crs, dist_crs, always_xy=True)
    
    lons = wgs_84_grid[:, :, 1].flatten()
    lats = wgs_84_grid[:, :, 0].flatten()
    
    x_transformed, y_transformed = transformer.transform(lons, lats)
    
    x_transformed = x_transformed.reshape((900, 900))
    y_transformed = y_transformed.reshape((900, 900))
    
    # Get the bounds of the distance raster to check if coordinates are within bounds
    try:
        bounds = grid.extent  # (x_min, x_max, y_min, y_max)
        x_min, x_max, y_min, y_max = bounds
    except:
        x_min, x_max = float('-inf'), float('inf')
        y_min, y_max = float('-inf'), float('inf')
    
    successful_projections = 0
    
    # For each cell in the radolan grid, sample the distance raster
    for i in range(900):
        for j in range(900):
            x_coord = x_transformed[i, j]
            y_coord = y_transformed[i, j]
            
            # Check if the transformed coordinates are within the dist raster bounds
            if (x_min <= x_coord <= x_max and y_min <= y_coord <= y_max):
                try:
                    col, row = grid.nearest_cell(x_coord, y_coord)
                    
                    if (0 <= row < dist.shape[0] and 0 <= col < dist.shape[1]):
                        value = dist[row, col]
                        
                        if (not np.isnan(value) and 
                            hasattr(dist, 'nodata') and value != dist.nodata):
                            new_grid[i, j] = value
                            successful_projections += 1
                        elif not hasattr(dist, 'nodata') and not np.isnan(value):
                            new_grid[i, j] = value
                            successful_projections += 1
                            
                except Exception:
                    logging.warning(f"Warning: Could not project cell {i}, {j} to radolan grid {x_coord}, {y_coord}. Will be set to np.nan.")
                    continue
    
    logging.info(f"Loop method: Successfully projected {successful_projections} out of {900*900} cells to RADOLAN grid")
    return new_grid


def convert_grid_to_radolan_grid_vectorized(dist: sRaster, grid: sGrid) -> np.ndarray:
    """
    Project each cell of the distance raster to the radolan grid using vectorized operations (faster).
    Values where no distance is available are set to np.nan.
    
    This is the optimized vectorized implementation for better performance.

    Args:
        dist (sRaster): Distance raster showing flow distances
        grid (sGrid): The Grid object containing the flow data
    
    Returns:
        np.ndarray: A grid of distance values in the radolan grid
    """
    from pyproj import CRS, Transformer
    
    wgs_84_grid = get_wgs84_grid()
    
    new_grid = np.full((900, 900), np.nan)
    dist_crs = grid.crs
    
    wgs84_crs = CRS.from_epsg(4326)
    transformer = Transformer.from_crs(wgs84_crs, dist_crs, always_xy=True)
    
    lons = wgs_84_grid[:, :, 1].flatten()
    lats = wgs_84_grid[:, :, 0].flatten()
    
    x_transformed, y_transformed = transformer.transform(lons, lats)
    
    # Get the bounds of the distance raster
    try:
        bounds = grid.extent  # (x_min, x_max, y_min, y_max)
        x_min, x_max, y_min, y_max = bounds
    except:
        x_min, x_max = float('-inf'), float('inf')
        y_min, y_max = float('-inf'), float('inf')
    
    # Vectorized bounds checking
    within_bounds = ((x_transformed >= x_min) & (x_transformed <= x_max) & 
                     (y_transformed >= y_min) & (y_transformed <= y_max))
    
    # Convert coordinates to cell indices using the grid's affine transform
    try:
        if hasattr(grid, 'affine') and grid.affine is not None:
            # Use the affine transform (most accurate method)
            affine = grid.affine
            # Convert world coordinates to pixel coordinates using affine inverse
            inverse_affine = ~affine
            col_indices = np.round((x_transformed - affine.c) / affine.a).astype(int)
            row_indices = np.round((y_transformed - affine.f) / affine.e).astype(int)
        else:
            # Fallback: manual coordinate to index conversion
            dx = (x_max - x_min) / dist.shape[1]
            dy = (y_max - y_min) / dist.shape[0]
            
            col_indices = np.round((x_transformed - x_min) / dx).astype(int)
            row_indices = np.round((y_max - y_transformed) / dy).astype(int)  # y is flipped
    except Exception as e:
        logging.warning(f"Could not use affine transform, falling back to manual conversion: {e}")
        # Fallback method
        dx = (x_max - x_min) / dist.shape[1] if dist.shape[1] > 0 else 1.0
        dy = (y_max - y_min) / dist.shape[0] if dist.shape[0] > 0 else 1.0
        
        col_indices = np.round((x_transformed - x_min) / dx).astype(int)
        row_indices = np.round((y_max - y_transformed) / dy).astype(int)
    
    # Check if indices are within raster bounds
    valid_indices = (within_bounds & 
                     (row_indices >= 0) & (row_indices < dist.shape[0]) &
                     (col_indices >= 0) & (col_indices < dist.shape[1]))
    
    # Get the valid coordinates and indices
    valid_positions = np.where(valid_indices)[0]
    
    if len(valid_positions) > 0:
        # Sample all valid values at once using advanced indexing
        valid_rows = row_indices[valid_indices]
        valid_cols = col_indices[valid_indices]
        
        try:
            sampled_values = dist[valid_rows, valid_cols]
            
            # Check for nodata values
            if hasattr(dist, 'nodata') and dist.nodata is not None:
                valid_data_mask = ~np.isnan(sampled_values) & (sampled_values != dist.nodata)
            else:
                valid_data_mask = ~np.isnan(sampled_values)
            
            # Convert flat indices back to 2D indices for the output grid
            valid_final_positions = valid_positions[valid_data_mask]
            output_i, output_j = np.divmod(valid_final_positions, 900)
            
            # Assign valid values to the output grid
            new_grid[output_i, output_j] = sampled_values[valid_data_mask]
            
            logging.info(f"Vectorized method: Successfully projected {len(valid_final_positions)} out of {900*900} cells to RADOLAN grid")
            
        except Exception as e:
            logging.warning(f"Error during vectorized sampling, falling back to individual sampling: {e}")
            # If vectorized sampling fails, fall back to individual sampling for valid positions only
            for idx in valid_positions:
                try:
                    i, j = divmod(idx, 900)
                    row, col = row_indices[idx], col_indices[idx]
                    value = dist[row, col]
                    
                    if (not np.isnan(value) and 
                        (not hasattr(dist, 'nodata') or value != dist.nodata)):
                        new_grid[i, j] = value
                except Exception:
                    logging.warning(f"Warning: Could not project cell {i}, {j} to radolan grid. Will be set to np.nan.")
                    continue
    else:
        logging.warning("No valid coordinates found within the distance raster bounds")
    
    return new_grid


def convert_grid_to_radolan_grid(dist: sRaster, grid: sGrid) -> np.ndarray:
    """
    Project each cell of the distance raster to the radolan grid. 
    Values where no distance is available are set to np.nan.
    
    This function uses the faster vectorized implementation by default.
    For performance comparison, use convert_grid_to_radolan_grid_loops() or convert_grid_to_radolan_grid_vectorized() directly.

    Args:
        dist (sRaster): Distance raster showing flow distances
        grid (sGrid): The Grid object containing the flow data
    
    Returns:
        np.ndarray: A grid of distance values in the radolan grid
    """
    return convert_grid_to_radolan_grid_vectorized(dist, grid)


def benchmark_conversion_methods(dist: sRaster, grid: sGrid) -> dict:
    """
    Benchmark both conversion methods and compare their performance.
    
    Args:
        dist (sRaster): Distance raster showing flow distances
        grid (sGrid): The Grid object containing the flow data
    
    Returns:
        dict: Benchmark results including timing and validation information
    """
    import time
    
    logging.info("Starting benchmark comparison of conversion methods...")
    
    # Test vectorized method
    logging.info("Testing vectorized method...")
    start_time = time.time()
    result_vectorized = convert_grid_to_radolan_grid_vectorized(dist, grid)
    vectorized_time = time.time() - start_time
    
    # Test loop method
    logging.info("Testing loop method...")
    start_time = time.time()
    result_loops = convert_grid_to_radolan_grid_loops(dist, grid)
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
    
    return benchmark_results


def main():
    kluse_dis_wgs84 = (7.158556, 51.255604) 
    # compute_catchement_for_location(kluse_dis_wgs84, downsample_factor=50)
    dist, grid = compute_multiple_catchments([kluse_dis_wgs84], downsample_factor=50)
    
    # Benchmark both methods
    benchmark_results = benchmark_conversion_methods(dist, grid)
    
    new_grid = convert_grid_to_radolan_grid(dist, grid)

    # save as np file
    np.save("dist_map_kluse.npy", new_grid)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    logging.warning("Install pysheds with `uv pip install git+https://github.com/yahah100/pysheds.git`")
    main()
