from pathlib import Path
import logging
import numpy as np
from pysheds.sgrid import sGrid
from pysheds.sview import Raster as sRaster
from pyproj import CRS, Transformer

from dwd_radolan_utils.pysheds_helper.utils import zoom_dem, convert_to_utm
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
    mosaic_path = data_dir / "combined_dgm_mosaic.tif"

    logging.info(f"Loading mosaic from {mosaic_path}")
    grid = sGrid.from_raster(mosaic_path, data_name="elevation")
    dem: sRaster = grid.read_raster(mosaic_path, data_name="elevation")

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


def convert_grid_to_radolan_grid(dist: sRaster, grid: sGrid) -> np.ndarray:
    """
    Project each cell of the distance raster to the radolan grid. Values where no distance is available are set to np.nan.

    Args:
        dist (sRaster): Distance raster showing flow distances
        grid (sGrid): The Grid object containing the flow data
    
    Returns:
        np.ndarray: A grid of distance values in the radolan grid
    """
    # get the target grid with wgs84 coordinates
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
                        elif not hasattr(dist, 'nodata') and not np.isnan(value):
                            new_grid[i, j] = value
                            
                except Exception:
                    logging.warning(f"Warning: Could not project cell {i}, {j} to radolan grid {x_coord}, {y_coord}. Will be set to np.nan.")
                    continue
    
    return new_grid


def main():
    kluse_dis_wgs84 = (7.158556, 51.255604) 
    # compute_catchement_for_location(kluse_dis_wgs84, downsample_factor=50)
    dist, grid = compute_multiple_catchments([kluse_dis_wgs84], downsample_factor=50)
    new_grid = convert_grid_to_radolan_grid(dist, grid)

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    logging.warning("Install pysheds with `uv pip install git+https://github.com/yahah100/pysheds.git`")
    main()
