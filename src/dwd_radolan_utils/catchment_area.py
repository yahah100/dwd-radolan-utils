from pathlib import Path
import logging
import numpy as np
from pysheds.grid import Grid
from pysheds.pview import Raster as pRaster
from pysheds.sview import Raster as sRaster
from pysheds.view import ViewFinder

from dwd_radolan_utils.pysheds_helper.utils import zoom_dem, convert_to_utm
from dwd_radolan_utils.pysheds_helper.plot_helper import plot_distance_catchment_area

Raster = pRaster | sRaster

def load_inflated_dem(
    data_dir=Path("data/dgm"), downsample_factor: int = 10
) -> tuple[pRaster, Grid]:
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
    grid = Grid.from_raster(mosaic_path, data_name="elevation")
    grid.read_raster(mosaic_path, data_name="elevation")

    dem = grid.elevation
    dem, grid = zoom_dem(dem, grid, downsample_factor=downsample_factor)

    logging.info(f"Filling pits")
    pit_filled_dem = grid.fill_pits(dem)

    logging.info(f"Filling depressions")
    flooded_dem = grid.fill_depressions(pit_filled_dem)

    logging.info(f"Resolving flats")
    inflated_dem: pRaster = grid.resolve_flats(flooded_dem)
    inflated_dem[inflated_dem < 0] = inflated_dem.nodata
    return inflated_dem, grid


def compute_accumulation(
    inflated_dem: pRaster, grid: Grid, dirmap: tuple[int, ...]
) -> tuple[pRaster, pRaster]:
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
    fdir: pRaster = grid.flowdir(inflated_dem, dirmap=dirmap)
    logging.info(f"Computing flow accumulation")
    acc: pRaster = grid.accumulation(fdir, dirmap=dirmap)
    return acc, fdir


def compute_catchment_area(
    fdir: pRaster,
    acc: pRaster,
    grid: Grid,
    dirmap: tuple[int, ...],
    coordinates: tuple[float, float],
) -> tuple[pRaster, pRaster, tuple[int, int]]:
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
    x_snap, y_snap = grid.snap_to_mask(acc > acc_threshold, target_coords)
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


def compute_catchement_for_location(coordinates: tuple[float, float], downsample_factor: int = 10) -> tuple[pRaster, Grid]:
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


def compute_multiple_catchments(coordinates: list[tuple[float, float]], downsample_factor: int = 50) -> tuple[Raster, Grid]:
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


def convert_grid_to_radolan_grid(dist: pRaster, grid: Grid):
    pass
    


def main():
    kluse_dis_wgs84 = (7.158556, 51.255604) 
    # compute_catchement_for_location(kluse_dis_wgs84, downsample_factor=50)
    compute_multiple_catchments([kluse_dis_wgs84], downsample_factor=50)

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    logging.warning("Install pysheds with `uv pip install git+https://github.com/yahah100/pysheds.git`")
    main()
