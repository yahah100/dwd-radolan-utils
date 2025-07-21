import numpy as np
from scipy.ndimage import zoom
from pathlib import Path
import rasterio
from rasterio.merge import merge
from tqdm import tqdm
from pyproj import Transformer

import matplotlib.pyplot as plt
from pysheds.grid import Grid
from pysheds.sview import Raster as sRaster
from pysheds.pview import Raster as pRaster
from pysheds.view import ViewFinder

Raster = pRaster | sRaster

def zoom_dem(dem: Raster, grid: Grid, downsample_factor: int = 4) -> tuple[Raster, Grid]:
    """Downsample a DEM raster and corresponding grid by a specified factor.
    
    Reduces the resolution of the DEM data while preserving spatial relationships
    and metadata. Useful for faster processing and visualization of large datasets.
    
    Args:
        dem: Digital elevation model raster data to be downsampled
        grid: Pysheds grid object containing spatial reference information
        downsample_factor: Factor by which to reduce resolution (default: 4)
            A factor of 4 means the output will have 1/4 the resolution in each dimension
    
    Returns:
        tuple: (downsampled_raster, new_grid) where:
            - downsampled_raster: Raster object with reduced resolution
            - new_grid: Grid object with updated spatial parameters
    """
    if downsample_factor == 1:
        return dem, grid
    elif downsample_factor > 1:
        zoom_factor = 1.0 / downsample_factor
        
        # Keep the original dtype and nodata value
        original_dtype = dem.dtype
        original_nodata = dem.nodata
        
        # Convert to float for processing, but we'll convert back
        dem_data = dem.copy()
        dem_data[dem_data == original_nodata] = np.nan if original_dtype in [np.float32, np.float64] else 0
        
        # Apply zoom
        dem_zoom = zoom(dem_data, zoom_factor, order=1)
        dem_zoom[np.isnan(dem_zoom)] = original_nodata
        dem_zoom = dem_zoom.astype(original_dtype)

        # Calculate new affine transform
        old_affine = dem.affine
        new_affine = old_affine * old_affine.scale(downsample_factor, downsample_factor)
        
        # Resample the mask
        new_mask = zoom(dem.mask.astype(float), zoom_factor, order=0) > 0.5
        
        # Use the EXACT same nodata value and ensure it's the right numpy type
        nodata_value = original_dtype.type(original_nodata)
        
        new_viewfinder = ViewFinder(
            affine=new_affine,
            shape=dem_zoom.shape,
            crs=dem.crs,
            nodata=nodata_value,
            mask=new_mask
        )

        new_grid = Grid(new_viewfinder)
        dem_zoom = pRaster(dem_zoom, viewfinder=new_viewfinder)
    
        return dem_zoom, new_grid

    else:
        raise ValueError(f"Downsample factor must be greater than 0, got {downsample_factor}")

def load_dem(data_dir: Path) -> tuple[Raster, Grid]:
    """Load and merge multiple DEM TIFF files into a single mosaic.
    
    Searches for all .tif files in the specified directory and creates a combined
    mosaic. If a mosaic already exists, loads it directly. The mosaic is saved
    with LZW compression to reduce file size.
    
    Args:
        data_dir: Path to directory containing DEM TIFF files
    
    Returns:
        tuple: (dem_raster, grid) where:
            - dem_raster: Combined DEM raster data
            - grid: Pysheds grid object for the mosaic
    """
    tif_files = list(data_dir.glob('*.tif'))
    path = data_dir / "combined_dgm_mosaic.tif"
    if not path.exists():
        src_files_to_mosaic = []

        for i, tif_file in tqdm(enumerate(tif_files)):
            try:
                src = rasterio.open(str(tif_file))
                src_files_to_mosaic.append(src)
            except Exception as e:
                print(f"Error opening {tif_file}: {e}")


        mosaic, out_trans = merge(src_files_to_mosaic)

        print(f"Mosaic created with shape: {mosaic.shape}")
        print(f"Mosaic data type: {mosaic.dtype}")

        # Get metadata from the first file and update for the mosaic
        out_meta = src_files_to_mosaic[0].meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": mosaic.shape[1], 
            "width": mosaic.shape[2],
            "transform": out_trans,
            "compress": "lzw"  # Add compression to reduce file size
        })

        print(f"Mosaic metadata: {out_meta}")

        # Save the combined mosaic
        print(f"Saving mosaic to: {path}")

        with rasterio.open(path, "w", **out_meta) as dest:
            dest.write(mosaic)

        # Close all source files to free memory
        for src in src_files_to_mosaic:
            src.close()
    grid = Grid.from_raster(path, data_name='elevation')
    dem: Raster = grid.read_raster(path, data_name='elevation')
    return dem, grid

def convert_to_utm(grid: Grid, coords_wgs84: tuple[float, float]) -> tuple[float, float]:
    """Convert WGS84 coordinates to the grid's coordinate reference system.
    
    Transforms longitude/latitude coordinates to the spatial reference system
    used by the grid (typically UTM). Includes validation to ensure the converted
    coordinates fall within the grid extent.
    
    Args:
        grid: Pysheds grid object containing spatial reference information
        coords_wgs84: Tuple of (longitude, latitude) in WGS84 decimal degrees
    
    Returns:
        tuple: (x, y) coordinates in the grid's coordinate reference system
    """
    # Check grid coordinate system and extent
    print(f"Grid CRS: {grid.crs}")
    print(f"Grid extent: {grid.extent}")
    print(f"Target coordinates WGS84 (long, lat): {coords_wgs84}")


    # Create transformer from WGS84 to grid's CRS
    transformer = Transformer.from_crs("EPSG:4326", grid.crs, always_xy=True)

    # Transform coordinates (longitude, latitude) -> (x, y) in grid's CRS
    target_x, target_y = transformer.transform(coords_wgs84[0], coords_wgs84[1])
    target_coords = (target_x, target_y)

    print(f"Target coordinates in grid CRS (x, y): {target_coords}")

    # Check if converted coordinates are within grid extent
    x_min, x_max, y_min, y_max = grid.extent
    if x_min <= target_x <= x_max and y_min <= target_y <= y_max:
        print("✅ Target coordinates are within grid extent!")
    else:
        print("❌ Target coordinates are OUTSIDE grid extent!")
        print(f"   Grid X range: {x_min:.1f} to {x_max:.1f}")
        print(f"   Grid Y range: {y_min:.1f} to {y_max:.1f}")
        print(f"   Target X: {target_x:.1f}")
        print(f"   Target Y: {target_y:.1f}")
    return target_coords