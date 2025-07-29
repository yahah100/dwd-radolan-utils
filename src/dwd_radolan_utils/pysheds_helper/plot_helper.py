import contextily as ctx
import numpy as np
import seaborn as sns
from matplotlib import colors
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pysheds.sgrid import sGrid as Grid
from pysheds.sview import Raster

from dwd_radolan_utils.pysheds_helper.utils import zoom_dem


def plot_catchment(grid: Grid, clipped_catch: Raster):
    """Plot a catchment area with colorbar and terrain visualization.

    Args:
        grid: Pysheds grid object containing spatial reference information
        catch: Catchment area raster data
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_alpha(0)

    plt.grid(True, zorder=0)
    ax.imshow(
        np.where(clipped_catch, clipped_catch, np.nan),
        extent=grid.extent,
        zorder=1,
        cmap="Greys_r",
    )
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Delineated Catchment", size=14)


def plot_dem(dem: Raster, grid: Grid, downsample_factor: int = 4):
    """Plot a digital elevation model with colorbar and terrain visualization.

    Args:
        dem: Digital elevation model raster data
        grid: Pysheds grid object containing spatial reference information
        downsample_factor: Factor by which to downsample the DEM for faster plotting (default: 4)
    """
    dem_zoom, grid_zoom = zoom_dem(dem, downsample_factor)

    dem_zoom[dem_zoom < -100] = np.nan

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_alpha(0)

    plt.imshow(dem_zoom, extent=grid.extent, cmap="terrain", zorder=1)
    plt.colorbar(label="Elevation (m)")
    plt.grid(zorder=0)
    plt.title("Digital elevation map", size=14)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()


def plot_flow_direction(fdir: Raster, grid: Grid, dirmap: tuple, downsample_factor: int = 1):
    """Plot flow direction grid showing water flow directions across the terrain.

    Args:
        fdir: Flow direction raster data
        grid: Pysheds grid object containing spatial reference information
        dirmap: Tuple mapping flow direction values to their meanings
        downsample_factor: Factor by which to downsample the data for plotting (default: 1)
    """
    if downsample_factor > 1:
        fdir_zoom, _ = zoom_dem(fdir, downsample_factor)
    else:
        fdir_zoom = fdir

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_alpha(0)

    plt.imshow(fdir_zoom, extent=grid.extent, cmap="viridis", zorder=2)
    boundaries = [0] + sorted(dirmap)
    plt.colorbar(boundaries=boundaries, values=sorted(dirmap))
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Flow direction grid", size=14)
    plt.grid(zorder=-1)
    plt.tight_layout()


def plot_flow_accumulation(grid: Grid, acc: Raster):
    """Plot flow accumulation showing the number of upstream cells flowing into each cell.

    Args:
        grid: Pysheds grid object containing spatial reference information
        acc: Flow accumulation raster data
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_alpha(0)
    plt.grid(True, zorder=0)
    im = ax.imshow(
        acc,
        extent=grid.extent,
        zorder=2,
        cmap="cubehelix",
        norm=colors.LogNorm(1, acc.max()),
        interpolation="bilinear",
    )
    plt.colorbar(im, ax=ax, label="Upstream Cells")
    plt.title("Flow Accumulation", size=14)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()


def plot_simple_branches(grid: Grid, branches: Raster):
    """Plot stream network branches derived from flow accumulation analysis.

    Args:
        grid: Pysheds grid object containing spatial reference information
        branches: Raster data containing stream network branches as geographic features
    """
    sns.set_palette("husl")
    fig, ax = plt.subplots(figsize=(8.5, 6.5))

    plt.xlim(grid.bbox[0], grid.bbox[2])
    plt.ylim(grid.bbox[1], grid.bbox[3])
    ax.set_aspect("equal")

    for branch in branches["features"]:
        line = np.asarray(branch["geometry"]["coordinates"])
        plt.plot(line[:, 0], line[:, 1])

    _ = plt.title("D8 channels", size=14)


def plot_simple_distance_map(grid: Grid, dist: Raster):
    """Plot a simple flow distance map showing distance to outlet in cells.

    Args:
        grid: Pysheds grid object containing spatial reference information
        dist: Flow distance raster data
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_alpha(0)
    plt.grid(True, zorder=0)
    im = ax.imshow(dist, extent=grid.extent, zorder=2, cmap="cubehelix_r")
    plt.colorbar(im, ax=ax, label="Distance to outlet (cells)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Flow Distance", size=14)


def plot_distance_catchment_area(grid: Grid, dist: Raster, x_snap: float, y_snap: float, padding: int = 500):
    """Plot detailed flow distance analysis with basemap and enhanced visualization.

    Creates a comprehensive visualization of flow distance to outlet with contextual basemap,
    proper UTM coordinate display, and detailed statistics.

    Args:
        grid: Pysheds grid object containing spatial reference information
        dist: Flow distance raster data
        x_snap: X coordinate of the pour point (outlet) in grid's coordinate system
        y_snap: Y coordinate of the pour point (outlet) in grid's coordinate system
        padding: Padding in meters to add around the data extent (default: 500)
    """
    colors_list = [
        "#2E8B57",
        "#32CD32",
        "#FFD700",
        "#FF8C00",
        "#FF4500",
        "#8B0000",
    ]  # Green to red
    n_bins = 100
    flow_cmap = LinearSegmentedColormap.from_list("flow_distance", colors_list, N=n_bins)

    # Prepare flow distance data (remove infinite values)
    dist_plot = dist.copy()
    dist_plot[dist_plot == np.inf] = np.nan
    dist_plot[dist_plot == 0] = np.nan

    # Get the actual extent - prioritize distance data extent if available
    if hasattr(dist, "extent"):
        data_extent = dist.extent
        data_crs = dist.crs if hasattr(dist, "crs") else grid.crs
        print(f"Using distance data extent: {data_extent}")
    else:
        data_extent = grid.extent
        data_crs = grid.crs
        print(f"Using grid extent: {data_extent}")

    print(f"Distance data shape: {dist_plot.shape}")
    print(f"Distance data CRS: {data_crs}")
    print(f"Pour point: ({x_snap}, {y_snap})")

    # Create the plot with better extent handling
    fig, ax = plt.subplots(figsize=(16, 12))

    # Add padding to the actual data extent
    x_min, x_max, y_min, y_max = data_extent

    # Set the plot extent with padding
    ax.set_xlim(x_min - padding, x_max + padding)
    ax.set_ylim(y_min - padding, y_max + padding)

    print(f"Plot extent with padding: X({x_min - padding:.0f}, {x_max + padding:.0f}), Y({y_min - padding:.0f}, {y_max + padding:.0f})")

    # Add basemap with proper parameters
    try:
        ctx.add_basemap(
            ax,
            crs=data_crs,  # Use the actual data CRS
            zoom="auto",  # Auto zoom level
            attribution_size=8,
        )
        print("✅ Basemap loaded successfully")
    except Exception as e:
        print(f"⚠️ Could not load basemap: {e}")
        # Add a simple background color if basemap fails
        ax.set_facecolor("#f0f0f0")

    # Plot flow distance with proper settings using the actual data extent
    im = ax.imshow(
        dist_plot,
        extent=data_extent,  # Use the actual distance data extent
        cmap=flow_cmap,
        alpha=0.8,  # Semi-transparent to show basemap
        interpolation="bilinear",
        zorder=2,
    )  # On top of basemap

    # Add colorbar with better formatting
    cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02, aspect=30)
    cbar.set_label("Flow Distance to Outlet (cells)", fontsize=14, fontweight="bold")
    cbar.ax.tick_params(labelsize=12)

    # Add pour point marker with better visibility
    ax.plot(
        x_snap,
        y_snap,
        "o",
        color="red",
        markersize=20,
        markeredgecolor="white",
        markeredgewidth=3,
        label="Pour Point (Outlet)",
        zorder=4,
    )

    # Styling and labels
    ax.set_xlabel("Easting (UTM Zone 32N, meters)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Northing (UTM Zone 32N, meters)", fontsize=14, fontweight="bold")
    ax.set_title(
        "Hydrological Flow Distance Analysis\nKluse Discharge Point, North Rhine-Westphalia",
        fontsize=18,
        fontweight="bold",
        pad=25,
    )

    # Format axes with proper UTM coordinates
    ax.ticklabel_format(style="plain", axis="both")
    ax.tick_params(labelsize=12)

    # Add coordinate grid
    ax.grid(True, alpha=0.4, linestyle="--", zorder=1)

    # Add legend with better positioning
    ax.legend(loc="upper left", fontsize=14, framealpha=0.9, fancybox=True, shadow=True)

    # Set equal aspect ratio
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.show()

    # Enhanced statistics
    print("\n📊 Enhanced Flow Distance Statistics:")
    print(f"   🎯 Pour Point: ({x_snap:.0f}, {y_snap:.0f}) UTM")
    print(f"   📏 Min distance: {np.nanmin(dist_plot):.1f} cells")
    print(f"   📏 Max distance: {np.nanmax(dist_plot):.1f} cells")
    print(f"   📊 Mean distance: {np.nanmean(dist_plot):.1f} cells")
    print(f"   📦 Catchment area: {np.sum(~np.isnan(dist_plot))} cells")
    print(f"   🗺️  Data extent: {data_extent}")

    # Convert to real-world units (each cell represents downsample_factor meters)
    cell_size = 5  # 5x downsampled from 1m DEM
    print("   🌍 Real distances (approx):")
    print(f"      Max flow path: {np.nanmax(dist_plot) * cell_size:.0f} meters")
    print(f"      Catchment area: {np.sum(~np.isnan(dist_plot)) * cell_size**2 / 10000:.1f} hectares")


def plot_flow_direction_with_basemap(
    grid: Grid,
    fdir: Raster,
    dirmap: tuple,
    downsample_factor: int = 1,
    padding: int = 100,
):
    """Plot flow direction analysis with OpenStreetMap basemap and enhanced visualization.

    Creates a comprehensive visualization of flow directions with contextual basemap,
    proper UTM coordinate display, and professional styling.

    Args:
        grid: Pysheds grid object containing spatial reference information
        fdir: Flow direction raster data
        dirmap: Tuple mapping flow direction values to their meanings
        downsample_factor: Factor by which to downsample the data for plotting (default: 1)
        padding: Padding in meters to add around the data extent (default: 100)
    """
    # Apply downsampling if requested
    if downsample_factor > 1:
        fdir_plot, grid_plot = zoom_dem(fdir, downsample_factor)
    else:
        fdir_plot = fdir
        grid_plot = grid

    # Get the actual extent of the flow direction data
    data_extent = grid_plot.extent

    # Make flow values of 0 transparent by setting them to NaN
    # Convert to float to allow NaN values
    fdir_plot_display = fdir_plot.copy().astype(np.float32)
    fdir_plot_display[fdir_plot_display == 0] = np.nan

    print(f"Flow direction data extent: {data_extent}")
    print(f"Flow direction data shape: {fdir_plot.shape}")
    print(f"Flow direction CRS: {grid_plot.crs}")
    print(f"Flow direction values range: {fdir_plot.min()} to {fdir_plot.max()}")
    print(f"Non-zero flow cells: {np.sum(fdir_plot != 0)} / {fdir_plot.size} ({100 * np.sum(fdir_plot != 0) / fdir_plot.size:.1f}%)")

    # Create the plot with better extent handling
    fig, ax = plt.subplots(figsize=(16, 12))

    # Add padding to the actual data extent
    x_min, x_max, y_min, y_max = data_extent

    # Set the plot extent with padding
    ax.set_xlim(x_min - padding, x_max + padding)
    ax.set_ylim(y_min - padding, y_max + padding)

    print(f"Plot extent with padding: X({x_min - padding:.0f}, {x_max + padding:.0f}), Y({y_min - padding:.0f}, {y_max + padding:.0f})")

    # Add basemap with proper parameters
    try:
        ctx.add_basemap(
            ax,
            crs=grid_plot.crs,  # Use the actual data grid CRS
            zoom="auto",  # Auto zoom level
            attribution_size=8,
        )
        print("✅ Basemap loaded successfully")
    except Exception as e:
        print(f"⚠️ Could not load basemap: {e}")
        # Add a simple background color if basemap fails
        ax.set_facecolor("#f0f0f0")

    # Create custom colormap for flow directions
    # Use a discrete colormap with distinct colors for each direction
    len(dirmap)

    # Plot flow directions with proper settings using the actual data extent
    im = ax.imshow(
        fdir_plot_display,  # Use the version with zero values set to NaN (transparent)
        extent=data_extent,  # Use the actual flow direction data extent
        cmap="viridis",
        alpha=0.7,  # Semi-transparent to show basemap
        interpolation="nearest",  # Use nearest neighbor for discrete data
        zorder=2,
    )  # On top of basemap

    # Add colorbar with direction labels
    boundaries = [0] + sorted(dirmap)
    cbar = plt.colorbar(im, boundaries=boundaries, values=sorted(dirmap))
    cbar.set_label("Flow Direction", fontsize=14, fontweight="bold")

    # Set colorbar ticks to match direction values
    direction_values = sorted(dirmap)
    cbar.set_ticks(direction_values)

    # Create direction labels (N, NE, E, SE, S, SW, W, NW)
    direction_labels = []
    direction_map = {
        1: "E",
        2: "SE",
        4: "S",
        8: "SW",
        16: "W",
        32: "NW",
        64: "N",
        128: "NE",
    }
    for val in direction_values:
        direction_labels.append(direction_map.get(val, str(val)))

    cbar.set_ticklabels(direction_labels)
    cbar.ax.tick_params(labelsize=10)

    # Styling and labels
    ax.set_xlabel("Easting (UTM Zone 32N, meters)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Northing (UTM Zone 32N, meters)", fontsize=14, fontweight="bold")
    ax.set_title(
        "Hydrological Flow Direction Analysis\nWater Flow Directions Across Terrain",
        fontsize=18,
        fontweight="bold",
        pad=25,
    )

    # Format axes with proper UTM coordinates
    ax.ticklabel_format(style="plain", axis="both")
    ax.tick_params(labelsize=12)

    # Add coordinate grid
    ax.grid(True, alpha=0.4, linestyle="--", zorder=1)

    # Set equal aspect ratio
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.show()

    # Enhanced statistics
    print("\n📊 Flow Direction Statistics:")
    print(f"   🧭 Unique directions: {len(np.unique(fdir_plot[~np.isnan(fdir_plot)]))} different flow directions")
    print(f"   📦 Grid cells: {fdir_plot.size} total cells")
    print(f"   🗺️  Data extent: {data_extent}")

    # Direction distribution (including zero values for complete statistics)
    unique_dirs, counts = np.unique(fdir_plot[~np.isnan(fdir_plot)], return_counts=True)
    print("   🎯 Direction distribution:")
    for dir_val, count in zip(unique_dirs, counts, strict=False):
        if dir_val == 0:
            dir_name = "No Flow (transparent)"
        else:
            dir_name = direction_map.get(int(dir_val), str(int(dir_val)))
        percentage = (count / np.sum(counts)) * 100
        print(f"      {dir_name}: {count:,} cells ({percentage:.1f}%)")
