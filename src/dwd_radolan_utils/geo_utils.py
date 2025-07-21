import numpy as np
import pandas as pd
import xarray as xr
import pyproj
from pyproj import CRS

RADOLAN_WKT = """PROJCS["Radolan projection",
GEOGCS["Radolan Coordinate System",
    DATUM["Radolan Kugel",
        SPHEROID["Erdkugel", 6370040.0, 0.0]
    ],
    PRIMEM["Greenwich", 0.0, AUTHORITY["EPSG","8901"]],
    UNIT["degree", 0.017453292519943295],
    AXIS["Longitude", EAST],
    AXIS["Latitude", NORTH]
],
PROJECTION["Stereographic_North_Pole"],
PARAMETER["central_meridian", 10.0],
PARAMETER["Standard_Parallel_1", 60.0],
PARAMETER["scale_factor", 1.0],
PARAMETER["false_easting", 0.0],
PARAMETER["false_northing", 0.0],
UNIT["km", 1000.0],
AXIS["X", EAST],
AXIS["Y", NORTH],
AUTHORITY["EPSG","1000001"]
]
"""



def turn_df_to_xarray(pd_dataframe: pd.DataFrame) -> xr.Dataset:
    """
    Convert Pandas DataFrame back to xarray.Dataset, adding attributes.

    Args:
        pd_dataframe (pd.DataFrame): Pandas DataFrame containing rain data.

    Returns:
        xarray.Dataset: An xarray.Dataset containing rain data with added attributes.
    """

    my_xarray = xr.Dataset.from_dataframe(pd_dataframe.set_index(['y', 'x', 'time']))
    my_xarray['y'].attrs = {'standard_name': 'projection_y_coordinate', 'long_name': 'y coordinate of projection', 'units': 'km'}
    my_xarray['x'].attrs = {'standard_name': 'projection_x_coordinate', 'long_name': 'x coordinate of projection', 'units': 'km'}
    my_xarray['time'].attrs = {'standard_name': 'time'}
    my_xarray['RW'].attrs = {'valid_min': 0, 'valid_max': 100, 'standard_name': 'rainfall_rate', 'long_name': 'RW', 'unit': 'mm h-1'}
    return my_xarray


def convert_radolan_to_wgs84(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Converts coordinates from the Radolan coordinate reference system (CRS) to the WGS84 CRS.

    Args:
        x (np.ndarray): Array of x-coordinates in the Radolan CRS.
        y (np.ndarray): Array of y-coordinates in the Radolan CRS.

    Returns:
    Tuple[np.ndarray, np.ndarray]: Tuple containing the converted x-coordinates and y-coordinates in the WGS84 CRS.
    """

    radolan_crs = CRS.from_wkt(RADOLAN_WKT)
    wgs84_crs = CRS.from_epsg(4326)

    transformer = pyproj.Transformer.from_crs(radolan_crs, wgs84_crs, always_xy=True)

    return transformer.transform(x, y)

def get_wgs84_grid() -> np.ndarray:
    """
    Returns a grid of WGS84 coordinates.

    Returns:
        wgs84_grid (numpy.ndarray): A grid of WGS84 coordinates with shape (900, 900, 2).
                                    The first dimension represents latitude, the second dimension represents longitude,
                                    and the third dimension represents the coordinates.
    """
    x_radolan_coords = np.arange(-522.9621669218559, 376.0378330781441+0.1, 1.0)
    y_radolan_coords = np.arange(-4658.144724265571,  -3759.1447242655713+0.1, 1.0)

    wgs84_coords = convert_radolan_to_wgs84(x_radolan_coords, y_radolan_coords)
    wgs84_coords = np.array(wgs84_coords).T
    wgs84_coords = np.flip(wgs84_coords, axis=1)

    lat = np.repeat(wgs84_coords[:, 0], 900).reshape(900, 900)
    lon = np.tile(wgs84_coords[:, 1], 900).T.reshape(900, 900)

    wgs84_grid = np.stack([lat, lon], axis=2)
    return wgs84_grid
    

def cut_out_shapes(x: np.ndarray, min_dim_1: int, max_dim_1: int, min_dim_2: int, max_dim_2: int) -> np.ndarray:
    """
    Cuts out shapes from a given array based on the specified dimensions.

    Args:
        x (np.ndarray): The input array with shape: (n, 900, 900) from which shapes will be cut out.
        min_dim_1 (int): Minimum dimension 1.
        max_dim_1 (int): Maximum dimension 1.
        min_dim_2 (int): Minimum dimension 2.
        max_dim_2 (int): Maximum dimension 2.

    Returns:
    np.ndarray: The array with shape: (n, max_dim_1-min_dim_1, max_dim_2-min_dim_2) containing the cut-out shapes.
    """

    if len(x.shape) == 3:
        return x[:, min_dim_1:max_dim_1, min_dim_2:max_dim_2]
    elif len(x.shape) == 2:
        return x[min_dim_1:max_dim_1, min_dim_2:max_dim_2]
    else:
        raise ValueError("Input array must have either 2 or 3 dimensions.")


