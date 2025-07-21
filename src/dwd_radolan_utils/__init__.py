"""
DWD RADOLAN Utils - Python utilities for downloading and processing DWD RADOLAN radar data.

This package provides utilities for:
- Downloading DWD RADOLAN data
- Creating extraction grids from shapes and coordinates  
- Extracting time series from radar data
- Catchment area extraction using pysheds
"""

from .download import download_dwd as dwd_radolan_download
# from .grid import create_extraction_grid  # Function doesn't exist yet
# from .extraction import extract_time_series_from_radar  # Function doesn't exist yet
from .catchment_area import compute_catchement_for_location, compute_multiple_catchments

__version__ = "0.1.0"
__author__ = "Your Name"

__all__ = [
    "dwd_radolan_download",
    # "create_extraction_grid", 
    # "extract_time_series_from_radar",
    "compute_catchement_for_location",  
    "compute_multiple_catchments",
] 