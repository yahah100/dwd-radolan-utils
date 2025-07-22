"""
DWD RADOLAN Utils - Python utilities for downloading and processing DWD RADOLAN radar data.

This package provides utilities for:
- Downloading DWD RADOLAN data
- Creating extraction grids from shapes and coordinates
- Extracting time series from radar data
- Catchment area extraction using pysheds
"""

from .catchment_area import compute_catchement_for_location, compute_multiple_catchments
from .download import download_dwd as dwd_radolan_download
from .extraction import extract_time_series_from_radar

__version__ = "0.1.0"
__author__ = "Your Name"

__all__ = [
    "dwd_radolan_download",
    "compute_catchement_for_location",
    "compute_multiple_catchments",
    "extract_time_series_from_radar",
]
