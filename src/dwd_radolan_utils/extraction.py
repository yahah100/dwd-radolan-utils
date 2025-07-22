from pathlib import Path
from datetime import datetime
from typing import Literal
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from dwd_radolan_utils.geo_utils import cut_out_shapes


def read_radar_data(
    path: Path,
    min_max_dict: dict[str, int],
    start_date: datetime | None = None,
    end_date: datetime | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reads radar data from .npz files within a specified date range.
    Args:
        path (Path): The directory path containing the radar data files.
        min_max_dict (dict[str, int]): The dictionary containing the minimum and maximum values of the grid.
        start_date (datetime | None): The start date of the desired date range.
        end_date (datetime | None): The end date of the desired date range.
    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing two numpy arrays:
            - The first array contains the radar data.
            - The second array contains the corresponding time data.
    Raises:
        Exception: If no files are found for the specified date range.
    Notes:
        - The function expects the radar data files to be in .npz format.
        - The corresponding time data files should have the same name as the radar data files but with "_time.npz" suffix.
        - The function filters the radar data based on the provided date range.
        - The function cuts out the shape of the grid to the provided min_max_dict.
    """
    # get all files in the directory
    all_time_files = list(path.glob("**/*_time.npz"))
    logging.debug(f"Found {len(all_time_files)} files in {path}")

    # filter files by date
    files = []
    for file in all_time_files:
        stem = file.stem.split("_")[0]
        start = datetime.strptime(stem.split("-")[0], "%Y%m%d")
        end = datetime.strptime(stem.split("-")[1], "%Y%m%d")
        logging.debug(f"Start: {start}, End: {end}")
        if start_date is None or end_date is None:
            files.append(file)
        else:
            if start >= start_date and end <= end_date:
                files.append(file)

    if len(files) == 0:
        raise Exception(
            f"No files found for the date range {start_date} to {end_date}. You may need to download more data first."
        )

    # cut out shape, this can be done earlier to ensure faster loading speeds
    logging.debug(
        f"Found {len(files)} files for the date range {start_date} to {end_date}"
    )

    len_array = 0
    time_data = []
    for time_file in files:
        with np.load(time_file, allow_pickle=True) as data:
            time = data["arr_0"]
            time = time.astype(np.datetime64)
            time_data.append(time)
            len_array += time.shape[0]
    np_time_data = np.concatenate(time_data, axis=0)

    np_radar_data = np.zeros(
        (
            len_array,
            min_max_dict["max_x"] - min_max_dict["min_x"],
            min_max_dict["max_y"] - min_max_dict["min_y"],
        )
    )
    counter = 0
    for file in tqdm(files):
        file = str(file).replace("_time", "")
        with np.load(file, allow_pickle=True) as data:
            key = list(data.keys())[0]
            data = data[key]

            data = cut_out_shapes(
                data,
                min_dim_1=min_max_dict["min_x"],
                max_dim_1=min_max_dict["max_x"],
                min_dim_2=min_max_dict["min_y"],
                max_dim_2=min_max_dict["max_y"],
            )

            np_radar_data[counter : counter + data.shape[0], :, :] = data
            counter += data.shape[0]

    if start_date is not None and end_date is not None:
        start_dt64: np.datetime64 = np.datetime64(start_date)
        end_dt64: np.datetime64 = np.datetime64(end_date)

        bool_time_filter = (np_time_data >= start_dt64) & (np_time_data <= end_dt64)
        np_radar_data = np_radar_data[bool_time_filter]
        np_time_data = np_time_data[bool_time_filter]
    logging.info(f"Shape of radar data: {np_radar_data.shape}")
    return np_radar_data, np_time_data


def compute_arg_min_max_dict_nan(grid: np.ndarray) -> dict[str, int]:
    """
    Compute the boundaries of the non-NaN region in the grid.

    Args:
        grid (np.ndarray): Input grid of shape (900, 900) or (n, 900, 900)

    Returns:
        dict[str, int]: Dictionary containing min/max boundaries for cropping:
            - min_x: minimum row index with valid data
            - max_x: maximum row index with valid data
            - min_y: minimum column index with valid data
            - max_y: maximum column index with valid data
    """
    non_nan_mask = ~np.isnan(grid)

    if not np.any(non_nan_mask):
        raise ValueError("All values in grid are NaN")

    if grid.ndim == 2:
        rows_with_data = np.any(non_nan_mask, axis=1)
        cols_with_data = np.any(non_nan_mask, axis=0)
    elif grid.ndim == 3:
        rows_with_data = np.any(non_nan_mask, axis=(0, 2))
        cols_with_data = np.any(non_nan_mask, axis=(0, 1))
    else:
        raise ValueError(f"Grid must be 2D or 3D, got {grid.ndim}D")

    min_row = np.argmax(rows_with_data)
    max_row = len(rows_with_data) - np.argmax(rows_with_data[::-1])

    min_col = np.argmax(cols_with_data)
    max_col = len(cols_with_data) - np.argmax(cols_with_data[::-1])

    logging.info(
        f"Found non-NaN data boundaries: rows [{min_row}:{max_row}], cols [{min_col}:{max_col}]"
    )

    return {
        "min_x": int(min_row),
        "max_x": int(max_row),
        "min_y": int(min_col),
        "max_y": int(max_col),
    }


def compute_arg_min_max_dict_bool(grid: np.ndarray) -> dict[str, int]:
    """
    Compute the boundaries of the region in the grid which has values.
    If the grid has only bool values, the boundaries are computed based on the bool values.
    If has nan values, the boundaries are computed based on the nan values.

    Args:
        grid (np.ndarray): Input grid of shape (900, 900) or (n, 900, 900)

    Returns:
        dict[str, int]: Dictionary containing min/max boundaries for cropping:
            - min_x: minimum row index with valid data
            - max_x: maximum row index with valid data
            - min_y: minimum column index with valid data
            - max_y: maximum column index with valid data
    """
    true_mask = grid.astype(bool)

    if not np.any(true_mask):
        raise ValueError("All values in boolean grid are False")

    if grid.ndim == 2:
        rows_with_data = np.any(true_mask, axis=1)
        cols_with_data = np.any(true_mask, axis=0)
    elif grid.ndim == 3:
        rows_with_data = np.any(true_mask, axis=(0, 2))
        cols_with_data = np.any(true_mask, axis=(0, 1))
    else:
        raise ValueError(f"Grid must be 2D or 3D, got {grid.ndim}D")

    min_row = np.argmax(rows_with_data)
    max_row = len(rows_with_data) - np.argmax(rows_with_data[::-1])

    min_col = np.argmax(cols_with_data)
    max_col = len(cols_with_data) - np.argmax(cols_with_data[::-1])

    logging.info(
        f"Found True value boundaries in boolean grid: rows [{min_row}:{max_row}], cols [{min_col}:{max_col}]"
    )

    return {
        "min_x": int(min_row),
        "max_x": int(max_row),
        "min_y": int(min_col),
        "max_y": int(max_col),
    }


def compute_arg_min_max_dict(grid: np.ndarray) -> dict[str, int]:
    """
    Compute the boundaries of the region in the grid which has values.
    If the grid has only bool values, the boundaries are computed based on the bool values.
    If has nan values, the boundaries are computed based on the nan values.

    Args:
        grid (np.ndarray): Input grid of shape (900, 900) or (n, 900, 900)

    Returns:
        dict[str, int]: Dictionary containing min/max boundaries for cropping:
            - min_x: minimum row index with valid data
            - max_x: maximum row index with valid data
            - min_y: minimum column index with valid data
            - max_y: maximum column index with valid data
    """
    if np.any(np.isnan(grid)):
        return compute_arg_min_max_dict_nan(grid)
    elif grid.dtype == bool or (
        np.all(np.isin(grid, [0, 1]))
        and grid.dtype in [np.int32, np.int64, np.uint8, np.float32, np.float64]
    ):
        return compute_arg_min_max_dict_bool(grid)
    else:
        raise ValueError(
            f"Grid has unsupported data type or values. Dtype: {grid.dtype}, unique values: {np.unique(grid)[:10]}"
        )


def aggregate_ts(
    grid: np.ndarray,
    radar_grid: np.ndarray,
    method: Literal["mean", "sum", "max", "min"],
) -> np.ndarray:
    """
    Aggregate the time series of the radar grid using the method.

    Args:
        grid (np.ndarray): Input grid of shape (x,y) containing distance or other data
        radar_grid (np.ndarray): Input radar grid of shape (t, x, y) containing radar data where t is the time dimension
        method (Literal["mean", "sum", "max", "min"]): The method to aggregate the time series

    Returns:
        np.ndarray: The aggregated time series of shape (t,)
    """
    if grid.dtype == bool:
        grid = grid.astype(int)

    radar_grid[np.isnan(radar_grid)] = 0
    grid[np.isnan(grid)] = 0
    timeseries_array = grid * radar_grid
    logging.info(f"Timeseries array shape: {timeseries_array.shape}")

    if method == "mean":
        return np.mean(timeseries_array, axis=(1, 2))
    elif method == "sum":
        return np.sum(timeseries_array, axis=(1, 2))
    elif method == "max":
        return np.max(timeseries_array, axis=(1, 2))
    elif method == "min":
        return np.min(timeseries_array, axis=(1, 2))
    else:
        raise ValueError(f"Method {method} not supported")


def extract_time_series_from_radar(
    grid: np.ndarray,
    path: Path = Path("data/dwd"),
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    grid_aggregation_method: Literal["mean", "sum", "max", "min"] = "mean",
    save: bool = True,
    save_path: Path = Path("data/ts.csv"),
    save_column_names: list[str] | None = None,
    save_file_format: Literal["csv", "parquet"] = "csv",
    save_index_name: str = "timestamp",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract time series from radar data using the computed boundaries.

    Args:
        grid (np.ndarray): Input grid of shape (n, 900, 900) containing distance or other data
        path (Path): Path to the radar data
        start_date (datetime | None): Start date of the radar data
        end_date (datetime | None): End date of the radar data
        grid_aggregation_method (Literal["mean", "sum", "max", "min"]): Method to aggregate the time series
        save (bool): Whether to save the time series array
        save_path (Path | None): The path to save the time series array
        save_column_names (list[str] | None): The names of the columns to save
        save_file_format (Literal["csv", "parquet"]): The file format to save the data
        save_index_name (str): The name for the index column

    Returns:
        tuple[np.ndarray, np.ndarray]: Tuple containing the time series array and the timestamps
    """
    print(grid.shape)
    min_max_dict = compute_arg_min_max_dict(grid)

    logging.info(f"Computed boundaries for extraction: {min_max_dict}")

    radolan_grid, timestamps = read_radar_data(path, min_max_dict, start_date, end_date)

    ts_array = np.zeros((radolan_grid.shape[0], grid.shape[0]))

    logging.info(f"Extracting time series with target shape: {ts_array.shape}")

    # cut the grid to the same shape as the radolan grid
    grid = cut_out_shapes(
        grid,
        min_dim_1=min_max_dict["min_x"],
        max_dim_1=min_max_dict["max_x"],
        min_dim_2=min_max_dict["min_y"],
        max_dim_2=min_max_dict["max_y"],
    )

    for i, grid_i in enumerate(grid):
        ts = aggregate_ts(grid_i, radolan_grid, grid_aggregation_method)
        logging.info(f"Timeseries shape: {ts.shape}")
        ts_array[:, i] = ts

    if save:
        save_ts_array(
            ts_array,
            timestamps,
            save_path,
            save_column_names,
            save_file_format,
            save_index_name,
        )

    return ts_array, timestamps


def save_ts_array(
    ts_array: np.ndarray,
    timestamps: np.ndarray,
    path: Path,
    column_names: list[str] | None = None,
    file_format: Literal["csv", "parquet"] = "csv",
    index_name: str = "timestamp",
) -> None:
    """
    Save the time series array to a csv file.

    Args:
        ts_array (np.ndarray): The time series array to save (t,n) where t is the time dimension and n is the number of time series
        timestamps (np.ndarray): The timestamps (t,) of the time series
        path (Path): The path to save the time series array
        column_names (list[str] | None): The names of the columns to save
        file_format (Literal["csv", "parquet"]): The file format to save the data
        index_name (str): The name for the index column

    Returns:
        None
    """
    if column_names is None:
        column_names = [f"ts_{i}" for i in range(ts_array.shape[1])]
    logging.info(
        f"Saving time series to {path} with column names: {column_names} and shape: {ts_array.shape}"
    )
    df = pd.DataFrame(ts_array, index=timestamps, columns=column_names)
    df.index.name = index_name
    if file_format == "csv":
        df.to_csv(path)
    elif file_format == "parquet":
        df.to_parquet(path)
    else:
        raise ValueError(f"File format {file_format} not supported")
    logging.info(f"Saved time series to {path}")


def main():
    path = Path("data/dwd")
    grid = np.load(Path("dist_map_kluse.npy"))
    ts_array, timestamps = extract_time_series_from_radar(grid, path)
    print(ts_array.shape)
    print(timestamps.shape)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()
