"""
Module for downloading DWD RADOLAN data.

This module provides functionality to download RADOLAN radar precipitation data
from the German Weather Service (DWD).
"""

import bz2
import gzip
import logging as log
import shutil
import tarfile
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import requests
import wradlib as wrl
from bs4 import BeautifulSoup, Tag
from tqdm import tqdm

TypeRadarData = Literal[
    "recent",  # current year
    "historical",  # till current year
    "now",  # current day
]


def list_dwd_files_for_var(url: str):
    """
    Retrieves a list of DWD files for a specific variable from the given URL.

    Args:
        url (str): The URL to scrape for DWD files.

    Returns:
        list: A list of DWD file names for the specified variable.
    """
    log.info(f"Download files from {url}")
    page = requests.get(url)
    if page.status_code != 200:
        raise Exception(f"Failed to download files from {url}")
    soup = BeautifulSoup(page.content, "html.parser")
    links = soup.find_all("a")
    # Properly handle BeautifulSoup link types
    file_links = []
    for link in links:
        if isinstance(link, Tag):
            href = link.get("href")
            if href and isinstance(href, str):
                file_links.append(href)
    # remove parent directory link
    file_links = [link for link in file_links if link != "../"]
    log.debug(f"Found the following links: {file_links}")
    return file_links


def convert_time_str(time_str: str):
    """
    Converts a time string to a specific format.

    Args:
        time_str (str): The time string to be converted.

    Returns:
        str: The converted time string.
    """
    new_time: datetime

    # Check string length to determine format
    if len(time_str) == 14:
        # Long format: YYYYMMDDHHMMSS
        try:
            new_time = datetime.strptime(time_str, "%Y%m%d%H%M%S")
        except ValueError:
            raise ValueError(f"Unable to parse time string: {time_str}")
    elif len(time_str) == 10:
        # Short format: YYMMDDHHMM
        try:
            new_time = datetime.strptime(time_str, "%y%m%d%H%M")
        except ValueError:
            raise ValueError(f"Unable to parse time string: {time_str}")
    else:
        # Try both formats as fallback
        try:
            new_time = datetime.strptime(time_str, "%Y%m%d%H%M%S")
        except ValueError:
            try:
                new_time = datetime.strptime(time_str, "%y%m%d%H%M")
            except ValueError:
                raise ValueError(f"Unable to parse time string: {time_str}")

    time_str = new_time.strftime("%Y%m%d-%H%M")
    return time_str


def convert_dwd_filename(
    file_name: Path,
    separator: str = "-",
    historical_data: bool = False,
    is_recent: bool = False,
):
    """
    Converts a DWD filename to a specific format.

    Args:
        file_name (str): The DWD filename to be converted.

    Returns:
        str: The converted filename.
    """
    file_suffix = file_name.suffix
    file_stem = file_name.stem

    if historical_data:
        # historical data format RW202201.tar
        if "." in file_stem:
            file_stem, file_ending = file_stem.split(".", 1)
            time_part = file_stem[2:]
        else:
            # Handle case where there's no dot in the stem (e.g., compressed files)
            time_part = file_stem[2:] if len(file_stem) > 2 else file_stem
            file_ending = "tar"

        date_str = datetime.strptime(time_part, "%Y%m").strftime("%Y%m%d-%H%M")
        final_file_name = f"radolan_historical_{date_str}.{file_ending}{file_suffix}"
    else:
        file_name_split = file_stem.split(separator)

        # Handle different filename formats gracefully
        if len(file_name_split) < 3:
            # If filename doesn't have expected format, use the whole stem as time string
            time_str = file_stem
        else:
            # For DWD filenames like "raa01-rw-2401011200-dwd---bin", time is always at position 2
            time_str = file_name_split[2]

        try:
            final_file_name = convert_time_str(time_str)
            final_file_name = f"radolan_recent_{final_file_name}.bin"
        except ValueError:
            # If time parsing fails, use original filename with prefix
            final_file_name = f"radolan_recent_{file_stem}.bin"
    return final_file_name


def get_suffix(file_name: Path):
    suffix = file_name.suffix
    # check if the file is a tar.gz file
    sec_suffix = file_name.stem.split(".")[-1]
    if sec_suffix == "tar":
        suffix = ".tar.gz"
    return suffix


def dowload_file_and_save(
    url: str,
    file_name: Path,
    save_path: Path,
    historical_data: bool = False,
    is_recent: bool = False,
):
    """
    Downloads a file from the given URL, saves it to the specified path,
    and returns the path of the saved file.

    Args:
        url (str): The URL of the file to download.
        file_name (str): The name of the file to download.
        save_path (str): The path where the downloaded file will be saved.
        historical_data (bool): Whether the data is historical or not.
        is_recent (bool): Whether the data is recent or not.
    Returns:
        Path: The path of the saved file.
    """
    log.info(f"Download file {file_name} from {url} historical_data={historical_data}")
    suffix = get_suffix(file_name=file_name)
    if suffix not in [".tar.gz", ".gz", ".bz2"]:
        raise Exception(f"File {file_name} is not in a supported format (gz, bz2)")

    save_path.mkdir(parents=True, exist_ok=True)
    new_file_name = convert_dwd_filename(
        file_name, historical_data=historical_data, is_recent=is_recent
    )
    new_file = Path(save_path).joinpath(new_file_name)

    new_unpacked_file_name = new_file.stem
    new_unpacked_file = Path(save_path).joinpath(new_unpacked_file_name)
    if suffix == ".tar.gz":
        new_unpacked_file = new_unpacked_file.with_name(
            str(new_unpacked_file.name).replace(".tar", "")
        )

    if new_unpacked_file.exists():
        log.debug(f"File {new_unpacked_file} already exists")
        return new_unpacked_file

    log.debug(f"Download file {file_name} from {url}")
    response = requests.get(f"{url}{file_name}")
    if response.status_code != 200:
        raise Exception(f"Failed to download file {file_name} from {url}")
    with open(new_file, "wb") as f:
        f.write(response.content)

    if suffix == ".tar.gz":
        log.debug(f"Unpacking tar.gz file {new_file} to {new_unpacked_file}")
        with tarfile.open(new_file, "r:gz") as tar:
            tar.extractall(path=new_unpacked_file)
        new_file.unlink()
        log.debug(f"Unpacked file {new_unpacked_file} saved")
        return new_unpacked_file
    elif suffix == ".gz":
        open_func = gzip.open
    elif suffix == ".bz2":
        open_func = bz2.BZ2File

    with open_func(new_file, "rb") as f_in, open(new_unpacked_file, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

    log.debug(f"new_file file {new_file}")
    log.debug(f"Unpacked file {new_unpacked_file} saved")

    new_file.unlink()

    return new_unpacked_file


def download_radolan_data(
    radolan_url: str,
    file_list: list[str],
    save_path=Path(".tmp"),
    is_recent: bool = False,
):
    radolan_data = np.zeros((len(file_list), 900, 900))
    time_list = []
    for i, file in tqdm(
        enumerate(file_list), desc="Downloading radolan data", total=len(file_list)
    ):
        file = dowload_file_and_save(
            url=radolan_url,
            file_name=Path(file),
            save_path=save_path,
            historical_data=False,
            is_recent=is_recent,
        )
        data, metadata = wrl.io.read_radolan_composite(file)
        data[data == -9999] = np.nan
        time = metadata["datetime"]
        time_list.append(time)
        radolan_data[i] = data
    return radolan_data, time_list


def download_historical_radolan_data(
    radolan_url: str, file: str, save_path=Path(".tmp")
):
    parth_dir = dowload_file_and_save(
        url=radolan_url, file_name=Path(file), save_path=save_path, historical_data=True
    )
    file_list = list(parth_dir.glob("*bin"))
    # sort the files by name
    file_list = sorted(file_list, key=lambda x: x.name)
    radolan_data = np.zeros((len(file_list), 900, 900))
    time_list = []
    for i, file in tqdm(
        enumerate(file_list), desc="Extracting radolan data", total=len(file_list)
    ):
        data, metadata = wrl.io.read_radolan_composite(file)
        data[data == -9999] = np.nan
        time = metadata["datetime"]
        time_list.append(time)
        radolan_data[i] = data
    return radolan_data, time_list


def filter_by_month(date_list: list[str], year: int, month: int):
    for date in date_list:
        date_time = datetime.strptime(date.split("-")[2], "%y%m%d%H%M")
        if date_time.month == month and date_time.year == year:
            yield date


def filter_by_year_links(
    year_link_list: list[str], start_date: datetime, end_date: datetime
):
    # add all year from start_date till end_date to year_list
    year_list = range(start_date.year, end_date.year + 1)

    for year in year_link_list:
        year_int = int(year.replace("/", ""))
        if year_int in year_list:
            yield year


def filter_by_start_end(date_list: list[str], start_date: datetime, end_date: datetime):
    for date in date_list:
        date_time = datetime.strptime(date.split("-")[2], "%y%m%d%H%M")
        if start_date <= date_time <= end_date:
            yield date


def get_month_year_list(
    start_date: datetime, end_date: datetime
) -> list[tuple[int, int]]:
    date_list = []
    for year in range(start_date.year, end_date.year + 1):
        if year == start_date.year:
            start_month = start_date.month
        else:
            start_month = 1

        if year == end_date.year:
            end_month = end_date.month
        else:
            end_month = 13

        for month in range(start_month, end_month):
            date_list.append((year, month))

    return date_list


def save_to_npz_files(data: np.ndarray, time_list: list[datetime], save_path: Path):
    file_name = f"{time_list[0].year}{time_list[0].month:02d}{time_list[0].day:02d}-{time_list[-1].year}{time_list[-1].month:02d}{time_list[-1].day:02d}"
    save_path.mkdir(parents=True, exist_ok=True)
    time_array = np.array(object=time_list, dtype="datetime64")
    data_save_path = save_path.joinpath(file_name).with_suffix(".npz")
    np.savez_compressed(data_save_path, data=data)
    time_path = save_path.joinpath(file_name + "_time").with_suffix(".npz")
    np.savez_compressed(time_path, time_array)


def download_dwd(
    type_radolan: TypeRadarData,
    start: datetime | None,
    end: datetime | None,
    save_path: Path = Path("data/dwd/"),
):
    if type_radolan == "now":
        radolan_url = "https://opendata.dwd.de/weather/radar/radolan/rw/"
        list_files = list_dwd_files_for_var(radolan_url)
        list_files = [file for file in list_files if file.endswith(".bz2")]
        log.info("Ignoring start and end date for type 'now'")
        raise Exception("Not implemented yet")

    elif type_radolan == "recent" and start is not None and end is not None:
        # if end - start > 1 month, download one month at a time
        if (end - start).days > 31:
            log.info(
                f"Downloading recent data from {start} to {end} to {save_path} one month at a time"
            )
            year_month_list = get_month_year_list(start_date=start, end_date=end)
            for year, month in year_month_list:
                download_one_month(
                    year=year, month=month, type_radolan="recent", save_path=save_path
                )
        else:
            log.info(f"Downloading recent data from {start} to {end} to {save_path}")
            radolan_url = "https://opendata.dwd.de/climate_environment/CDC/grids_germany/hourly/radolan/recent/bin/"
            list_files = list_dwd_files_for_var(radolan_url)
            list_files = [file for file in list_files if file.endswith(".gz")]
            list_files = list(
                filter_by_start_end(
                    date_list=list_files, start_date=start, end_date=end
                )
            )
            if len(list_files) == 0:
                raise Exception(f"No files found for start {start} and end {end}")
            radolan_data, time_list = download_radolan_data(
                radolan_url=radolan_url,
                file_list=list_files,
                save_path=Path(".tmp"),
                is_recent=True,
            )
            save_to_npz_files(radolan_data, time_list, save_path=save_path)

    elif type_radolan == "historical" and start is not None and end is not None:
        log.info(f"Downloading historical data from {start} to {end} to {save_path}")
        log.warning("Downloading historical data always downloads the whole month")
        year_month_list = get_month_year_list(start_date=start, end_date=end)

        for year, month in year_month_list:
            download_one_month(
                year=year, month=month, type_radolan="historical", save_path=save_path
            )
    else:
        raise Exception(
            f"Invalid type_radolan {type_radolan} or missing start and end date"
        )


def add_one_month(date: datetime) -> datetime:
    """Add one month to a datetime object, handling month/year transitions correctly."""
    if date.month == 12:
        return datetime(date.year + 1, 1, date.day)
    next_month = date.replace(month=date.month + 1)
    return next_month


def download_one_month(
    year: int,
    month: int,
    type_radolan: TypeRadarData,
    save_path: Path = Path("data/dwd/"),
):
    assert year is not None and month is not None, "Year and month must be provided"
    start = datetime(year=year, month=month, day=1)
    end = add_one_month(start)
    if type_radolan == "now":
        raise Exception(
            "Can't download current month with type 'now' now only includes one day"
        )

    elif type_radolan == "recent":
        radolan_url = "https://opendata.dwd.de/climate_environment/CDC/grids_germany/hourly/radolan/recent/bin/"
        list_files = list_dwd_files_for_var(radolan_url)
        list_files = [file for file in list_files if file.endswith(".gz")]

        list_files = list(
            filter_by_start_end(date_list=list_files, start_date=start, end_date=end)
        )
        radolan_data, time_list = download_radolan_data(
            radolan_url=radolan_url,
            file_list=list_files,
            save_path=Path(".tmp"),
            is_recent=True,
        )
        save_to_npz_files(radolan_data, time_list, save_path=save_path)

    elif type_radolan == "historical":
        radolan_url = f"https://opendata.dwd.de/climate_environment/CDC/grids_germany/hourly/radolan/historical/bin/{year}/"
        list_files = list_dwd_files_for_var(radolan_url)

        list_files = [
            file for file in list_files if file.endswith(f"{year}{month:02}.tar.gz")
        ]
        if len(list_files) == 0:
            log.warning(f"No files found for year {year} and month {month}")
            return  # Exit early if no files found
        elif len(list_files) > 1:
            log.warning(f"Multiple files found for year {year} and month {month}")
            return
        file = list_files[0]
        radolan_data, time_list = download_historical_radolan_data(
            radolan_url=radolan_url, file=file, save_path=Path(".tmp")
        )
        save_to_npz_files(radolan_data, time_list, save_path=save_path)


def main():
    type_radolan: TypeRadarData = "recent"
    start_date = datetime(year=2025, month=5, day=1)
    end_date = datetime(year=2025, month=5, day=3)

    download_dwd(type_radolan=type_radolan, start=start_date, end=end_date)


if __name__ == "__main__":
    log.basicConfig(
        level=log.INFO,  # Set the lowest log level you want to capture
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            # log.FileHandler("app.log"),  # Log to a file
            log.StreamHandler()  # Log to console
        ],
    )
    main()
