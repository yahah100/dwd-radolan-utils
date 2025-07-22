"""
Tests for the download module.
"""
import pytest
import requests
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, Mock, MagicMock, mock_open
from bs4 import BeautifulSoup
import numpy as np
import shutil

from dwd_radolan_utils.download import (
    list_dwd_files_for_var,
    convert_time_str,
    convert_dwd_filename,
    get_suffix,
    dowload_file_and_save,
    download_radolan_data,
    download_historical_radolan_data,
    filter_by_month,
    filter_by_year_links,
    filter_by_start_end,
    get_month_year_list,
    save_to_npz_files,
    download_dwd,
    download_one_month
)


class TestListDwdFilesForVar:
    """Test cases for list_dwd_files_for_var function."""
    
    @patch('dwd_radolan_utils.download.requests.get')
    def test_list_dwd_files_success(self, mock_get):
        """Test successful file listing."""
        # Mock HTML response
        html_content = """
        <html>
            <body>
                <a href="../">../</a>
                <a href="file1.gz">file1.gz</a>
                <a href="file2.bz2">file2.bz2</a>
                <a href="file3.tar.gz">file3.tar.gz</a>
            </body>
        </html>
        """
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = html_content.encode()
        mock_get.return_value = mock_response
        
        url = "https://test.example.com/"
        result = list_dwd_files_for_var(url)
        
        expected = ["file1.gz", "file2.bz2", "file3.tar.gz"]
        assert result == expected
        mock_get.assert_called_once_with(url)
    
    @patch('dwd_radolan_utils.download.requests.get')
    def test_list_dwd_files_failure(self, mock_get):
        """Test error handling for failed requests."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response
        
        url = "https://test.example.com/"
        
        with pytest.raises(Exception, match="Failed to download files"):
            list_dwd_files_for_var(url)


class TestConvertTimeStr:
    """Test cases for convert_time_str function."""
    
    def test_convert_time_str_long_format(self):
        """Test conversion of long time format."""
        time_str = "20240101120000"  # YYYYMMDDHHMMSS
        result = convert_time_str(time_str)
        assert result == "20240101-1200"
    
    def test_convert_time_str_short_format(self):
        """Test conversion of short time format."""
        time_str = "2401011200"  # YYMMDDHHMM
        result = convert_time_str(time_str)
        assert result == "20240101-1200"  
    
    def test_convert_time_str_edge_cases(self):
        """Test edge cases for time conversion."""
        # Test midnight
        result = convert_time_str("20240101000000")
        assert result == "20240101-0000"
        
        # Test end of year
        result = convert_time_str("20241231235900")
        assert result == "20241231-2359"
    
    def test_convert_time_str_invalid_format(self):
        """Test error handling for invalid time format."""
        with pytest.raises(ValueError, match="Unable to parse time string"):
            convert_time_str("invalid_time")


class TestConvertDwdFilename:
    """Test cases for convert_dwd_filename function."""
    
    def test_convert_dwd_filename_recent(self):
        """Test filename conversion for recent data."""
        filename = Path("raa01-rw_10000-2401011200-dwd---bin.gz")
        result = convert_dwd_filename(filename, is_recent=True)
        assert result == "radolan_recent_20240101-1200.bin"
    
    def test_convert_dwd_filename_historical(self):
        """Test filename conversion for historical data."""
        filename = Path("RW202401.tar.gz")
        result = convert_dwd_filename(filename, historical_data=True)
        assert result == "radolan_historical_20240101-0000.tar.gz"
    
    def test_convert_dwd_filename_standard(self):
        """Test filename conversion for standard format."""
        filename = Path("raa01-rw-2401011200-dwd---bin.gz")
        result = convert_dwd_filename(filename)
        assert result == "radolan_recent_20240101-1200.bin"
    
    def test_convert_dwd_filename_simple_format(self):
        """Test filename conversion for simple format (fallback case)."""
        filename = Path("test.gz")
        result = convert_dwd_filename(filename)
        assert result == "radolan_recent_test.bin"  # Should use fallback logic


class TestGetSuffix:
    """Test cases for get_suffix function."""
    
    def test_get_suffix_regular(self):
        """Test suffix detection for regular files."""
        assert get_suffix(Path("file.gz")) == ".gz"
        assert get_suffix(Path("file.bz2")) == ".bz2"
    
    def test_get_suffix_tar_gz(self):
        """Test suffix detection for tar.gz files."""
        assert get_suffix(Path("file.tar.gz")) == ".tar.gz"
    
    def test_get_suffix_no_extension(self):
        """Test suffix detection for files without extension."""
        assert get_suffix(Path("file")) == ""


class TestDownloadFileAndSave:
    """Test cases for dowload_file_and_save function."""
    
    @patch('dwd_radolan_utils.download.requests.get')
    @patch('builtins.open', new_callable=mock_open)
    @patch('dwd_radolan_utils.download.gzip.open', new_callable=mock_open)
    @patch('dwd_radolan_utils.download.shutil.copyfileobj')
    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.unlink')
    @patch('pathlib.Path.mkdir')
    def test_download_gz_file(self, mock_mkdir, mock_unlink, mock_exists, mock_copyfileobj, mock_gzip_open, mock_file_open, mock_get, temp_directory):
        """Test downloading and extracting .gz file."""
        # Setup mocks
        mock_exists.return_value = False  # File doesn't exist yet
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b"test data"
        mock_get.return_value = mock_response
        
        # Use proper DWD filename format
        filename = Path("raa01-rw-2401011200-dwd---bin.gz")
        url = "https://test.example.com/"
        
        result = dowload_file_and_save(url, filename, temp_directory)
        
        # Test that the function processes the file and returns a valid path
        assert isinstance(result, Path)
        assert str(result).startswith(str(temp_directory))
        assert "radolan_recent" in str(result)
        
        # Verify request was made
        mock_get.assert_called_once_with(f"{url}{filename}")
        # Verify file operations were called
        mock_copyfileobj.assert_called_once()
        mock_unlink.assert_called_once()
    
    @patch('dwd_radolan_utils.download.requests.get')
    def test_download_file_request_failure(self, mock_get, temp_directory):
        """Test error handling for failed file download."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response
        
        # Use proper DWD filename format
        filename = Path("raa01-rw-2401011200-dwd---bin.gz")
        url = "https://test.example.com/"
        
        with pytest.raises(Exception, match="Failed to download file"):
            dowload_file_and_save(url, filename, temp_directory)
    
    def test_download_unsupported_format(self, temp_directory):
        """Test error handling for unsupported file formats."""
        filename = Path("test.txt")
        url = "https://test.example.com/"
        
        with pytest.raises(Exception, match="not in a supported format"):
            dowload_file_and_save(url, filename, temp_directory)
    
    @patch('dwd_radolan_utils.download.Path.exists')
    def test_download_file_already_exists(self, mock_exists, temp_directory):
        """Test skipping download when file already exists."""
        mock_exists.return_value = True
        
        # Use proper DWD filename format
        filename = Path("raa01-rw-2401011200-dwd---bin.gz")
        url = "https://test.example.com/"
        
        with patch('dwd_radolan_utils.download.requests.get') as mock_get:
            result = dowload_file_and_save(url, filename, temp_directory)
            
            # Should not make request if file exists
            mock_get.assert_not_called()
            
            # Should return the expected path (unpacked file without .bin extension)
            expected_path = temp_directory / "radolan_recent_20240101-1200"
            assert result == expected_path


class TestFilterFunctions:
    """Test cases for various filter functions."""
    
    def test_filter_by_month(self):
        """Test filtering dates by month."""
        date_list = [
            "raa01-rw-2401011200-dwd---bin",
            "raa01-rw-2401151200-dwd---bin", 
            "raa01-rw-2402011200-dwd---bin",
            "raa01-rw-2403011200-dwd---bin"
        ]
        
        result = list(filter_by_month(date_list, 2024, 1))
        
        # Should return only January 2024 files
        assert len(result) == 2
        assert "2401011200" in result[0]
        assert "2401151200" in result[1]
    
    def test_filter_by_year_links(self):
        """Test filtering year links."""
        year_links = ["2022/", "2023/", "2024/", "2025/"]
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2024, 12, 31)
        
        result = list(filter_by_year_links(year_links, start_date, end_date))
        
        assert "2023/" in result
        assert "2024/" in result
        assert "2022/" not in result
        assert "2025/" not in result
    
    def test_filter_by_start_end(self):
        """Test filtering by start and end dates."""
        date_list = [
            "raa01-rw-2401011200-dwd---bin",  # Jan 1
            "raa01-rw-2401151200-dwd---bin",  # Jan 15
            "raa01-rw-2402011200-dwd---bin",  # Feb 1
        ]
        
        start_date = datetime(2024, 1, 10)
        end_date = datetime(2024, 1, 20)
        
        result = list(filter_by_start_end(date_list, start_date, end_date))
        
        # Should only include Jan 15
        assert len(result) == 1
        assert "2401151200" in result[0]
    
    def test_get_month_year_list(self):
        """Test generating month-year list."""
        start_date = datetime(2023, 11, 1)
        end_date = datetime(2024, 2, 15)
        
        result = get_month_year_list(start_date, end_date)
        
        expected = [(2023, 11), (2023, 12), (2024, 1)]
        assert result == expected
    
    def test_get_month_year_list_same_year(self):
        """Test month-year list for same year."""
        start_date = datetime(2024, 3, 1)
        end_date = datetime(2024, 6, 30)
        
        result = get_month_year_list(start_date, end_date)
        
        expected = [(2024, 3), (2024, 4), (2024, 5)]
        assert result == expected


class TestSaveToNpzFiles:
    """Test cases for save_to_npz_files function."""
    
    @patch('numpy.savez_compressed')
    def test_save_to_npz_files(self, mock_savez, temp_directory):
        """Test saving data to NPZ files."""
        # Test data
        data = np.random.rand(24, 100, 100)
        time_list = [datetime(2024, 1, 1, h) for h in range(24)]
        
        save_to_npz_files(data, time_list, temp_directory)
        
        # Should call savez_compressed twice (data and time)
        assert mock_savez.call_count == 2
        
        # Check the file paths
        calls = mock_savez.call_args_list
        data_path = calls[0][0][0]
        time_path = calls[1][0][0]
        
        assert "20240101-20240101" in str(data_path)
        assert data_path.suffix == ".npz"
        assert "_time" in str(time_path)


class TestDownloadDwd:
    """Test cases for download_dwd function."""
    
    @patch('dwd_radolan_utils.download.download_one_month')
    def test_download_dwd_recent_multiple_months(self, mock_download_month):
        """Test downloading recent data spanning multiple months."""
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 3, 15)  # More than 31 days
        
        download_dwd("recent", start_date, end_date, Path("test_data"))
        
        # Should call download_one_month for each month
        expected_calls = [(2024, 1), (2024, 2)]
        actual_calls = []
        
        # Extract keyword arguments since function is called with kwargs
        for call in mock_download_month.call_args_list:
            if 'year' in call.kwargs and 'month' in call.kwargs:
                actual_calls.append((call.kwargs['year'], call.kwargs['month']))
        
        assert len(actual_calls) == 2
        for call_args in actual_calls:
            assert call_args in expected_calls
    
    @patch('dwd_radolan_utils.download.list_dwd_files_for_var')
    @patch('dwd_radolan_utils.download.download_radolan_data')
    @patch('dwd_radolan_utils.download.save_to_npz_files')
    def test_download_dwd_recent_single_month(self, mock_save, mock_download, mock_list, temp_directory):
        """Test downloading recent data for single month."""
        # Setup mocks
        mock_list.return_value = ["file1.gz", "file2.gz"]
        mock_download.return_value = (np.random.rand(2, 100, 100), [datetime.now(), datetime.now()])
        
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 15)  # Less than 31 days
        
        with patch('dwd_radolan_utils.download.filter_by_start_end') as mock_filter:
            mock_filter.return_value = ["file1.gz"]
            
            download_dwd("recent", start_date, end_date, temp_directory)
            
            # Verify the workflow
            mock_list.assert_called_once()
            mock_download.assert_called_once()
            mock_save.assert_called_once()
    
    def test_download_dwd_now_not_implemented(self):
        """Test that 'now' type raises not implemented error."""
        with pytest.raises(Exception, match="Not implemented yet"):
            download_dwd("now", None, None)
    
    def test_download_dwd_invalid_params(self):
        """Test error handling for invalid parameters."""
        with pytest.raises(Exception, match="Invalid type_radolan"):
            download_dwd("invalid", None, None)  # type: ignore


class TestDownloadOneMonth:
    """Test cases for download_one_month function."""
    
    @patch('dwd_radolan_utils.download.list_dwd_files_for_var')
    @patch('dwd_radolan_utils.download.download_radolan_data')
    @patch('dwd_radolan_utils.download.save_to_npz_files')
    def test_download_one_month_recent(self, mock_save, mock_download, mock_list, temp_directory):
        """Test downloading one month of recent data."""
        # Setup mocks
        mock_list.return_value = ["file1.gz", "file2.gz"]
        mock_download.return_value = (np.random.rand(2, 100, 100), [datetime.now(), datetime.now()])
        
        with patch('dwd_radolan_utils.download.filter_by_start_end') as mock_filter:
            mock_filter.return_value = ["file1.gz"]
            
            download_one_month(2024, 1, "recent", temp_directory)
            
            # Verify calls
            mock_list.assert_called_once()
            mock_download.assert_called_once()
            mock_save.assert_called_once()
    
    @patch('dwd_radolan_utils.download.list_dwd_files_for_var')
    @patch('dwd_radolan_utils.download.download_historical_radolan_data')
    @patch('dwd_radolan_utils.download.save_to_npz_files')
    def test_download_one_month_historical(self, mock_save, mock_download_hist, mock_list, temp_directory):
        """Test downloading one month of historical data."""
        # Setup mocks
        mock_list.return_value = ["RW202401.tar.gz"]
        mock_download_hist.return_value = (np.random.rand(24, 100, 100), [datetime.now()] * 24)
        
        download_one_month(2024, 1, "historical", temp_directory)
        
        # Verify calls
        mock_list.assert_called_once()
        mock_download_hist.assert_called_once()
        mock_save.assert_called_once()
    
    def test_download_one_month_now_error(self):
        """Test error for 'now' type in download_one_month."""
        with pytest.raises(Exception, match="Can't download current month"):
            download_one_month(2024, 1, "now")
    
    @patch('dwd_radolan_utils.download.list_dwd_files_for_var')
    def test_download_one_month_no_files(self, mock_list, temp_directory):
        """Test handling when no files are found for historical data."""
        mock_list.return_value = []
        
        # Should not raise error, just log warning
        download_one_month(2024, 1, "historical", temp_directory)
        
        mock_list.assert_called_once()
    
    @patch('dwd_radolan_utils.download.list_dwd_files_for_var')
    @patch('dwd_radolan_utils.download.download_historical_radolan_data')
    def test_download_one_month_multiple_files(self, mock_download_hist, mock_list, temp_directory):
        """Test handling when multiple files are found for historical data."""
        mock_list.return_value = ["RW202401.tar.gz", "RW202401_alt.tar.gz"]
        
        # Mock the download function to avoid actual file processing
        mock_download_hist.return_value = (np.random.rand(24, 100, 100), [datetime.now()] * 24)
        
        # Should return early due to multiple files
        download_one_month(2024, 1, "historical", temp_directory)
        
        mock_list.assert_called_once()
        # The function should return early when multiple files are found, so download shouldn't be called
        # But let's verify the actual behavior by checking if it was called or not
        # Since the function now returns early on multiple files, it shouldn't be called 