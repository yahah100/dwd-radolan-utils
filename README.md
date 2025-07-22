# DWD RADOLAN Utils

[![CI/CD Pipeline](https://github.com/yourusername/dwd-radolan-utils/workflows/CI/CD%20Pipeline/badge.svg)](https://github.com/yourusername/dwd-radolan-utils/actions)
[![codecov](https://codecov.io/gh/yourusername/dwd-radolan-utils/branch/main/graph/badge.svg)](https://codecov.io/gh/yourusername/dwd-radolan-utils)
[![PyPI version](https://badge.fury.io/py/dwd-radolan-utils.svg)](https://badge.fury.io/py/dwd-radolan-utils)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/release/python-3130/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Python utilities for downloading and processing DWD RADOLAN radar data. This package provides tools for working with German Weather Service (DWD) radar precipitation data, including downloading, processing, and extracting time series from radar grids.

## Features

- **Download RADOLAN data** from DWD servers (recent, historical, and current data)
- **Extract time series** from radar data using custom grids or catchment areas
- **Compute catchment areas** from Digital Elevation Models (DEM)
- **Geographic utilities** for coordinate conversion and grid manipulation
- **Comprehensive testing** with pytest and continuous integration
- **Type hints** and modern Python practices

## Installation

### From PyPI (recommended)

```bash
pip install dwd-radolan-utils
```

### From source

```bash
git clone https://github.com/yourusername/dwd-radolan-utils.git
cd dwd-radolan-utils
pip install -e .
```

### Development installation

```bash
git clone https://github.com/yourusername/dwd-radolan-utils.git
cd dwd-radolan-utils
uv sync --dev
```

## Quick Start

### Download RADOLAN data

```python
from datetime import datetime
from pathlib import Path
from dwd_radolan_utils.download import download_dwd

# Download recent data
start_date = datetime(2024, 1, 1)
end_date = datetime(2024, 1, 31)

download_dwd(
    type_radolan="recent",
    start=start_date,
    end=end_date,
    save_path=Path("data/dwd/")
)
```

### Extract time series from radar data

```python
import numpy as np
from dwd_radolan_utils.extraction import extract_time_series_from_radar

# Create a boolean mask for your area of interest
grid = np.zeros((900, 900), dtype=bool)
grid[400:500, 400:500] = True  # Select a 100x100 pixel region

# Extract time series
ts_array, timestamps = extract_time_series_from_radar(
    grid=grid,
    path=Path("data/dwd/"),
    start_date=start_date,
    end_date=end_date,
    grid_aggregation_method="mean"
)
```

### Compute catchment areas

```python
from dwd_radolan_utils.catchment_area import compute_catchement_for_location

# Coordinates for a discharge station (lon, lat)
coordinates = (7.158556, 51.255604)  # Kluse station

# Compute catchment area and distance grid
dist_grid, grid = compute_catchement_for_location(
    coordinates=coordinates,
    downsample_factor=50
)
```

## Development

This project uses modern Python development tools and practices.

### Setup

```bash
# Install uv (modern Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone https://github.com/yourusername/dwd-radolan-utils.git
cd dwd-radolan-utils
uv sync --dev

# Install pre-commit hooks
make install-hooks
```

### Testing

The project has comprehensive test coverage with pytest:

```bash
# Run all tests
make test

# Run with coverage
make test-cov

# Run fast tests only (excluding slow integration tests)
make test-fast

# Run specific test file
uv run pytest tests/test_extraction.py

# Run with verbose output
uv run pytest -v
```

### Code Quality

```bash
# Run all quality checks
make all-checks

# Individual checks
make lint          # Linting with ruff
make format        # Code formatting
make type-check    # Type checking with mypy
make security      # Security checks with bandit
```

### Available Commands

See all available development commands:

```bash
make help
```

## Testing Framework

### Test Structure

```
tests/
├── conftest.py              # Shared fixtures and configuration
├── test_extraction.py       # Tests for extraction module
├── test_geo_utils.py        # Tests for geographic utilities
├── test_download.py         # Tests for download functionality
├── test_catchment_area.py   # Tests for catchment area computation
└── test_integration.py      # Integration tests
```

### Test Categories

- **Unit tests**: Test individual functions and classes
- **Integration tests**: Test module interactions
- **Fixtures**: Reusable test data and mocks in `conftest.py`

### Running Specific Tests

```bash
# Test specific module
uv run pytest tests/test_extraction.py

# Test specific function
uv run pytest tests/test_extraction.py::TestReadRadarData::test_read_radar_data_basic

# Test with markers
uv run pytest -m "not slow"  # Exclude slow tests
```

## Continuous Integration

The project uses GitHub Actions for CI/CD:

- **Testing**: Automated testing on Python 3.13
- **Code Quality**: Linting, formatting, and type checking
- **Security**: Vulnerability scanning with bandit and safety
- **Coverage**: Code coverage reporting with codecov
- **Publishing**: Automated PyPI publishing on releases

### Workflows

- `.github/workflows/ci.yml`: Main CI/CD pipeline
- `.github/workflows/pre-commit.yml`: Pre-commit checks

## Project Structure

```
dwd-radolan-utils/
├── src/dwd_radolan_utils/   # Main package
│   ├── __init__.py
│   ├── download.py          # Data downloading functionality
│   ├── extraction.py        # Time series extraction
│   ├── geo_utils.py         # Geographic utilities
│   ├── catchment_area.py    # Catchment area computation
│   └── pysheds_helper/      # Helper functions for pysheds
├── tests/                   # Test suite
├── .github/workflows/       # GitHub Actions
├── pyproject.toml          # Project configuration
├── Makefile                # Development commands
├── tox.ini                 # Testing environments
└── .pre-commit-config.yaml # Pre-commit configuration
```

## Dependencies

### Core Dependencies

- `numpy`: Numerical computing
- `pandas`: Data manipulation
- `xarray`: N-dimensional arrays
- `wradlib`: Weather radar utilities
- `pysheds`: Watershed delineation
- `pyproj`: Cartographic projections
- `requests`: HTTP library
- `tqdm`: Progress bars

### Development Dependencies

- `pytest`: Testing framework
- `pytest-cov`: Coverage reporting
- `ruff`: Linting and formatting
- `mypy`: Type checking
- `pre-commit`: Git hooks
- `bandit`: Security linting

## Configuration Files

- `pyproject.toml`: Main project configuration (dependencies, build system, tools)
- `tox.ini`: Multi-environment testing
- `.pre-commit-config.yaml`: Pre-commit hooks
- `Makefile`: Development commands

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes and add tests
4. Run quality checks: `make all-checks`
5. Commit with conventional commits format
6. Push and create a pull request

### Commit Message Format

Use conventional commits:

```
feat: add new feature
fix: fix bug
docs: update documentation
test: add tests
refactor: refactor code
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- German Weather Service (DWD) for providing RADOLAN data
- The wradlib team for weather radar utilities
- The pysheds team for watershed analysis tools

## Citation

If you use this package in research, please cite:

```bibtex
@software{dwd_radolan_utils,
  title={DWD RADOLAN Utils: Python utilities for German weather radar data},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/dwd-radolan-utils}
}
```
