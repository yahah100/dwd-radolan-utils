# Testing Guide

This document provides comprehensive information about the testing setup for `dwd-radolan-utils`.

## Quick Start

```bash
# Run all tests
make test

# Run with coverage
make test-cov

# Run specific test file
uv run pytest tests/test_extraction.py

# Run specific test
uv run pytest tests/test_extraction.py::TestReadRadarData::test_read_radar_data_basic
```

## Test Structure

The project uses `pytest` as the primary testing framework with the following structure:

```
tests/
├── conftest.py              # Shared fixtures and configuration
├── test_extraction.py       # Tests for extraction module (22 tests)
├── test_geo_utils.py        # Tests for geographic utilities (12 tests) 
├── test_download.py         # Tests for download functionality (33 tests)
├── test_catchment_area.py   # Tests for catchment area computation (13 tests)
└── test_integration.py      # Integration tests (6 tests)
```

**Total: 82 tests** covering all major functionality.

## Test Categories

### Unit Tests
- **Individual functions**: Test specific functions in isolation
- **Error handling**: Test edge cases and error conditions
- **Data validation**: Test input/output validation
- **Mocking**: Use mocks to isolate dependencies

### Integration Tests
- **Module interactions**: Test how different modules work together
- **File I/O**: Test reading/writing operations
- **Complete workflows**: Test end-to-end scenarios

## Fixtures (conftest.py)

The test suite includes comprehensive fixtures for common test data:

### Data Fixtures
- `sample_radar_data`: 3D radar data arrays
- `sample_time_data`: Datetime arrays for time series
- `sample_grid`: 2D numerical grids
- `sample_boolean_grid`: Boolean mask grids
- `sample_dataframe`: Pandas DataFrames
- `mock_dem_data`: Digital Elevation Model data

### Mock Fixtures
- `mock_pysheds_grid`: Mock pysheds Grid objects
- `mock_pysheds_raster`: Mock pysheds Raster objects
- `mock_radolan_response`: Mock HTTP responses
- `temp_directory`: Temporary directories for file operations

### File Fixtures
- `mock_radar_files`: Creates temporary radar data files
- `sample_wgs84_coordinates`: Sample coordinate data

## Running Tests

### Basic Testing
```bash
# Run all tests
uv run pytest

# Run with verbose output
uv run pytest -v

# Run specific module
uv run pytest tests/test_extraction.py

# Run specific test class
uv run pytest tests/test_extraction.py::TestReadRadarData

# Run specific test function
uv run pytest tests/test_extraction.py::TestReadRadarData::test_read_radar_data_basic
```

### Coverage Testing
```bash
# Run with coverage
uv run pytest --cov=dwd_radolan_utils

# Generate HTML coverage report
uv run pytest --cov=dwd_radolan_utils --cov-report=html

# Generate coverage report to terminal
uv run pytest --cov=dwd_radolan_utils --cov-report=term-missing
```

### Test Markers
```bash
# Run fast tests only (excluding slow integration tests)
uv run pytest -m "not slow"

# Run only integration tests
uv run pytest -m "integration"
```

## Makefile Commands

The project includes convenient Makefile commands:

```bash
make test          # Run all tests
make test-cov      # Run tests with coverage
make test-fast     # Run fast tests only
make lint          # Run linting
make format        # Format code
make type-check    # Run type checking
make security      # Run security checks
make all-checks    # Run all quality checks
```

## Test Configuration

### pytest Configuration (pyproject.toml)
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
python_functions = "test_*"
addopts = "-v"
```

### Coverage Configuration (tox.ini)
```ini
[coverage:run]
source = src
omit = 
    */tests/*
    */test_*
    */__pycache__/*

[coverage:report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise NotImplementedError
    if __name__ == .__main__.:
```

## Testing Best Practices

### 1. Test Organization
- Group related tests in classes
- Use descriptive test names
- Follow the AAA pattern (Arrange, Act, Assert)

### 2. Mocking
- Mock external dependencies (HTTP requests, file system)
- Use fixtures for common mock objects
- Isolate the code under test

### 3. Test Data
- Use fixtures for reusable test data
- Keep test data minimal but realistic
- Use temporary directories for file operations

### 4. Error Testing
- Test both success and failure cases
- Use `pytest.raises()` for exception testing
- Test edge cases and boundary conditions

## Example Test Structure

```python
class TestMyFunction:
    """Test cases for my_function."""
    
    def test_my_function_basic(self, sample_data):
        """Test basic functionality."""
        # Arrange
        input_data = sample_data
        expected = "expected_result"
        
        # Act
        result = my_function(input_data)
        
        # Assert
        assert result == expected
    
    def test_my_function_error_handling(self):
        """Test error handling."""
        with pytest.raises(ValueError, match="Invalid input"):
            my_function(invalid_input)
```

## Continuous Integration

Tests run automatically on:
- Pull requests
- Pushes to main/develop branches
- Release creation

### GitHub Actions Workflow
- **Test Matrix**: Python 3.13
- **Quality Checks**: Linting, formatting, type checking
- **Security**: Bandit and safety checks
- **Coverage**: Codecov integration

## Adding New Tests

### 1. Create Test File
```bash
# For new module
touch tests/test_new_module.py
```

### 2. Basic Test Template
```python
"""
Tests for the new_module.
"""
import pytest
from dwd_radolan_utils.new_module import new_function

class TestNewFunction:
    """Test cases for new_function."""
    
    def test_new_function_basic(self):
        """Test basic functionality."""
        result = new_function("input")
        assert result == "expected"
```

### 3. Run New Tests
```bash
# Test specific file
uv run pytest tests/test_new_module.py

# Test with coverage
uv run pytest tests/test_new_module.py --cov=dwd_radolan_utils.new_module
```

## Debugging Tests

### 1. Print Debugging
```python
def test_debug_example():
    result = my_function()
    print(f"Debug: result = {result}")  # Use -s flag to see output
    assert result == expected
```

```bash
uv run pytest tests/test_file.py::test_debug_example -s
```

### 2. PDB Debugging
```python
def test_pdb_example():
    import pdb; pdb.set_trace()  # Set breakpoint
    result = my_function()
    assert result == expected
```

### 3. Pytest Debugging
```bash
# Drop into PDB on first failure
uv run pytest --pdb

# Drop into PDB on first error
uv run pytest --pdb-trace
```

## Performance Testing

For performance-critical functions, consider adding benchmarks:

```python
import time

def test_performance_benchmark():
    """Test that function completes within time limit."""
    start_time = time.time()
    
    # Run your function
    result = expensive_function()
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Assert performance requirement
    assert execution_time < 1.0  # Should complete in under 1 second
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure the package is installed in development mode:
   ```bash
   uv pip install -e .
   ```

2. **Missing Dependencies**: Install development dependencies:
   ```bash
   uv sync --dev
   ```

3. **File Permission Errors**: Ensure test temporary directories are writable

4. **Memory Issues**: For large test datasets, use smaller sample data

### Test Isolation

Tests should be independent and not affect each other:
- Use temporary directories for file operations
- Clean up resources in teardown
- Don't rely on execution order

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Pytest Coverage](https://pytest-cov.readthedocs.io/)
- [Testing Best Practices](https://docs.python-guide.org/writing/tests/)
- [Mock Documentation](https://docs.python.org/3/library/unittest.mock.html) 