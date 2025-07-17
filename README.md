# GSX: Google Cloud (Storage) Exporter


## Description

Automatically export your data to Google Cloud Storage with ease. Supports
Parquet, Pickle, and JSON formats.

## Installation

```bash
uv add gsx
```

Or using pip:

```bash
pip install gsx
```

## Usage

### Initialization

Initialize the GSX object with your Google Cloud Project ID and Bucket name.

```python
from gsx import GSX, GSXCache
gcp_instance = GSX("your_project_id", "your_bucket_name")
```

### Decorator Usage

Simply use the `export` decorator on functions that you want to export data
from.

```python
@gcp_instance("data.parquet")
def get_data():
    return pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})

get_data()  # This will automatically export the DataFrame to "data.parquet" in the specified GCP bucket.
```

### Custom Time Parameter

You can specify a custom timestamp for the blob's `custom_time` field:

```python
from datetime import datetime

# Set a custom time for the blob
custom_time = datetime(2023, 1, 1, 12, 0, 0)

@gcp_instance("data.parquet", custom_time=custom_time)
def get_data():
    return pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})

get_data()  # This will export the DataFrame with the specified custom_time
```

### Caching Functions

GSX provides a separate `GSXCache` class for caching function results in Google Cloud Storage. The cache uses Parquet format by default but can be configured to use JSON or Pickle formats:

```python
from gsx import GSX, GSXCache
from datetime import timedelta

# Create GSX instance
gcp_instance = GSX("your_project_id", "your_bucket_name")

# Create cache instances with different formats
parquet_cache = GSXCache(gcp_instance, cache_format="parquet")  # Default
json_cache = GSXCache(gcp_instance, cache_format="json")
pickle_cache = GSXCache(gcp_instance, cache_format="pickle")

# Cache DataFrame results in Parquet format (default)
@parquet_cache.cache()
def get_dataframe():
    # This will be cached as .parquet file
    return pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})

# Cache JSON-serializable results
@json_cache.cache(ttl=timedelta(hours=1))
def get_json_data():
    # This will be cached as .json file
    return {"key": "value", "numbers": [1, 2, 3]}

# Cache any Python object with Pickle
@pickle_cache.cache()
def get_complex_data():
    # This will be cached as .pickle file
    return {"dataframe": pd.DataFrame({"a": [1, 2]}), "list": [1, 2, 3]}

# Cache with custom time metadata
@json_cache.cache(custom_time=datetime(2023, 1, 1, 12, 0, 0))
def get_timestamped_data():
    return {"timestamp": "2023-01-01", "data": "cached"}

# Usage examples
df_result = get_dataframe()      # First call - executes and caches
df_cached = get_dataframe()      # Second call - returns cached result

json_result = get_json_data()    # Cached with 1 hour TTL
complex_result = get_complex_data()  # Cached indefinitely

# Clear cache for specific function
parquet_cache.clear_cache("get_dataframe")

# Clear all cached results for this cache instance
json_cache.clear_cache()
```

### Cache Formats

- **Parquet** (default): Best for pandas DataFrames, efficient storage and fast read/write
- **JSON**: Best for dictionaries, lists, and simple data structures
- **Pickle**: Best for complex Python objects, supports any serializable data

### Cache Features

- **Configurable Format**: Choose between Parquet, JSON, or Pickle based on your data type
- **Automatic Key Generation**: Cache keys are automatically generated based on function name and arguments
- **TTL Support**: Set time-to-live for cached results to auto-expire old data
- **Custom Time Metadata**: Set custom timestamps on cached blobs
- **Cache Management**: Clear cache for specific functions or all cached results
- **Error Resilience**: Cache failures don't affect the original function execution

## Features

- **Auto Export**: Automatically export function outputs to GCP.
- **Multiple Formats**: Support for Parquet, Pickle, and JSON.
- **Function Caching**: Cache function results in Google Cloud Storage with TTL support.
- **Custom Time Metadata**: Set custom timestamps on blobs and cached results.

## Development

To set up the project for development:

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest tests/

# Run linting
uv run ruff check
```

<!--
## Documentation

For more details and API documentation, please refer to [docs/](./docs/).

-->
