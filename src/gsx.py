"""
GSX: Google Cloud Platform Exporter
=====================================

This package provides a way to automatically export data to Google Cloud Storage.
Supports exporting data in Parquet, Pickle, and JSON formats.

Attributes
----------
__version__ : str
    The current version of the package.
"""

import hashlib
import json
import pickle
import warnings
from datetime import datetime, timedelta
from functools import wraps
from io import BytesIO
from typing import Any, Callable

from google.cloud import storage
from google.cloud.exceptions import NotFound


class GSX:
    """
    Google Cloud Platform Exporter (GSX).

    Initialize the GSX object with a Google Cloud Project ID and Bucket name.

    Parameters
    ----------
    project_id : str
        The Google Cloud project ID.
    bucket_name : str
        The Google Cloud bucket name.

    Attributes
    ----------
    project_id : str
    bucket_name : str
    storage_client : google.cloud.storage.Client
    bucket : google.cloud.storage.Bucket
    """

    def __init__(self, project_id: str, bucket_name: str):
        self.project_id = project_id
        self.bucket_name = bucket_name
        self.storage_client = self.build_client()
        self.bucket = self.storage_client.bucket(self.bucket_name)

    def build_client(self):
        """
        Build and return a Google Cloud Storage client.

        Returns
        -------
        google.cloud.storage.Client
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            client = storage.Client(project=self.project_id)
        return client

    def _export_parquet(
        self, result, blob: storage.Blob, custom_time: datetime | None = None
    ):
        """
        Export a Pandas DataFrame to a blob in Parquet format.

        Parameters
        ----------
        result : pd.DataFrame
        blob : google.cloud.storage.Blob
        custom_time : datetime, optional
            Custom timestamp for the blob's custom_time field.
        """
        if custom_time is not None:
            blob.custom_time = custom_time

        with BytesIO() as buffer:
            result.to_parquet(buffer)
            blob.upload_from_string(
                buffer.getvalue(),
                content_type="application/octet-stream",
            )

    def _export_pickle(
        self, result, blob: storage.Blob, custom_time: datetime | None = None
    ):
        """
        Export a Python object to a blob in Pickle format.

        Parameters
        ----------
        result : Any
        blob : google.cloud.storage.Blob
        custom_time : datetime, optional
            Custom timestamp for the blob's custom_time field.
        """
        if custom_time is not None:
            blob.custom_time = custom_time

        with BytesIO() as buffer:
            result.to_pickle(buffer)
            blob.upload_from_string(
                buffer.getvalue(),
                content_type="application/octet-stream",
            )

    def _export_json(
        self, result, blob: storage.Blob, custom_time: datetime | None = None
    ):
        """
        Export a Python object to a blob in JSON format.

        Parameters
        ----------
        result : Any
        blob : google.cloud.storage.Blob
        custom_time : datetime, optional
            Custom timestamp for the blob's custom_time field.
        """
        if custom_time is not None:
            blob.custom_time = custom_time

        blob.upload_from_string(
            json.dumps(result),
            content_type="application/json",
        )

    def _export_generic(
        self,
        result: Any,
        blob: storage.Blob,
        func: Callable | None = None,
        custom_time: datetime | None = None,
        **blob_kws,
    ):
        if custom_time is not None:
            blob.custom_time = custom_time

        if func is not None:
            result = func(result)

        blob.upload_from_string(
            result,
            **blob_kws,
        )

    def export(self, path: str, custom_time: datetime | None = None) -> Callable:
        """
        Decorator to export the result of a function to a given path.

        Parameters
        ----------
        path : str
            The path (blob name) in the bucket to export data.
        custom_time : datetime, optional
            Custom timestamp for the blob's custom_time field.

        Returns
        -------
        Callable
            A decorated function.
        """

        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                result = func(*args, **kwargs)
                blob_name = path
                blob = self.bucket.blob(blob_name)

                if path.endswith(".parquet"):
                    self._export_parquet(result, blob, custom_time)

                elif path.endswith(".pickle"):
                    self._export_pickle(result, blob, custom_time)

                elif path.endswith(".json"):
                    self._export_json(result, blob, custom_time)

                return result

            return wrapper

        return decorator

    def __call__(self, path, custom_time: datetime | None = None) -> Callable:
        return self.export(path, custom_time)


class GSXCache:
    """
    Google Cloud Storage Cache for function results.

    This class provides caching functionality for function results using Google Cloud Storage.
    By default, it uses Parquet format for caching but can be configured to use JSON or Pickle.

    Parameters
    ----------
    gsx_instance : GSX
        An instance of the GSX class to use for storage operations.
    cache_format : str, optional
        The file format to use for caching. Default is "parquet".
        Supported formats: "parquet", "json", "pickle"
    cache_prefix : str, optional
        The prefix to use for cache keys. Default is "cache/"

    Attributes
    ----------
    gsx : GSX
        The GSX instance used for storage operations.
    cache_format : str
        The file format used for caching.
    cache_prefix : str
        The prefix used for cache keys.
    """

    def __init__(
        self, gsx_instance, cache_format: str = "parquet", cache_prefix: str = "cache/"
    ):
        self.gsx = gsx_instance
        self.cache_format = cache_format.lower()
        self.cache_prefix = cache_prefix

        # Validate cache format
        if self.cache_format not in ["parquet", "json", "pickle"]:
            raise ValueError(
                f"Unsupported cache format: {cache_format}. Supported formats: parquet, json, pickle"
            )

    def _generate_cache_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """
        Generate a cache key based on function name and arguments.

        Parameters
        ----------
        func_name : str
            The name of the function being cached.
        args : tuple
            Positional arguments passed to the function.
        kwargs : dict
            Keyword arguments passed to the function.

        Returns
        -------
        str
            A unique cache key for the function call.
        """
        # Create a string representation of the function call
        call_signature = {"func_name": func_name, "args": args, "kwargs": kwargs}

        # Convert to JSON string for consistent hashing
        call_str = json.dumps(call_signature, sort_keys=True, default=str)

        # Generate MD5 hash for the cache key
        cache_key = hashlib.md5(call_str.encode()).hexdigest()

        # Add appropriate file extension based on cache format
        extension = f".{self.cache_format}"

        return f"{self.cache_prefix}{func_name}/{cache_key}{extension}"

    def _serialize_result(self, result: Any) -> bytes:
        """
        Serialize the result based on the cache format.

        Parameters
        ----------
        result : Any
            The result to serialize.

        Returns
        -------
        bytes
            The serialized result.
        """
        if self.cache_format == "parquet":
            # For Parquet, we expect a DataFrame
            if not hasattr(result, "to_parquet"):
                raise ValueError(
                    f"Cannot cache non-DataFrame result in Parquet format. Got: {type(result)}"
                )

            with BytesIO() as buffer:
                result.to_parquet(buffer)
                return buffer.getvalue()

        elif self.cache_format == "json":
            return json.dumps(result, default=str).encode()

        elif self.cache_format == "pickle":
            return pickle.dumps(result)

        else:
            raise ValueError(f"Unsupported cache format: {self.cache_format}")

    def _deserialize_result(self, data: bytes) -> Any:
        """
        Deserialize the result based on the cache format.

        Parameters
        ----------
        data : bytes
            The serialized data.

        Returns
        -------
        Any
            The deserialized result.
        """
        if self.cache_format == "parquet":
            try:
                import pandas as pd

                return pd.read_parquet(BytesIO(data))
            except ImportError:
                raise ImportError("pandas is required for Parquet caching")

        elif self.cache_format == "json":
            return json.loads(data.decode())

        elif self.cache_format == "pickle":
            return pickle.loads(data)

        else:
            raise ValueError(f"Unsupported cache format: {self.cache_format}")

    def _get_cached_result(self, cache_key: str, ttl: timedelta | None = None) -> Any:
        """
        Retrieve a cached result from Google Storage.

        Parameters
        ----------
        cache_key : str
            The cache key to look up.
        ttl : timedelta, optional
            Time-to-live for the cached result. If provided, cached results
            older than this will be considered expired.

        Returns
        -------
        Any
            The cached result if found and valid, None otherwise.
        """
        try:
            blob = self.gsx.bucket.blob(cache_key)

            # Check if blob exists
            if not blob.exists():
                return None

            # Check TTL if provided
            if ttl is not None:
                blob.reload()  # Ensure we have the latest metadata
                if (
                    blob.time_created
                    and datetime.now(blob.time_created.tzinfo) - blob.time_created > ttl
                ):
                    return None

            # Download and deserialize the cached result
            cached_data = blob.download_as_bytes()
            return self._deserialize_result(cached_data)

        except (NotFound, Exception):
            return None

    def _set_cached_result(
        self, cache_key: str, result: Any, custom_time: datetime | None = None
    ) -> None:
        """
        Store a result in the cache.

        Parameters
        ----------
        cache_key : str
            The cache key to store under.
        result : Any
            The result to cache.
        custom_time : datetime, optional
            Custom timestamp for the blob's custom_time field.
        """
        try:
            blob = self.gsx.bucket.blob(cache_key)

            if custom_time is not None:
                blob.custom_time = custom_time

            # Serialize and upload the result
            serialized_data = self._serialize_result(result)

            # Set content type based on cache format
            content_type = {
                "parquet": "application/octet-stream",
                "json": "application/json",
                "pickle": "application/octet-stream",
            }.get(self.cache_format, "application/octet-stream")

            blob.upload_from_string(serialized_data, content_type=content_type)

        except Exception:
            # If caching fails, we don't want to break the original function
            pass

    def cache(
        self, ttl: timedelta | None = None, custom_time: datetime | None = None
    ) -> Callable:
        """
        Decorator to cache function results in Google Storage.

        Parameters
        ----------
        ttl : timedelta, optional
            Time-to-live for cached results. If not provided, cached results
            will be valid indefinitely.
        custom_time : datetime, optional
            Custom timestamp for the blob's custom_time field.

        Returns
        -------
        Callable
            A decorated function that caches its results.

        Examples
        --------
        >>> cache = GSXCache(gsx_instance, cache_format="parquet")
        >>> @cache.cache(ttl=timedelta(hours=1))
        ... def get_dataframe():
        ...     return pd.DataFrame({"x": [1, 2, 3]})

        >>> cache = GSXCache(gsx_instance, cache_format="json")
        >>> @cache.cache()
        ... def get_data():
        ...     return {"key": "value"}
        """

        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Generate cache key
                cache_key = self._generate_cache_key(func.__name__, args, kwargs)

                # Try to get cached result
                cached_result = self._get_cached_result(cache_key, ttl)

                if cached_result is not None:
                    return cached_result

                # Cache miss - execute the function
                result = func(*args, **kwargs)

                # Store result in cache
                self._set_cached_result(cache_key, result, custom_time)

                return result

            return wrapper

        return decorator

    def __call__(
        self, ttl: timedelta | None = None, custom_time: datetime | None = None
    ) -> Callable:
        """Shorthand for cache() method."""
        return self.cache(ttl, custom_time)

    def clear_cache(self, func_name: str | None = None) -> None:
        """
        Clear cached results.

        Parameters
        ----------
        func_name : str, optional
            If provided, only clear cache for this specific function.
            If not provided, clear all cached results.
        """
        try:
            if func_name:
                # Clear cache for specific function
                prefix = f"{self.cache_prefix}{func_name}/"
            else:
                # Clear all cache
                prefix = self.cache_prefix

            # List and delete all blobs with the cache prefix
            blobs = self.gsx.bucket.list_blobs(prefix=prefix)
            for blob in blobs:
                blob.delete()

        except Exception:
            # If clearing cache fails, we don't want to break anything
            pass
