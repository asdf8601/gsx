import pandas as pd
from gsx import GSX, GSXCache
import json
from datetime import datetime, timedelta
import pickle


def test_build_client(mocker):
    MockClient = mocker.patch("gsx.storage.Client")
    GSX("project_id", "bucket_name")
    MockClient.assert_called_with(project="project_id")


def test_export_parquet(mocker):
    # Mock the internals
    from io import BytesIO

    mocker.patch("gsx.storage.Client")
    mock_export_parquet = mocker.patch.object(GSX, "_export_parquet")
    mock_blob = mocker.Mock()
    gcp_instance = GSX("project_id", "bucket_name")
    mocker.patch.object(
        gcp_instance,
        "bucket",
        mocker.MagicMock(blob=mocker.MagicMock(return_value=mock_blob)),
    )

    fake_df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    fake_buffer = BytesIO()
    fake_df.to_parquet(fake_buffer)

    mock_func = mocker.Mock(return_value=fake_df)

    # Apply the decorator and call the function
    decorated_func = gcp_instance.export("path.parquet")(mock_func)
    decorated_func()

    # Check if _export_parquet was called
    mock_export_parquet.assert_called_with(fake_df, mock_blob, None)


def test_export_pickle(mocker):
    mocker.patch("gsx.storage.Client")
    mock_bytesio = mocker.patch("gsx.BytesIO")
    gcp_instance = GSX("project_id", "bucket_name")
    blob = mocker.Mock()
    result = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

    gcp_instance._export_pickle(result, blob)

    mock_bytesio.return_value.__enter__.return_value.getvalue.assert_called()
    blob.upload_from_string.assert_called()


def test_export_json(mocker):
    mocker.patch("gsx.storage.Client")
    mock_json = mocker.patch("gsx.json")
    gcp_instance = GSX("project_id", "bucket_name")
    blob = mocker.Mock()
    mock_dict = {"key": "value"}

    gcp_instance._export_json(mock_dict, blob)

    mock_json.dumps.assert_called_with(mock_dict)
    blob.upload_from_string.assert_called()


def test_export_generic_records(mocker):
    mocker.patch("gsx.storage.Client")
    gcp_instance = GSX("project_id", "bucket_name")
    blob = mocker.Mock()

    records = [{"key": "value"}, {"key2": "value2"}]
    gcp_instance._export_generic(
        records,
        blob,
        func=lambda lst: "\n".join(map(json.dumps, lst)),
        content_type="application/text",
    )

    blob.upload_from_string.assert_called_with(
        '{"key": "value"}\n{"key2": "value2"}',
        content_type="application/text",
    )


def test_export_decorator(mocker):
    mocker.patch("gsx.storage.Client")
    mock_export_parquet = mocker.patch("gsx.GSX._export_parquet")
    gcp_instance = GSX("project_id", "bucket_name")
    blob = mocker.Mock()
    mock_func = mocker.Mock()
    mock_func.return_value = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

    mocker.patch.object(gcp_instance.bucket, "blob", return_value=blob)

    decorator = gcp_instance.export("path.parquet")
    decorated_func = decorator(mock_func)
    decorated_func()

    mock_func.assert_called()
    mock_export_parquet.assert_called_with(mock_func.return_value, blob, None)


def test_export_decorator_automatic(mocker):
    mocker.patch("gsx.storage.Client")
    mock_export_parquet = mocker.patch("gsx.GSX._export_parquet")

    gcp_instance = GSX("project_id", "bucket_name")
    blob = mocker.Mock()
    mocker.patch.object(gcp_instance.bucket, "blob", return_value=blob)

    mock_func = mocker.Mock()
    mock_func.return_value = pd.DataFrame({"a": [1, 2], "b": [3, 5]})

    decorator = gcp_instance("path.parquet")
    decorated_func = decorator(mock_func)
    decorated_func()

    mock_func.assert_called()
    mock_export_parquet.assert_called_with(mock_func.return_value, blob, None)


def test_export_with_custom_time(mocker):
    mocker.patch("gsx.storage.Client")
    mock_export_parquet = mocker.patch("gsx.GSX._export_parquet")

    gcp_instance = GSX("project_id", "bucket_name")
    blob = mocker.Mock()
    mocker.patch.object(gcp_instance.bucket, "blob", return_value=blob)

    mock_func = mocker.Mock()
    mock_func.return_value = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

    custom_time = datetime(2023, 1, 1, 12, 0, 0)
    decorator = gcp_instance.export("path.parquet", custom_time=custom_time)
    decorated_func = decorator(mock_func)
    decorated_func()

    mock_func.assert_called()
    mock_export_parquet.assert_called_with(mock_func.return_value, blob, custom_time)


def test_export_parquet_with_custom_time(mocker):
    mocker.patch("gsx.storage.Client")
    gcp_instance = GSX("project_id", "bucket_name")
    blob = mocker.Mock()
    result = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    custom_time = datetime(2023, 1, 1, 12, 0, 0)

    gcp_instance._export_parquet(result, blob, custom_time)

    assert blob.custom_time == custom_time
    blob.upload_from_string.assert_called()


def test_export_json_with_custom_time(mocker):
    mocker.patch("gsx.storage.Client")
    gcp_instance = GSX("project_id", "bucket_name")
    blob = mocker.Mock()
    result = {"key": "value"}
    custom_time = datetime(2023, 1, 1, 12, 0, 0)

    gcp_instance._export_json(result, blob, custom_time)

    assert blob.custom_time == custom_time
    blob.upload_from_string.assert_called_with(
        json.dumps(result), content_type="application/json"
    )


# GSXCache Tests
def test_gsxcache_init(mocker):
    mocker.patch("gsx.storage.Client")
    gcp_instance = GSX("project_id", "bucket_name")

    # Test default format
    cache = GSXCache(gcp_instance)
    assert cache.cache_format == "parquet"
    assert cache.cache_prefix == "cache/"

    # Test custom format
    cache = GSXCache(gcp_instance, cache_format="json", cache_prefix="custom_cache/")
    assert cache.cache_format == "json"
    assert cache.cache_prefix == "custom_cache/"


def test_gsxcache_invalid_format(mocker):
    mocker.patch("gsx.storage.Client")
    gcp_instance = GSX("project_id", "bucket_name")

    # Test invalid format raises error
    try:
        GSXCache(gcp_instance, cache_format="invalid")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Unsupported cache format" in str(e)


def test_gsxcache_generate_cache_key(mocker):
    mocker.patch("gsx.storage.Client")
    gcp_instance = GSX("project_id", "bucket_name")
    cache = GSXCache(gcp_instance, cache_format="parquet")

    # Test cache key generation
    cache_key = cache._generate_cache_key("test_func", (1, 2), {"key": "value"})

    assert cache_key.startswith("cache/test_func/")
    assert cache_key.endswith(".parquet")
    assert (
        len(cache_key.split("/")[-1].replace(".parquet", "")) == 32
    )  # MD5 hash length

    # Test that same inputs produce same cache key
    cache_key2 = cache._generate_cache_key("test_func", (1, 2), {"key": "value"})
    assert cache_key == cache_key2

    # Test that different inputs produce different cache keys
    cache_key3 = cache._generate_cache_key("test_func", (1, 3), {"key": "value"})
    assert cache_key != cache_key3


def test_gsxcache_serialize_deserialize_parquet(mocker):
    mocker.patch("gsx.storage.Client")
    gcp_instance = GSX("project_id", "bucket_name")
    cache = GSXCache(gcp_instance, cache_format="parquet")

    # Test DataFrame serialization
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    serialized = cache._serialize_result(df)

    # Should be bytes
    assert isinstance(serialized, bytes)

    # Test deserialization
    deserialized = cache._deserialize_result(serialized)
    pd.testing.assert_frame_equal(df, deserialized)


def test_gsxcache_serialize_deserialize_json(mocker):
    mocker.patch("gsx.storage.Client")
    gcp_instance = GSX("project_id", "bucket_name")
    cache = GSXCache(gcp_instance, cache_format="json")

    # Test JSON serialization
    data = {"key": "value", "number": 42}
    serialized = cache._serialize_result(data)

    # Should be bytes
    assert isinstance(serialized, bytes)

    # Test deserialization
    deserialized = cache._deserialize_result(serialized)
    assert deserialized == data


def test_gsxcache_serialize_deserialize_pickle(mocker):
    mocker.patch("gsx.storage.Client")
    gcp_instance = GSX("project_id", "bucket_name")
    cache = GSXCache(gcp_instance, cache_format="pickle")

    # Test pickle serialization
    data = {"key": "value", "complex": [1, 2, 3]}
    serialized = cache._serialize_result(data)

    # Should be bytes
    assert isinstance(serialized, bytes)

    # Test deserialization
    deserialized = cache._deserialize_result(serialized)
    assert deserialized == data


def test_gsxcache_serialize_non_dataframe_parquet(mocker):
    mocker.patch("gsx.storage.Client")
    gcp_instance = GSX("project_id", "bucket_name")
    cache = GSXCache(gcp_instance, cache_format="parquet")

    # Test that non-DataFrame raises error for parquet format
    try:
        cache._serialize_result({"key": "value"})
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Cannot cache non-DataFrame result in Parquet format" in str(e)


def test_gsxcache_cache_hit_parquet(mocker):
    mocker.patch("gsx.storage.Client")
    gcp_instance = GSX("project_id", "bucket_name")
    cache = GSXCache(gcp_instance, cache_format="parquet")

    # Mock blob that exists and has cached data
    mock_blob = mocker.Mock()
    mock_blob.exists.return_value = True

    # Create a test DataFrame and serialize it
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    serialized_df = cache._serialize_result(df)
    mock_blob.download_as_bytes.return_value = serialized_df

    mocker.patch.object(gcp_instance.bucket, "blob", return_value=mock_blob)

    # Mock the function that would be cached
    mock_func = mocker.Mock(return_value=pd.DataFrame({"x": [5, 6]}))
    mock_func.__name__ = "test_func"

    # Apply cache decorator
    cached_func = cache.cache()(mock_func)

    # Call the function
    result = cached_func("arg1", key="value")

    # Should return cached result
    pd.testing.assert_frame_equal(result, df)
    # Original function should not be called
    mock_func.assert_not_called()


def test_gsxcache_cache_miss_json(mocker):
    mocker.patch("gsx.storage.Client")
    gcp_instance = GSX("project_id", "bucket_name")
    cache = GSXCache(gcp_instance, cache_format="json")

    # Mock blob that doesn't exist
    mock_blob = mocker.Mock()
    mock_blob.exists.return_value = False
    mocker.patch.object(gcp_instance.bucket, "blob", return_value=mock_blob)

    # Mock the function that would be cached
    mock_func = mocker.Mock(return_value={"fresh": "result"})
    mock_func.__name__ = "test_func"

    # Apply cache decorator
    cached_func = cache.cache()(mock_func)

    # Call the function
    result = cached_func("arg1", key="value")

    # Should return fresh result
    assert result == {"fresh": "result"}
    # Original function should be called
    mock_func.assert_called_once_with("arg1", key="value")
    # Result should be cached
    mock_blob.upload_from_string.assert_called_once()


def test_gsxcache_cache_with_ttl_expired(mocker):
    mocker.patch("gsx.storage.Client")
    gcp_instance = GSX("project_id", "bucket_name")
    cache = GSXCache(gcp_instance, cache_format="pickle")

    # Mock blob that exists but is expired
    mock_blob = mocker.Mock()
    mock_blob.exists.return_value = True
    mock_blob.time_created = datetime.now() - timedelta(hours=2)  # 2 hours ago
    mock_blob.download_as_bytes.return_value = pickle.dumps("expired_result")
    mocker.patch.object(gcp_instance.bucket, "blob", return_value=mock_blob)

    # Mock the function that would be cached
    mock_func = mocker.Mock(return_value="fresh_result")
    mock_func.__name__ = "test_func"

    # Apply cache decorator with 1 hour TTL
    cached_func = cache.cache(ttl=timedelta(hours=1))(mock_func)

    # Call the function
    result = cached_func("arg1", key="value")

    # Should return fresh result because cache is expired
    assert result == "fresh_result"
    # Original function should be called
    mock_func.assert_called_once_with("arg1", key="value")


def test_gsxcache_cache_with_ttl_valid(mocker):
    mocker.patch("gsx.storage.Client")
    gcp_instance = GSX("project_id", "bucket_name")
    cache = GSXCache(gcp_instance, cache_format="pickle")

    # Mock blob that exists and is not expired
    mock_blob = mocker.Mock()
    mock_blob.exists.return_value = True
    mock_blob.time_created = datetime.now() - timedelta(minutes=30)  # 30 minutes ago
    mock_blob.download_as_bytes.return_value = pickle.dumps("valid_cached_result")
    mocker.patch.object(gcp_instance.bucket, "blob", return_value=mock_blob)

    # Mock the function that would be cached
    mock_func = mocker.Mock(return_value="fresh_result")
    mock_func.__name__ = "test_func"

    # Apply cache decorator with 1 hour TTL
    cached_func = cache.cache(ttl=timedelta(hours=1))(mock_func)

    # Call the function
    result = cached_func("arg1", key="value")

    # Should return cached result because it's still valid
    assert result == "valid_cached_result"
    # Original function should not be called
    mock_func.assert_not_called()


def test_gsxcache_clear_cache_specific_function(mocker):
    mocker.patch("gsx.storage.Client")
    gcp_instance = GSX("project_id", "bucket_name")
    cache = GSXCache(gcp_instance)

    # Mock blobs
    mock_blob1 = mocker.Mock()
    mock_blob2 = mocker.Mock()
    mock_blobs = [mock_blob1, mock_blob2]

    mocker.patch.object(gcp_instance.bucket, "list_blobs", return_value=mock_blobs)

    # Clear cache for specific function
    cache.clear_cache("test_func")

    # Should list blobs with specific prefix
    gcp_instance.bucket.list_blobs.assert_called_with(prefix="cache/test_func/")

    # Should delete all found blobs
    mock_blob1.delete.assert_called_once()
    mock_blob2.delete.assert_called_once()


def test_gsxcache_clear_cache_all(mocker):
    mocker.patch("gsx.storage.Client")
    gcp_instance = GSX("project_id", "bucket_name")
    cache = GSXCache(gcp_instance)

    # Mock blobs
    mock_blob1 = mocker.Mock()
    mock_blob2 = mocker.Mock()
    mock_blobs = [mock_blob1, mock_blob2]

    mocker.patch.object(gcp_instance.bucket, "list_blobs", return_value=mock_blobs)

    # Clear all cache
    cache.clear_cache()

    # Should list blobs with cache prefix
    gcp_instance.bucket.list_blobs.assert_called_with(prefix="cache/")

    # Should delete all found blobs
    mock_blob1.delete.assert_called_once()
    mock_blob2.delete.assert_called_once()


def test_gsxcache_cache_with_custom_time(mocker):
    mocker.patch("gsx.storage.Client")
    gcp_instance = GSX("project_id", "bucket_name")
    cache = GSXCache(gcp_instance, cache_format="json")

    # Mock blob that doesn't exist (cache miss)
    mock_blob = mocker.Mock()
    mock_blob.exists.return_value = False
    mocker.patch.object(gcp_instance.bucket, "blob", return_value=mock_blob)

    # Mock the function that would be cached
    mock_func = mocker.Mock(return_value={"fresh": "result"})
    mock_func.__name__ = "test_func"

    # Apply cache decorator with custom time
    custom_time = datetime(2023, 1, 1, 12, 0, 0)
    cached_func = cache.cache(custom_time=custom_time)(mock_func)

    # Call the function
    result = cached_func("arg1", key="value")

    # Should return fresh result
    assert result == {"fresh": "result"}
    # Custom time should be set on blob
    assert mock_blob.custom_time == custom_time


def test_gsxcache_shorthand_call(mocker):
    mocker.patch("gsx.storage.Client")
    gcp_instance = GSX("project_id", "bucket_name")
    cache = GSXCache(gcp_instance, cache_format="json")

    # Mock blob that doesn't exist
    mock_blob = mocker.Mock()
    mock_blob.exists.return_value = False
    mocker.patch.object(gcp_instance.bucket, "blob", return_value=mock_blob)

    # Mock the function that would be cached
    mock_func = mocker.Mock(return_value={"data": "value"})
    mock_func.__name__ = "test_func"

    # Apply cache decorator using shorthand
    cached_func = cache()(mock_func)

    # Call the function
    result = cached_func("arg1", key="value")

    # Should return fresh result
    assert result == {"data": "value"}
    # Original function should be called
    mock_func.assert_called_once_with("arg1", key="value")
