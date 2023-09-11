from functools import wraps
import warnings
from google.cloud import storage
import json
from io import BytesIO
from functools import partial
from typing import Callable


def build(project_id, bucket_name):
    """
    Build a partial function that converts data to GCP format.

    Parameters
    ----------
    project_id : str
        The ID of the GCP project.
    bucket_name : str
        The name of the GCP bucket.

    Returns
    -------
    partial function
        A partial function that converts data to GCP format, with the
        project_id and bucket_name pre-filled.
    """
    return partial(gcpx, project_id=project_id, bucket_name=bucket_name)


def gcpx(
    path: str,
    project_id: str | None = None,
    bucket_name: str = "my-bucket",
) -> Callable:
    """
    Decorator function that uploads the result of a function to Google Cloud
    Storage.

    Parameters
    ----------
    path : str
        The path of the file to be uploaded.
    project_id : str, optional
        The ID of the Google Cloud project. If not provided, the default
        project ID will be used.
    bucket_name : str, optional
        The name of the Google Cloud Storage bucket. Default is "my-bucket".

    Returns
    -------
    function
        The decorated function.

    Examples
    --------
    @gcpx("data.parquet", project_id="my-project", bucket_name="my-bucket")
    def process_data():
        # process data and return result
        return result
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        storage_client = storage.Client(project=project_id)
        bucket = storage_client.bucket(bucket_name)

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)

            blob_name = path
            blob = bucket.blob(blob_name)
            print(f"saving {blob}")

            if path.endswith(".parquet"):
                with BytesIO() as buffer:
                    result.to_parquet(buffer)
                    blob.upload_from_string(
                        buffer.getvalue(),
                        content_type="application/octet-stream",
                    )

            elif path.endswith(".pickle"):
                with BytesIO() as buffer:
                    result.to_pickle(buffer)
                    blob.upload_from_string(
                        buffer.getvalue(),
                        content_type="application/octet-stream",
                    )

            elif path.endswith(".json"):
                blob.upload_from_string(
                    json.dumps(result), content_type="application/json"
                )

            return result

        return wrapper

    return decorator


if __name__ == "__main__":
    # Ejemplo de uso
    import datetime
    import pandas as pd

    my_project = "maximal-record-121815"
    bucket = "st-datascience-workspace"
    to_datascience = build(project_id=my_project, bucket_name=bucket)
    date = datetime.datetime.now().strftime("%Y%m%d%H%M")

    @to_datascience(f"togcp/{date}/test.parquet")
    def test_parquet() -> pd.DataFrame:
        print("Foo called!")
        return pd.DataFrame({"a": [1, 2], "b": [3, 4]})

    @to_datascience(f"togcp/{date}/test2.pickle")
    def test_pickle() -> pd.DataFrame:
        print("Boo called!")
        return pd.DataFrame({"a": [1, 2], "b": [3, 4]})

    @to_datascience(f"togcp/{date}/test3.json")
    def test_dict() -> dict:
        print("Moo called!")
        return {"key": "value"}

    test_parquet()
    test_pickle()
    test_dict()
