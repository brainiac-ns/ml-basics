import pickle
import time

import boto3
import numpy as np
from flask import Flask, request

app = Flask(__name__)

s3_client = boto3.client("s3")
bucket_name = "ml-basic"
model_key = "models/lin_reg/model.sav"

# workflow test


@app.route("/health")
def health():
    return "ALIVE"


@app.route("/healthh")
def healthh():
    return "ALIVEEEEE"


@app.route("/healthhh")
def healthhh():
    return "ALIVEEEEE3"


@app.route("/", methods=["POST"])
def hello():
    start_time = time.time()

    model = load_model_from_s3(bucket_name, "models/lin_reg")
    data = request.get_json()

    summary = data.get("Summary")
    precip_type = data.get("Precip Type")
    apparent_temp = data.get("Apparent Temp")
    humidity = data.get("Humidity")
    wind_speed = data.get("Wind Speed")
    wind_bearing = data.get("Wind Bearing")
    visibility = data.get("Visibility")
    loud_cover = data.get("Loud Cover")
    pressure = data.get("Pressure")
    daily_summary = data.get("Daily Summary")

    model = pickle.loads(model)

    prediction = model.predict(
        np.array(
            [
                summary,
                precip_type,
                apparent_temp,
                humidity,
                wind_speed,
                wind_bearing,
                visibility,
                loud_cover,
                pressure,
                daily_summary,
            ]
        ).reshape(1, -1)
    )

    end_time = time.time()
    execution_time_ms = (end_time - start_time) * 1000
    print(execution_time_ms)

    return str(prediction[0][0])


def load_model_from_s3(bucket_name, folder_name):
    s3_client = boto3.client("s3")

    # List objects in the specified S3 folder
    response = s3_client.list_objects_v2(
        Bucket=bucket_name, Prefix=folder_name
    )
    objects = response.get("Contents", [])

    # Sort objects by timestamp in descending order
    objects.sort(key=lambda obj: obj["LastModified"], reverse=True)

    # Check if the cache is empty or the latest model is newer than the cached model
    latest_model_timestamp = objects[0]["LastModified"]
    if (
        not hasattr(load_model_from_s3, "cached_model")
        or latest_model_timestamp
        > load_model_from_s3.cached_model["timestamp"]
    ):
        # Download the latest model from S3
        model_key = objects[0]["Key"]
        response = s3_client.get_object(Bucket=bucket_name, Key=model_key)
        model_data = response["Body"].read()

        # Cache the latest model along with its timestamp
        load_model_from_s3.cached_model = {
            "timestamp": latest_model_timestamp,
            "data": model_data,
        }

    return load_model_from_s3.cached_model["data"]


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
