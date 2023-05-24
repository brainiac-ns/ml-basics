import logging
import os
from abc import ABC, abstractmethod
from io import StringIO
from typing import List

import boto3
import numpy as np
import pandas as pd
from dotenv import load_dotenv

LOGGER = logging.getLogger(__name__)


class Base(ABC):
    def __init__(self, model_path: str, bucket_name: str = "ml-basic") -> None:
        load_dotenv()
        self.session = boto3.Session(
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        )
        try:
            self.s3_client = self.session.client("s3")
        except Exception as e:
            LOGGER.exception(e)

        self.model_path = model_path
        self.bucket_name = bucket_name

    def read_data(self, path: str) -> pd.DataFrame:
        if not self.bucket_name:
            return pd.read_csv(path)

        response = self.s3_client.get_object(Bucket=self.bucket_name, Key=path)
        csv_data = response["Body"].read().decode("utf-8")

        df = pd.read_csv(StringIO(csv_data))

        return df

    def upload_model(self, file_key: str):
        self.s3_client.upload_file(self.model_path, self.bucket_name, file_key)

    @abstractmethod
    def preprocess(
        self,
        factorization_list: List[str],
        columns: List[str],
        target: List[str],
        normalize_column_name: str,
    ) -> None:
        pass

    @abstractmethod
    def train(self) -> None:
        pass

    @abstractmethod
    def predict(self, X) -> np.array:
        pass
