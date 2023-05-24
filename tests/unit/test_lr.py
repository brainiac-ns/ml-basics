import logging
import os
import shutil
import unittest
from unittest.mock import patch

import pandas as pd

from regression.regression_model import RegressionTask

LOGGER = logging.getLogger(__name__)


class TestLinearRegression(unittest.TestCase):
    def setUp(self) -> None:
        os.mkdir("test-models/")

    def tearDown(self) -> None:
        shutil.rmtree("test-models")

    @patch("pandas.read_csv")
    def test_preprocess(self, mock_reading):
        data = {
            "A": [1, 2, 3, 4, 5],
            "B": [7.388889, 10, 20, 1, 2],
            "C": [1, 3, 4, 1, 2],
            "D": [1, 40, 2, 4, 3],
        }
        df = pd.DataFrame(data)
        mock_reading.return_value = df
        lin_reg_model = RegressionTask(bucket_name="")
        lin_reg_model.preprocess(
            factorization_list=[],
            columns=["A", "B", "C"],
            target=["D"],
            normalize_column_name="B",
            num_components=2,
        )

        self.assertEqual(lin_reg_model.X_train.shape, (3, 2))
        self.assertEqual(lin_reg_model.y_train.shape, (3, 1))
        self.assertEqual(lin_reg_model.X_test.shape, (2, 2))
        self.assertEqual(lin_reg_model.y_test.shape, (2, 1))

    @patch("pandas.read_csv")
    def test_factorize_columns(self, mock_reading):
        data = {"A": ["cloudy", "sunny"], "B": ["sunny", "rainy"]}
        df = pd.DataFrame(data)
        mock_reading.return_value = df
        lin_reg_model = RegressionTask(df, bucket_name="")
        lin_reg_model.factorize(["A", "B"])

        self.assertEqual(list(lin_reg_model.df["A"]), [1, 2])
        self.assertEqual(list(lin_reg_model.df["B"]), [1, 2])

    @patch("boto3.Session.client")
    @patch("pandas.read_csv")
    def test_train_model(self, mock_reading, mock_client):
        data = {
            "A": [1, 2, 3, 4, 5],
            "B": [7.388889, 10, 20, 1, 2],
            "C": [1, 3, 4, 1, 2],
            "D": [1, 40, 2, 4, 3],
        }
        df = pd.DataFrame(data)
        mock_reading.return_value = df
        mock_client.upload_file.return_value = None
        lin_reg_model = RegressionTask(model_path="models/trained.sav", bucket_name="")
        lin_reg_model.preprocess(
            factorization_list=[],
            columns=["A", "B", "C"],
            target=["D"],
            normalize_column_name="B",
        )

        lin_reg_model.train()
        self.assertEqual(os.listdir("models/")[2], "trained.sav")

    @patch("boto3.Session.client")
    @patch("pandas.read_csv")
    def test_predict(self, mock_reading, mock_client):
        data = {
            "A": [1, 2, 3, 4, 5],
            "B": [7.388889, 10, 20, 1, 2],
            "C": [1, 3, 4, 1, 2],
            "D": [1, 40, 2, 4, 3],
        }
        df = pd.DataFrame(data)
        mock_reading.return_value = df
        mock_client.upload_file.return_value = None
        lin_reg_model = RegressionTask(model_path="models/trained.sav", bucket_name="")
        lin_reg_model.preprocess(
            factorization_list=[],
            columns=["A", "B", "C"],
            target=["D"],
            normalize_column_name="B",
        )
        lin_reg_model.train()
        X_test = df[["A", "B", "C"]]
        predictions = lin_reg_model.predict(X_test)
        self.assertIsNotNone(predictions)

    @patch("boto3.Session.client")
    @patch("pandas.read_csv")
    def test_evaluate(self, mock_reading, mock_client):
        data = {
            "A": [1, 2, 3, 4, 5],
            "B": [7.388889, 10, 20, 1, 2],
            "C": [1, 3, 4, 1, 2],
            "D": [1, 40, 2, 4, 3],
        }
        df = pd.DataFrame(data)
        mock_reading.return_value = df
        mock_client.upload_file.return_value = None
        lin_reg_model = RegressionTask(model_path="models/trained.sav", bucket_name="")
        lin_reg_model.preprocess(
            factorization_list=[],
            columns=["A", "B", "C"],
            target=["D"],
            normalize_column_name="B",
        )
        lin_reg_model.train()
        metric = lin_reg_model.evaluate()
        self.assertGreater(metric, 0)
