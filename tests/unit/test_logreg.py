import logging
import os
import shutil
import unittest
from unittest.mock import patch

import pandas as pd

from classification.classification_model import ClassificationTask
from classification.classification_types import Types

LOGGER = logging.getLogger(__name__)


class TestLogisticRegression(unittest.TestCase):
    def setUp(self) -> None:
        os.mkdir("test-models/")

    def tearDown(self) -> None:
        shutil.rmtree("test-models")

    @patch("boto3.Session.client")
    @patch("pandas.read_csv")
    def test_train(self, mock_reading, mock_client):
        data = {
            "label": [0, 1, 0, 1, 0],
            "pixel1": [1, 2, 3, 4, 5],
            "pixel2": [2, 4, 6, 4, 2],
        }
        df = pd.DataFrame(data)
        mock_reading.return_value = df
        mock_client.upload_file.return_value = None
        log_reg_model = ClassificationTask(
            model_path="models/test.sav",
            bucket_name="",
            model_type=Types.LOGISTIC_REGRESSION.value,
        )
        log_reg_model.train()
        self.assertEqual(os.listdir("models/")[2], "test.sav")

    @patch("boto3.Session.client")
    @patch("pandas.read_csv")
    def test_predict(self, mock_reading, mock_client):
        data = {
            "label": [0, 1, 0, 1, 0],
            "pixel1": [1, 2, 3, 4, 5],
            "pixel2": [2, 4, 6, 4, 2],
        }
        df = pd.DataFrame(data)
        mock_reading.return_value = df
        mock_client.upload_file.return_value = None
        log_reg_model = ClassificationTask(
            model_path="test-models/test.sav",
            bucket_name="",
            model_type=Types.LOGISTIC_REGRESSION.value,
        )
        log_reg_model.train()
        self.X_test = df.drop("label", axis=1).values
        predictions = log_reg_model.predict(self.X_test)
        self.assertIsNotNone(predictions)

    @patch("boto3.Session.client")
    @patch("pandas.read_csv")
    def test_evaluate(self, mock_reading, mock_client):
        data = {
            "label": [0, 1, 0, 1, 0],
            "pixel1": [1, 2, 3, 4, 5],
            "pixel2": [2, 4, 6, 4, 2],
        }
        df = pd.DataFrame(data)
        mock_reading.return_value = df
        mock_client.upload_file.return_value = None
        log_reg_model = ClassificationTask(
            model_path="test-models/test.sav",
            bucket_name="",
            model_type=Types.LOGISTIC_REGRESSION.value,
        )
        log_reg_model.train()
        evaluated = log_reg_model.evaluate()
        self.assertGreater(evaluated, 0)
