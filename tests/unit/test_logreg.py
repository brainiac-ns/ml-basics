import logging
import os
import shutil
import unittest
from unittest.mock import patch

import pandas as pd

from logistic_regression.app import LogisticReg

LOGGER = logging.getLogger(__name__)


class TestLogisticRegression(unittest.TestCase):
    def setUp(self) -> None:
        os.mkdir("test-models/")

    def tearDown(self) -> None:
        shutil.rmtree("test-models")

    @patch("pandas.read_csv")
    def test_train(self, mock_reading):
        data = {
            "label": [0, 1, 0, 1, 0],
            "pixel1": [1, 2, 3, 4, 5],
            "pixel2": [2, 4, 6, 4, 2],
        }
        df = pd.DataFrame(data)
        mock_reading.return_value = df
        log_reg_model = LogisticReg(model_path="test-models/test.sav")
        log_reg_model.train()
        self.assertEqual(os.listdir("test-models/")[0], "test.sav")

    @patch("pandas.read_csv")
    def test_predict(self, mock_reading):
        data = {
            "label": [0, 1, 0, 1, 0],
            "pixel1": [1, 2, 3, 4, 5],
            "pixel2": [2, 4, 6, 4, 2],
        }
        df = pd.DataFrame(data)
        mock_reading.return_value = df
        log_reg_model = LogisticReg(model_path="test-models/test.sav")
        log_reg_model.train()
        self.X_test = df.drop("label", axis=1).values
        predictions = log_reg_model.predict(self.X_test)
        self.assertIsNotNone(predictions)

    @patch("pandas.read_csv")
    def test_evaluate(self, mock_reading):
        data = {
            "label": [0, 1, 0, 1, 0],
            "pixel1": [1, 2, 3, 4, 5],
            "pixel2": [2, 4, 6, 4, 2],
        }
        df = pd.DataFrame(data)
        mock_reading.return_value = df
        log_reg_model = LogisticReg(model_path="test-models/test.sav")
        log_reg_model.train()
        evaluated = log_reg_model.evaluate()
        self.assertGreater(evaluated, 0)
