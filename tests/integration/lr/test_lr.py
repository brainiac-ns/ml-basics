import logging
import os
import shutil
import unittest
from unittest.mock import patch

from regression.regression_model import RegressionTask

LOGGER = logging.getLogger(__name__)


class TestLinearRegression(unittest.TestCase):
    @patch("boto3.Session.client")
    def setUp(self, mock_client) -> None:
        mock_client.upload_file.return_value = None
        self.lin_reg_model = RegressionTask(
            path="tests/integration/lr/test_data/test.csv",
            model_path="test-models/test.sav",
            bucket_name="",
        )
        os.mkdir("test-models/")

    def tearDown(self) -> None:
        shutil.rmtree("test-models")

    def test_predict(self):
        self.lin_reg_model.preprocess()

        self.lin_reg_model.train()
        self.assertEqual(os.listdir("test-models/")[0], "test.sav")

        metric = self.lin_reg_model.evaluate()
        self.assertGreater(metric, 0)
