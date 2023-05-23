import logging
import os
import shutil
import unittest
from unittest.mock import MagicMock, patch

from classification.classification_model import ClassificationTask
from classification.classification_types import Types

LOGGER = logging.getLogger(__name__)


class TestLogisticRegression(unittest.TestCase):
    @patch("boto3.Session.client")
    def setUp(self, mock_client) -> None:
        mock_client.return_value.upload_file = MagicMock()
        self.log_reg_model = ClassificationTask(
            train_path="tests/integration/log_reg/test_data/test.csv",
            test_path="tests/integration/log_reg/test_data/test.csv",
            model_path="test-models/logisticreg.sav",
            bucket_name="",
            model_type=Types.LOGISTIC_REGRESSION.value,
        )
        os.mkdir("test-models/")

    def tearDown(self) -> None:
        shutil.rmtree("test-models")

    def test_predict(self):
        self.log_reg_model.train()
        self.assertEqual(os.listdir("test-models/")[0], "logisticreg.sav")
        evaluate = self.log_reg_model.evaluate()
        self.assertGreater(evaluate, 0)
