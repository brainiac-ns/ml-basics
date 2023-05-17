import logging
import os
import shutil
import unittest

from classification.generic_model import ClassificationTask

LOGGER = logging.getLogger(__name__)


class TestLogisticRegression(unittest.TestCase):
    def setUp(self) -> None:
        self.log_reg_model = ClassificationTask(
            train_path="tests/integration/log_reg/test_data/test.csv",
            test_path="tests/integration/log_reg/test_data/test.csv",
            model_path="test-models/logisticreg.sav",
            bucket_name="",
            model_type="Logistic Regression",
        )
        os.mkdir("test-models/")

    def tearDown(self) -> None:
        shutil.rmtree("test-models")

    def test_predict(self):
        self.log_reg_model.train()
        self.assertEqual(os.listdir("test-models/")[0], "logisticreg.sav")
        evaluate = self.log_reg_model.evaluate()
        self.assertGreater(evaluate, 0)
