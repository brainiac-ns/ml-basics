import unittest
from src.linear_regression.linear_regression import LinearReg
import shutil
import os
import logging

LOGGER = logging.getLogger(__name__)


class TestLinearRegression(unittest.TestCase):

    def setUp(self) -> None:
        self.lin_reg_model = LinearReg(
            path="tests/integration/lr/test_data/test.csv",
            model_path="test-models/test.sav")
        os.mkdir("test-models/")

    def tearDown(self) -> None:
        shutil.rmtree("test-models")

    def test_predict(self):
        self.lin_reg_model.preprocess()

        self.lin_reg_model.train_model()
        self.assertEqual(os.listdir("test-models/")[0], "test.sav")

        metric = self.lin_reg_model.evaluate()
        self.assertGreater(metric, 0)
