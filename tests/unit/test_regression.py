import logging
import unittest

import pytest
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from regression.regression_factory import Factory
from regression.regression_types import Types

LOGGER = logging.getLogger(__name__)


class TestFactory(unittest.TestCase):
    def test_factory(self):
        factory = Factory(Types.LINEAR_REGRESSION.value)
        self.assertIsInstance(factory(), LinearRegression)

        factory = Factory(Types.XGBOOSTREGRESS.value)
        self.assertIsInstance(factory(), XGBRegressor)

        invalid_type = "INVALID_TYPE"
        factory = Factory(invalid_type)
        with pytest.raises(ValueError) as exc_info:
            factory()
        LOGGER.info(str(exc_info.value))
        self.assertEqual(str(exc_info.value), "Invalid model type")
