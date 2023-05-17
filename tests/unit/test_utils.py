import logging
import unittest

import pandas as pd
import pytest

from src.exceptions.custom_exceptions import NullDfException
from src.utils import normalize_column

LOGGER = logging.getLogger(__name__)


class TestUtils(unittest.TestCase):
    def test_normalize(self):
        data = {"data": [0, 1, 10, 100]}
        df = pd.DataFrame(data)
        expected = pd.DataFrame({"data": [0, 0.01, 0.1, 1]})
        normalized_df = normalize_column(df, "data")
        self.assertTrue(expected.equals(normalized_df))

    def test_null_input_normalize(self):
        df = pd.DataFrame()
        normalized_df = normalize_column(df, "data")
        self.assertTrue(df.equals(normalized_df))

    def test_null_input_normalize_2(self):
        with pytest.raises(NullDfException) as exc_info:
            normalize_column(None, "data")

        LOGGER.info(str(exc_info.value))
        self.assertEqual(str(exc_info.value), "DF is None")
