import logging
import unittest
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier, XGBRegressor

from classification.model_factory import Factory
from classification.model_types import Types

LOGGER = logging.getLogger(__name__)


class TestFactory(unittest.TestCase):
    def test_factory(self):
        factory = Factory(Types.LOGISTIC_REGRESSION.value)
        self.assertIsInstance(factory(), LogisticRegression)

        factory = Factory(Types.DECISION_TREE.value)
        self.assertIsInstance(factory(), DecisionTreeClassifier)

        factory = Factory(Types.RANDOM_FORESTS.value)
        self.assertIsInstance(factory(), RandomForestClassifier)

        factory = Factory(Types.SUPPORT_VECTOR_MACHINE.value)
        self.assertIsInstance(factory(), SVC)

        factory = Factory(Types.GAUSSIAN_NB.value)
        self.assertIsInstance(factory(), GaussianNB)

        factory = Factory(Types.KNN.value)
        self.assertIsInstance(factory(), KNeighborsClassifier)

        factory = Factory(Types.XGBOOSTCLASS.value)
        self.assertIsInstance(factory(), XGBClassifier)

        factory = Factory(Types.XGBOOSTREGRESS.value)
        self.assertIsInstance(factory(), XGBRegressor)

        invalid_type = "INVALID_TYPE"
        factory = Factory(invalid_type)
        with pytest.raises(ValueError) as exc_info:
            factory()
        LOGGER.info(str(exc_info.value))
        self.assertEqual(str(exc_info.value), "Invalid model type")
