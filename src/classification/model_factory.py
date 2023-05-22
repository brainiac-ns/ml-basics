from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier, XGBRegressor

from classification.model_types import Types


class Factory:
    def __init__(self, model_type: str):
        """
        Initialize the Factory class with the specified model type.

        Args:
            model_type: type of the machine learning model.
        """
        self.model_type = model_type

    def __call__(self) -> BaseEstimator:
        """
        Create and return machine learning model.

        Returns:
            Machine learning model.

        Raises:
            ValueError: If the model type is invalid.
        """
        if self.model_type == Types.LOGISTIC_REGRESSION.value:
            return LogisticRegression()
        elif self.model_type == Types.DECISION_TREE.value:
            return DecisionTreeClassifier()
        elif self.model_type == Types.RANDOM_FORESTS.value:
            return RandomForestClassifier()
        elif self.model_type == Types.SUPPORT_VECTOR_MACHINE.value:
            return SVC()
        elif self.model_type == Types.GAUSSIAN_NB.value:
            return GaussianNB()
        elif self.model_type == Types.KNN.value:
            return KNeighborsClassifier()
        elif self.model_type == Types.XGBOOSTCLASS.value:
            return XGBClassifier()
        elif self.model_type == Types.XGBOOSTREGRESS.value:
            return XGBRegressor()
        else:
            raise ValueError("Invalid model type")
