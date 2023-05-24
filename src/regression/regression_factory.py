from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from regression.regression_types import Types


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
        if self.model_type == Types.LINEAR_REGRESSION.value:
            return LinearRegression()
        elif self.model_type == Types.XGBOOSTREGRESS.value:
            return XGBRegressor()
        else:
            raise ValueError("Invalid model type")
