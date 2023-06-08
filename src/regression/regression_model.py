import logging
import pickle
from typing import List

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from base import Base
from regression.regression_factory import Factory
from regression.regression_types import Types
from utils import normalize_column

logging.basicConfig(level=logging.INFO, format="%(message)s")
LOGGER = logging.getLogger(__name__)


class RegressionTask(Base):
    def __init__(
        self,
        path: str = "data/regression/weatherHistory.csv",
        model_path: str = "models/trained.sav",
        bucket_name: str = "ml-basic",
        model_type: Types = Types.XGBOOSTREGRESS.value,
    ):
        """
        This is initializing the Regression task class

        Args:
            path (str): Path to the data file
            n_jobs (int): Number of times to run during training
            model_path (str): Path to the saved model file

        """
        super().__init__(model_path, bucket_name)
        self.df = self.read_data(path)
        factory = Factory(model_type)
        self.model = factory()
        self.test_path = self.read_data(path)
        self.model_path = model_path
        self.model_type = model_type

    def preprocess(
        self,
        factorization_list: List[str] = [
            "Summary",
            "Precip Type",
            "Daily Summary",
        ],
        columns: List[str] = [
            "Summary",
            "Precip Type",
            "Apparent Temperature (C)",
            "Humidity",
            "Wind Speed (km/h)",
            "Wind Bearing (degrees)",
            "Visibility (km)",
            "Loud Cover",
            "Pressure (millibars)",
            "Daily Summary",
        ],
        target: List[str] = ["Temperature (C)"],
        normalize_column_name: str = "Pressure (millibars)",
        num_components: int = 3,
    ) -> None:
        """
        Preprocess the data by normalizing a column and factorizing columns

        Returns:
            Dataframe: Preprocessed variable
            Series: Target variable

        """
        LOGGER.info("Preprocessing... (normalization and pca)")
        self.df = normalize_column(self.df, normalize_column_name)
        self.factorize(factorization_list)

        print("Data dimensions before PCA:")
        print(self.df[columns].shape)

        X = self.df[columns]
        y = self.df[target]

        pca = PCA(n_components=num_components)
        X_pca = pca.fit_transform(X)
        self.df_pca = pd.DataFrame(X_pca, columns=columns[:num_components])
        (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test,
        ) = train_test_split(self.df_pca, y, test_size=0.3, random_state=101)
        print("Data dimensions after PCA:")
        print(self.df_pca.shape)

    def factorize(self, columns) -> None:
        """
        Factorize the columns

        Args:
            columns: Column names that need to be factorized

        """
        LOGGER.info("Factorizing...")
        for c in columns:
            self.df[c] = pd.factorize(self.df[c])[0] + 1

    def train(self) -> None:
        """
        Train using regression model

        """
        LOGGER.info("Training started")
        trained = self.model.fit(self.X_train, self.y_train)
        pickle.dump(trained, open(self.model_path, "wb"))
        self.upload_model(self.model_path)
        LOGGER.info("Training ended")

    def evaluate(self) -> int:
        """
        Evaluating the model using RMSE

        Returns:
        - metric: Root mean squared error.

        """
        LOGGER.info("Evaluation started")
        metric = np.sqrt(
            metrics.mean_squared_error(self.y_test, self.predict(self.X_test))
        )
        print(f"RMSE: {metric}")
        LOGGER.info("Evaluation ended")
        return metric

    def predict(self, X) -> np.array:
        """
        Make predictions using the model

        Args:
        - X: Input data

        Returns:
        - predictions: Predicted values after using the model

        """
        trained_model = pickle.load(open(self.model_path, "rb"))
        return trained_model.predict(X)


if __name__ == "__main__":
    regression_model = RegressionTask("data/weatherHistory.csv")
    regression_model.preprocess()
    regression_model.train()
    evaluated_model = regression_model.evaluate()
