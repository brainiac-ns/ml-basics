from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.linear_model import LinearRegression
from utils import normalize_column
from sklearn import metrics
import numpy as np
from typing import List
import pickle


class LinearReg:
    def __init__(
        self,
        path: str = "data/regression/weatherHistory.csv",
        n_jobs: int = 10,
        model_path: str = "models/trained.sav",
    ):
        """
        This is initializing the LinearReg classa

        Args:
            path (str): Path to the data file
            n_jobs (int): Number of times to run during training
            model_path (str): Path to the saved model file

        """
        self.df = pd.read_csv(path)
        self.lm = LinearRegression(n_jobs=n_jobs)
        self.model_path = model_path

    def preprocess(
        self,
        factorization_list: List[str] = ["Summary", "Precip Type", "Daily Summary"],
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
    ) -> None:
        """
        Preprocess the data by normalizing a column and factorizing columns

        Returns:
            Dataframe: Preprocessed variable
            Series: Target variable

        """
        self.df = normalize_column(self.df, normalize_column_name)
        self.factorize(factorization_list)

        X = self.df[columns]
        y = self.df[target]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=101
        )

    def factorize(self, columns) -> None:
        """
        Factorize the columns

        Args:
            columns: Column names that need to be factorized

        """
        for c in columns:
            self.df[c] = pd.factorize(self.df[c])[0] + 1

    def train_model(self) -> None:
        """
        Train the model using linear regressionn

        """
        trained = self.lm.fit(self.X_train, self.y_train)
        pickle.dump(trained, open(self.model_path, "wb"))

    def evaluate(self) -> int:
        """
        Evaluating the model using RMSE

        Returns:
        - metric: Root mean squared error.

        """
        metric = np.sqrt(
            metrics.mean_squared_error(self.y_test, self.predict(self.X_test))
        )
        print(f"RMSE: {metric}")
        return metric

    def predict(self, X) -> LinearRegression:
        """
        Make predictions using the model

        Args:
        - X: Input data

        Returns:
        - predictions: Predicted values after using the model

        """
        return self.lm.predict(X)


if __name__ == "__main__":
    lin_reg_model = LinearReg()
    lin_reg_model.preprocess()
    lin_reg_model.train_model()
    evaluated_model = lin_reg_model.evaluate()
