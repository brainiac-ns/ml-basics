import logging
import pickle
from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from sklearn import metrics
from sklearn.model_selection import train_test_split
from utils import normalize_column

logging.basicConfig(level=logging.INFO, format="%(message)s")
LOGGER = logging.getLogger(__name__)


class NeuralNetRegression:
    def __init__(self, path: str = "data/regression/weatherHistory_train.csv"):
        self.df = pd.read_csv(path)

    def preprocess(
        self,
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
        Preprocess the data by normalizing a column

        Args:
            columns (List[str]): List of column names to include in preprocessing
            target (List[str]): List of target column names
            normalize_column_name (str): Name of the column to normalize

        """

        self.df = normalize_column(self.df, normalize_column_name)

        X = self.df[columns].values
        y = self.df[target].values

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=101
        )

    def factorize(self, columns) -> None:
        """
        Factorize the columns

        Args:
            columns: Column names that need to be factorized

        """
        LOGGER.info("Factorizing...")
        for c in columns:
            self.df[c] = pd.factorize(self.df[c])[0] + 1

    def build_model(self) -> None:
        self.model = Sequential(
            [
                Dense(
                    units=15,
                    activation="relu",
                    input_dim=10,
                    kernel_initializer="normal",
                ),
                Dense(
                    units=5,
                    activation="relu",
                    input_dim=10,
                    kernel_initializer="normal",
                ),
                Dense(
                    units=1,
                    activation="linear",
                    input_dim=10,
                    kernel_initializer="normal",
                ),
            ]
        )

        self.model.compile(loss="mean_squared_error", optimizer="adam")

    def train(
        self,
        epochs: int = 100,
        batch_size: int = 20,
    ) -> None:
        LOGGER.info("Training started")
        self.model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batch_size)
        LOGGER.info("Training ended")

    def evaluate(self) -> float:
        """
        Evaluate the model using RMSE

        Returns:
            float: Root Mean Squared Error (RMSE) metric

        """
        LOGGER.info("Evaluation started")
        y_pred = self.model.predict(self.X_test)
        metric = np.sqrt(metrics.mean_squared_error(self.y_test, y_pred))
        mae = metrics.mean_absolute_error(self.y_test, y_pred)
        LOGGER.info(f"RMSE: {metric}")
        LOGGER.info(f"MAE: {mae}")
        LOGGER.info("Evaluation ended")
        num_samples = 500
        step_size = len(self.y_test) // num_samples
        samples = np.arange(0, len(self.y_test), step_size)

        plt.figure(figsize=(10, 6))
        plt.plot(samples, self.y_test[samples], label="Actual Temperature", linewidth=2)
        plt.plot(samples, y_pred[samples], label="Predicted Temperature", linewidth=2)
        plt.xlabel("Samples")
        plt.ylabel("Temperature (in Celsius)")
        plt.title("Actual vs. Predicted Temperature")
        plt.legend()
        plt.grid(True)
        plt.show()
        return metric

    def save_model(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.model, f)
        LOGGER.info("Model saved")


if __name__ == "__main__":
    neural_net = NeuralNetRegression()
    neural_net.factorize(["Summary", "Precip Type", "Daily Summary"])
    LOGGER.info("Preprocessing... (normalization)")
    neural_net.preprocess()
    LOGGER.info("Preprocess finished. Building model...")
    neural_net.build_model()
    LOGGER.info("Model built. ")
    neural_net.train()
    neural_net.evaluate()
    neural_net.save_model("models/ann_regression.sav")
