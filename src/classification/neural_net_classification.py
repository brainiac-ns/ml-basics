import logging
import pickle
from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(message)s")
LOGGER = logging.getLogger(__name__)


class NeuralNetClassification:
    def __init__(self, path: str = "data/log_reg/fashion-mnist_train.csv"):
        self.df = pd.read_csv(path)

    def preprocess(self):
        LOGGER.info("Starting preprocess...")
        self.y_train = self.df["label"].values
        self.X_train = self.df.drop("label", axis=1).values
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_train, self.y_train, test_size=0.2, random_state=42
        )
        self.num_classes = np.max(self.y_train) + 1
        self.y_train = to_categorical(self.y_train, num_classes=self.num_classes)
        self.y_test = to_categorical(self.y_test, num_classes=self.num_classes)
        LOGGER.info("Preprocess finished.")

    def build_model(self):
        LOGGER.info("Building model...")
        self.model = Sequential()
        self.model.add(Dense(15, activation="relu"))
        self.model.add(Dense(5, activation="relu"))
        self.model.add(Dense(self.num_classes, activation='softmax'))
        self.model.compile(
            optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
        )
        LOGGER.info("Model built.")

    def train_model(self, epochs: int = 40, batch_size: int = 20):
        LOGGER.info("Training started...")
        self.model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batch_size)
        LOGGER.info("Training ended.")

    def predict(self, X):
        predicted_labels = self.model.predict(X)
        return predicted_labels

    def evaluate(self):
        y_pred = np.argmax(self.predict(self.X_test), axis=-1)
        accuracy = accuracy_score(np.argmax(self.y_test, axis=-1), y_pred)
        return accuracy

    def save_model(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.model, f)
        LOGGER.info("Model saved")


if __name__ == "__main__":
    classification_model = NeuralNetClassification()
    classification_model.preprocess()
    classification_model.build_model()
    classification_model.train_model()
    accuracy = classification_model.evaluate()
    classification_model.save_model("models/ann_classification.sav")
    print("Accuracy:", accuracy)
    print("Job Done!")
