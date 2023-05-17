import array
import logging
import pickle

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score

LOGGER = logging.getLogger(__name__)


class LogisticReg:
    def __init__(
        self,
        train_path: str = "data/log_reg/fashion-mnist_train.csv",
        model_path: str = "models/logisticreg.sav",
        test_path: str = "data/log_reg/fashion-mnist_test.csv",
    ):
        """
        Initializing the LogisticReg class.

        Args:
            train_path (str): Path to the trianing data.
            model_path (str): Path to the saved model file.
            test_path(str): Path to the test data.
        """
        self.df = pd.read_csv(train_path)
        self.lm = LogisticRegression()
        self.model_path = model_path
        self.test_path = pd.read_csv(test_path)
        self.y_train = self.df["label"].values
        self.X_train = self.df.drop("label", axis=1).values
        self.y_test = self.test_path["label"].values
        self.X_test = self.test_path.drop("label", axis=1).values

    def train(self) -> None:
        """
        Train the model using logistic regression
        """
        LOGGER.info("Training started")
        logisticreg = self.lm.fit(self.X_train, self.y_train)
        pickle.dump(logisticreg, open(self.model_path, "wb"))
        LOGGER.info("Training ended")

    def predict(self, X_test) -> array:
        """
        Make predictions using the model
        Args:
        - X_test: Input data
        Returns:
        - predictions(array): Predicted values after using the model
        """
        predictions = self.lm.predict(X_test)
        return predictions

    def evaluate(self) -> array:
        """
        Evaluate the performance of a classification model by generating a classification report.

        Returns:
        array: A textual summary of the classification performance(precision, recall, F1-score, and support)
        """
        LOGGER.info("Evaluation ended")
        pred = self.lm.predict(self.X_test)
        print(classification_report(self.y_test, pred))
        return f1_score(self.y_test, pred, average="micro")


if __name__ == "__main__":
    log_reg_model = LogisticReg()
    log_reg_model.train()
    evaluated = log_reg_model.evaluate()
    print(evaluated)
