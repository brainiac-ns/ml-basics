import array
import logging
import pickle
import os
from typing import List

from dotenv import load_dotenv
from sklearn.metrics import f1_score

from base import Base
from classification.classification_factory import Factory
from classification.classification_types import Types

logging.basicConfig(level=logging.INFO, format="%(message)s")
LOGGER = logging.getLogger(__name__)


class ClassificationTask(Base):
    def __init__(
        self,
        train_path: str = "data/log_reg/fashion-mnist_train.csv",
        model_path: str = "models/logisticreg.sav",
        test_path: str = "data/log_reg/fashion-mnist_test.csv",
        bucket_name: str = "ml-basic",
        model_type: Types = Types.LOGISTIC_REGRESSION.value,
    ):
        """
        Initializing the Classification class.

        Args:
            train_path (str): Path to the trianing data.
            model_path (str): Path to the saved model file.
            test_path(str): Path to the test data.
            model_type: Type of the machine learning model.
        """
        super().__init__(model_path, bucket_name)
        self.df = self.read_data(train_path)
        factory = Factory(model_type)
        self.model = factory()
        self.test_path = self.read_data(test_path)
        self.y_train = self.df["label"].values
        self.X_train = self.df.drop("label", axis=1).values
        self.y_test = self.test_path["label"].values
        self.X_test = self.test_path.drop("label", axis=1).values
        self.model_path = model_path
        self.model_type = model_type

    def train(self) -> None:
        """
        Train the model using logistic regression
        """
        LOGGER.info("Training started")
        classification_model = self.model.fit(self.X_train, self.y_train)
        os.mkdir("models")
        pickle.dump(classification_model, open(self.model_path, "wb"))
        self.upload_model(self.model_path)
        LOGGER.info("Training ended")

    def predict(self, X_test) -> array:
        """
        Make predictions using the model
        Args:
        - X_test: Input data
        Returns:
        - predictions(array): Predicted values after using the model
        """
        predictions = self.model.predict(X_test)
        return predictions

    def evaluate(self) -> array:
        """
        Evaluate the performance of a classification model by generating
        a classification report.

        Returns:
        array: A textual summary of the classification
        performance(precision, recall, F1-score, and support)
        """
        LOGGER.info("Evaluation ended")
        pred = self.model.predict(self.X_test)
        print(self.model_type)
        return f1_score(self.y_test, pred, average="micro")

    def preprocess(
        self,
        factorization_list: List[str],
        columns: List[str],
        target: List[str],
        normalize_column_name: str,
        n_components: int = 2,
    ) -> None:
        pass


if __name__ == "__main__":
    load_dotenv()
    classification_model = ClassificationTask(
        bucket_name="ml-basic",
        test_path="data/log_reg/fashion-mnist_test.csv",
        train_path="data/log_reg/fashion-mnist_train.csv",
        model_path="models/logisticreg.sav",
    )
    classification_model.train()
    evaluated = classification_model.evaluate()
    print(evaluated)
    with open("metrics.txt", "w") as outfile:
        outfile.write("Accuracy: " + str(evaluated) + "\n")
