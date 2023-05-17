import array
import logging
import os
import pickle
from classification.model_factory import Factory
from classification.model_types import Types
from base import Base
from dotenv import load_dotenv
from typing import List
from sklearn.metrics import classification_report, f1_score

LOGGER = logging.getLogger(__name__)


class ClassificationTask(Base):
    def __init__(
        self,
        train_path: str = "data/log_reg/fashion-mnist_train.csv",
        model_path: str = "models/logisticreg.sav",
        test_path: str = "data/log_reg/fashion-mnist_test.csv",
        bucket_name: str = "ml-basic",
        model_type: Types = Types.SUPPORT_VECTOR_MACHINE,
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

    def train(self) -> None:
        """
        Train the model using logistic regression
        """
        LOGGER.info("Training started")
        generic_model = self.model.fit(self.X_train, self.y_train)
        pickle.dump(generic_model, open(self.model_path, "wb"))
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
        Evaluate the performance of a classification model by generating a classification report.

        Returns:
        array: A textual summary of the classification performance(precision, recall, F1-score, and support)
        """
        LOGGER.info("Evaluation ended")
        pred = self.model.predict(self.X_test)
        print(classification_report(self.y_test, pred))
        return f1_score(self.y_test, pred, average="micro")

    def preprocess(
        self,
        factorization_list: List[str],
        columns: List[str],
        target: List[str],
        normalize_column_name: str,
    ) -> None:
        pass


if __name__ == "__main__":
    load_dotenv()
    generic_model = ClassificationTask(
        bucket_name="", model_type=os.getenv("MODEL_TYPE")
    )
    generic_model.train()
    evaluated = generic_model.evaluate()
    print(evaluated)
