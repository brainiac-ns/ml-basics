import array
import logging
import pickle
from typing import List

from base import Base
from dotenv import load_dotenv
from sklearn.metrics import f1_score, r2_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from classification.model_factory import Factory
from classification.model_types import Types

logging.basicConfig(level=logging.INFO, format="%(message)s")
LOGGER = logging.getLogger(__name__)

class ClassificationTask(Base):
    def __init__(
        self,
        train_path: str = "data/log_reg/fashion-mnist_train.csv",
        model_path: str = "models/logisticreg.sav",
        test_path: str = "data/log_reg/fashion-mnist_test.csv",
        bucket_name: str = "ml-basic",
        model_type: Types = Types.XGBOOSTREGRESS.value,
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
        model_name = {
            Types.LOGISTIC_REGRESSION.value: "logistic_regression",
            Types.DECISION_TREE.value: "decision_tree",
            Types.RANDOM_FORESTS.value: "random_forests",
            Types.SUPPORT_VECTOR_MACHINE.value: "support_vector_machine",
            Types.GAUSSIAN_NB.value: "gaussian_nb",
            Types.KNN.value: "kneighbors_classifier",
            Types.XGBOOSTCLASS.value: "xgboost_classifier",
            Types.XGBOOSTREGRESS.value: "xgboost_regressor"
        }.get(model_type, "model_type")
        self.model_path = f"models/{model_name}.sav"
        self.model_type = model_type

    def train(self) -> None:
        """
        Train the model using logistic regression
        """
        LOGGER.info("Training started")
        generic_model = self.model.fit(self.X_train, self.y_train)
        pickle.dump(generic_model, open(self.model_path, "wb"))
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
        Evaluate the performance of a classification model by generating a classification report.

        Returns:
        array: A textual summary of the classification performance(precision, recall, F1-score, and support)
        """
        LOGGER.info("Evaluation ended")
        pred = self.model.predict(self.X_test)
        print(self.model_type)
        return r2_score(self.y_test, pred) if (self.model_type == Types.XGBOOSTREGRESS.value) else f1_score(self.y_test, pred, average="micro")

    def preprocess(
        self,
        factorization_list: List[str],
        columns: List[str],
        target: List[str],
        normalize_column_name: str,
        n_components: int = 2
    ) -> None:
        pass


if __name__ == "__main__":
    load_dotenv()
    LOGGER.info("AAAAAAAAAAAAAAAAAAAAA")
    generic_model = ClassificationTask(
        bucket_name="ml-basic"
    )
    generic_model.train()
    evaluated = generic_model.evaluate()
    print(evaluated)
    print("---------------------")
    scaler = StandardScaler()
    scaled_train_data = scaler.fit_transform(generic_model.X_train)
    scaled_test_data = scaler.transform(generic_model.X_test)

    pca = PCA(n_components=2)
    pca.fit(scaled_train_data)
    x_train_pca = pca.transform(scaled_train_data)
    x_test_pca = pca.transform(scaled_test_data)

    print(scaled_train_data.shape)
    print(x_train_pca.shape)