import logging
import pandas as pd
from base import Base
import pickle
import nltk
import array
from typing import Tuple
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

logging.basicConfig(level=logging.INFO, format="%(message)s")
LOGGER = logging.getLogger(__name__)


class NaiveBayes(Base):
    """
    A class representing a Naive Bayes classifier.

    Parameters:
    train_path: The path to the training data file.
    labels_path: The path to the labels file.
    model_path: The path to save the trained model.
    """
    def __init__(
            self,
            train_path: str = "../data/text/train.tsv",
            labels_path: str = "../data/text/labels.txt",
            model_path: str = "models/naivebayes.sav",
            bucket_name: str = "ml-basic",
    ):

        super().__init__(model_path, bucket_name)
        self.train_path = train_path
        self.labels_path = labels_path
        self.model = MultinomialNB()
        LOGGER.info("Loading data")
        train_data = pd.read_csv(self.train_path, sep="\t")
        train_data["Phrase"] = train_data["Phrase"].str.lower()
        self.X_train = train_data["Phrase"].str.lower()
        self.y_train = train_data["Sentiment"]
        LOGGER.info("Splitting data into train and test sets")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_train, self.y_train, test_size=0.3, random_state=42)

    def preprocess(self, text: str) -> str:
        """
        Preprocesses the given text (using Lemmmatization and StopWord removal)

        Args:
        text: The text to preprocess

        Returns:
        The preprocessed text

        """
        words = nltk.word_tokenize(text)
        lemmatizer = WordNetLemmatizer()
        lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
        stop_words = set(stopwords.words('english'))
        filtered_words = [word for word in lemmatized_words if word.lower() not in stop_words]
        processed_text = " ".join(filtered_words)
        return processed_text

    def vectorize_data(self) -> None:
        """
        Vectorizes the training and test data

        """
        LOGGER.info("Vectorizing data")
        self.vectorizer = CountVectorizer()
        self.vectorizer.fit(self.X_train)
        self.X_train = self.vectorizer.transform(self.X_train)
        self.X_test = self.vectorizer.transform(self.X_test)

    def train(self) -> None:
        """
        Trains the model

        """
        LOGGER.info("Training the model")
        self.model.fit(self.X_train, self.y_train)
        pickle.dump(self.model, open(self.model_path, "wb"))
        self.upload_model(self.model_path)

    def predict(self) -> array:
        """
        Makes predictions on the test data

        Returns:
        Predictions of the labels

        """
        LOGGER.info("Predicting")
        return self.model.predict(self.X_test)

    def evaluate_model(self) -> Tuple:
        """
        Evaluates the trained model

        Returns:
        A tuple containing the predicted labels and accuracy

        """
        LOGGER.info("Evaluating")
        predictions = self.model.predict(self.X_test)
        accuracy = self.model.score(self.X_test, self.y_test)
        return predictions, accuracy


if __name__ == "__main__":
    nb = NaiveBayes()
    LOGGER.info("Preprocess started")
    nb.X_train = nb.X_train.apply(nb.preprocess)
    LOGGER.info("Preprocess ended")
    nb.vectorize_data()
    nb.train()
    f1, predictions, accuracy = nb.evaluate_model()
    LOGGER.info("Predictions:")
    LOGGER.info(predictions)
    LOGGER.info("Accuracy:")
    LOGGER.info(accuracy)

