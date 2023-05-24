import logging
import pandas as pd
from base import Base
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

logging.basicConfig(level=logging.INFO, format="%(message)s")
LOGGER = logging.getLogger(__name__)


class NaiveBayes(Base):
    def __init__(
            self,
            train_path: str = "data/text/train.tsv",
            labels_path: str = "data/text/labels.txt",
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
        self.X_train = train_data["Phrase"]
        self.y_train = train_data["Sentiment"]
        LOGGER.info("Splitting data into train and test sets")
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_train, self.y_train, test_size=0.3, random_state=42
        )
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test


    #def preprocess(self):
        #LOGGER.info("Loading data")
        #train_data = pd.read_csv(self.train_path, sep="\t")

        #train_data["Phrase"] = train_data["Phrase"].str.lower()

        #X = train_data["Phrase"]
        #y = train_data["Sentiment"]
        #LOGGER.info("Splitting data into train and test sets")
        #X_train, X_test, y_train, y_test 
        #train_test_split(
         #   X, y, test_size=0.3, random_state=42
        #)

        #return X_train, X_test, y_train, y_test
    
    def vectorize_data(self):
        LOGGER.info("Vectorizing data")
        self.vectorizer = CountVectorizer()
        self.X_train = self.vectorizer.fit_transform(self.X_train)
        self.X_test = self.vectorizer.transform(self.X_test)

    def train(self):
        LOGGER.info("Training the model")
        nb_model = self.model.fit(self.X_train, self.y_train)
        pickle.dump(nb_model, open(self.model_path, "wb"))
        self.upload_model(self.model_path)

    def predict(self):
        LOGGER.info("Predicting")
        return self.model.predict(self.X_test)

    def evaluate_model(self):
        LOGGER.info("Evaluating")
        predictions = self.predict()
        accuracy = self.model.score(self.X_test, self.y_test)
        return predictions, accuracy


if __name__ == "__main__":
    nb = NaiveBayes()
    #nb.preprocess()
    nb.vectorize_data()
    nb.train()
    predictions, accuracy = nb.evaluate_model()
    LOGGER.info("Predictions:")
    LOGGER.info(predictions)
    LOGGER.info("Accuracy:")
    LOGGER.info(accuracy)
