import logging
import os
import shutil
import unittest
from unittest.mock import patch

import pandas as pd
from naive_bayes.nb_model import NaiveBayes

logging.basicConfig(level=logging.INFO, format="%(message)s")
LOGGER = logging.getLogger(__name__)


class TestNaiveBayes(unittest.TestCase):
    def setUp(self):
        os.mkdir("test-models/")

    def tearDown(self):
        shutil.rmtree("test-models")

    @patch("pandas.read_csv")
    def test_preprocess(self, mock_reading):
        data = {
            "Phrase": ["I love it", "It's terrible", "Great movie", "Bad experience"],
            "Sentiment": [1, 0, 1, 0],
        }
        df = pd.DataFrame(data)
        mock_reading.return_value = df

        nb_model = NaiveBayes(bucket_name="")
        preprocessed_text = nb_model.preprocess("I love it")
        expected_text = "love"
        self.assertEqual(preprocessed_text, expected_text)

    @patch("pandas.read_csv")
    def test_vectorize_data(self, mock_reading):
        data = {
            "Phrase": ["I love it", "It's terrible", "Great movie", "Bad experience"],
            "Sentiment": [1, 0, 1, 0],
        }
        df = pd.DataFrame(data)
        mock_reading.return_value = df

        nb_model = NaiveBayes(bucket_name="")
        nb_model.X_train = df["Phrase"]
        nb_model.vectorize_data()

        self.assertIsNotNone(nb_model.vectorizer)
        self.assertIsNotNone(nb_model.X_train)

    @patch("boto3.Session.client")
    @patch("pandas.read_csv")
    def test_train(self, mock_reading, mock_client):
        data = {
            "Phrase": ["I love it", "It's terrible", "Great movie", "Bad experience"],
            "Sentiment": [1, 0, 1, 0],
        }
        df = pd.DataFrame(data)
        mock_reading.return_value = df
        mock_client.upload_file.return_value = None
        nb_model = NaiveBayes(model_path="models/naivebayes.sav", bucket_name="")
        nb_model.X_train = df["Phrase"]
        nb_model.y_train = df["Sentiment"]
        nb_model.train()
        self.assertEqual(os.listdir("test-models/")[0], "naivebayes.sav")

    @patch("pandas.read_csv")
    def test_predict(self, mock_reading):
        data = {
            "Phrase": ["I love it", "It's terrible", "Great movie", "Bad experience"],
            "Sentiment": [1, 0, 1, 0],
        }
        df = pd.DataFrame(data)
        mock_reading.return_value = df

        nb_model = NaiveBayes(model_path="models/naivebayes.sav", bucket_name="")
        nb_model.X_test = df["Phrase"]
        nb_model.y_test = df["Sentiment"]
        nb_model.train()
        predictions = nb_model.predict()

        self.assertIsNotNone(predictions)

    @patch("pandas.read_csv")
    def test_evaluate_model(self, mock_reading):
        data = {
            "Phrase": ["I love it", "It's terrible", "Great movie", "Bad experience"],
            "Sentiment": [1, 0, 1, 0],
        }
        df = pd.DataFrame(data)
        mock_reading.return_value = df

        nb_model = NaiveBayes(model_path="models/naivebayes.sav", bucket_name="")
        nb_model.X_train = df["Phrase"]
        nb_model.y_train = df["Sentiment"]
        nb_model.X_test = df["Phrase"]
        nb_model.y_test = df["Sentiment"]

        nb_model.X_train = nb_model.X_train.apply(nb_model.preprocess)
        nb_model.X_test = nb_model.X_test.apply(nb_model.preprocess)
        nb_model.vectorize_data()
        nb_model.train()
        accuracy = nb_model.evaluate_model()

        self.assertGreater(accuracy, 0)
