from classification.model_types import Types
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


class Factory:
    def __init__(self, model_type: str):
        self.model_type = model_type

    def __call__(self):
        if self.model_type == Types.LOGISTIC_REGRESSION.value:
            return LogisticRegression()
        elif self.model_type == Types.DECISION_TREE.value:
            return DecisionTreeClassifier()
        elif self.model_type == Types.RANDOM_FORESTS.value:
            return RandomForestClassifier()
        elif self.model_type == Types.SUPPORT_VECTOR_MACHINE.value:
            return SVC()
        else:
            raise ValueError("Invalid model type")
