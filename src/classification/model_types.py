from enum import Enum


class Types(Enum):
    LOGISTIC_REGRESSION = "Logistic Regression"
    DECISION_TREE = "Decision Tree"
    RANDOM_FORESTS = "Random Forests"
    SUPPORT_VECTOR_MACHINE = "Support Vector Machine"
    GAUSSIAN_NB = "GaussianNB"
    KNN = "KNeighborsClassifier"
    XGBOOSTCLASS = "XGBoostClassifier"
    XGBOOSTREGRESS = "XGBoostRegressor"
