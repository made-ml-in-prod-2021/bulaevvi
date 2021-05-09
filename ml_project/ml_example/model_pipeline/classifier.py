import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib


class Classifier:
    """
    Assign classifier, Logistic Regression or Random Forest Classifier
    """

    def __init__(self, params, model_type: str):
        self.model_type = model_type
        if model_type == "Logistic Regression":
            self.model = LogisticRegression(C=params.C, penalty=params.penalty, fit_intercept=params.fit_intercept,
                                            random_state=params.random_state)
        if model_type == "Random Forest Classifier":
            self.model = RandomForestClassifier(n_estimators=params.n_estimators, max_depth=params.max_depth,
                                            random_state=params.random_state)

    def fit(self, X: np.array, y: np.array) -> None:
        self.model.fit(X, y)

    def predict(self, X: np.array) -> np.array:
        if X.shape[0] > 0:
            return self.model.predict(X)
        return np.array([])

    def dump_model(self, path: str):
        joblib.dump(self.model, path)


def get_classification_report(y_true: np.array, y_pred: np.array):
    return classification_report(y_true, y_pred)