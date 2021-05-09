from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List
import pandas as pd
import numpy as np
from cloudpickle import dump


class CustomMinMaxScaler(BaseEstimator, TransformerMixin):
    """
    Custom Min Max Scaler
    """

    def __init__(self, feature_range=(0, 1), copy=True):
        self.feature_range = feature_range
        self.copy = copy

    def fit(self, X, y=None):
        """
        Fit transformer
        y = None - for sklearn consistency
        """
        feature_range = self.feature_range
        data_min = np.min(X, axis=0)
        data_range = np.max(X, axis=0) - data_min
        self.scale_ = (feature_range[1] - feature_range[0]) / data_range
        self.min_ = feature_range[0] - data_min * self.scale_
        return self

    def transform(self, X):
        """Tranform data from fitted scale and min"""
        X *= self.scale_
        X += self.min_
        return X


class DataProcessingPipeline:
    """
    Data processing pipeline to separately process categorical and numerical feature and concatenate it in the end
    """

    def __init__(self, categorical_features: List[str], numerical_features: List[str]):
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.pipeline = None

    @staticmethod
    def _build_categorical_pipeline(categorical_features) -> Pipeline:
        categorical_pipeline = Pipeline(
            [
                ('selection', FunctionTransformer(lambda x: x[categorical_features], validate=False)),
                ("ohe", OneHotEncoder(sparse=False)),
            ]
        )
        return categorical_pipeline

    @staticmethod
    def _build_numerical_pipeline(numerical_features) -> Pipeline:
        numerical_pipeline = Pipeline(
            [
                ('selection', FunctionTransformer(lambda x: x[numerical_features], validate=False)),
                ("scaler", CustomMinMaxScaler()),
            ]
        )
        return numerical_pipeline

    def fit(self, data: pd.DataFrame) -> None:
        self.pipeline = Pipeline(steps=[
            ('feature_processing', FeatureUnion(
                transformer_list=[
                    ('numeric_processing', self._build_numerical_pipeline(self.numerical_features)),
                    ('category_processing', self._build_categorical_pipeline(self.categorical_features)),
                ]
            ))
        ])
        self.pipeline.fit(data)

    def transform(self, data: pd.DataFrame) -> np.array:
        if not data.empty:
            return self.pipeline.transform(data)
        return np.array([])

    def dump_preprocessor(self, path: str) -> None:
        with open(path, 'wb') as f:
            dump(self.pipeline, f)

