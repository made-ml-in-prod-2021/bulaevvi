import sys
sys.path.insert(0, "../ml_example")
import pytest
from data import read_data
from enities import read_training_pipeline_params
from model_pipeline import DataProcessingPipeline, Classifier


@pytest.fixture
def init_data():
    data = read_data("../data/raw/heart.csv")
    return data


@pytest.fixture
def train_params():
    params = read_training_pipeline_params("../configs/config_lr.yml")
    return params


@pytest.fixture(scope="session")
def data_processing_model():
    init_data = read_data("../data/raw/heart.csv")
    train_params = read_training_pipeline_params("../configs/config_lr.yml")
    classifier = Classifier(train_params.classifier_params, train_params.model_type)
    pipeline = DataProcessingPipeline(train_params.feature_params.categorical_features,
                                      train_params.feature_params.numerical_features)
    pipeline.fit(init_data)
    transformed_data = pipeline.transform(init_data)
    classifier.fit(transformed_data, init_data['target'].values)
    return pipeline, classifier