import sys
sys.path.insert(0, "../ml_example")
from data import split_train_test_data
from model_pipeline import DataProcessingPipeline, Classifier, CustomMinMaxScaler


def test_split_train_test_data(init_data, train_params):
    train, test = split_train_test_data(init_data, train_params.split_params)
    assert not train.empty
    assert not test.empty


def test_data_processing_pipeline(init_data, train_params):
    pipeline = DataProcessingPipeline(train_params.feature_params.categorical_features,
                                      train_params.feature_params.numerical_features)
    pipeline.fit(init_data)
    transformed_data = pipeline.transform(init_data)
    assert init_data.shape[0] == transformed_data.shape[0]
    assert init_data.shape[1] <= transformed_data.shape[1]


def test_classifier(init_data, train_params):
    classifier = Classifier(train_params.classifier_params, train_params.model_type)
    pipeline = DataProcessingPipeline(train_params.feature_params.categorical_features,
                                      train_params.feature_params.numerical_features)
    pipeline.fit(init_data)
    transformed_data = pipeline.transform(init_data)
    classifier.fit(transformed_data, init_data['target'].values)
    predicted = classifier.predict(transformed_data)
    assert predicted.shape[0] == init_data.shape[0]


def test_custom_min_max_scaler(init_data, train_params):
    scaler = CustomMinMaxScaler()
    fit_data = init_data.iloc[: init_data.shape[0] // 2]
    transform_data = init_data.iloc[init_data.shape[0] // 2 :]
    scaler.fit(fit_data[train_params.feature_params.numerical_features])
    transformed_data = scaler.transform(transform_data[train_params.feature_params.numerical_features])
    assert all(val < 1 for val in transformed_data.mean(axis=0).values)