import sys

sys.path.insert(0, "../ml_example")

from hypothesis import given, strategies
from hypothesis.extra.pandas import data_frames, column


@given(data_frames([
    column('sex', dtype=int, elements=strategies.integers(min_value=0, max_value=1)),
    column('age', dtype=int, elements=strategies.integers(min_value=18, max_value=100)),
    column('cp', dtype=int, elements=strategies.integers(min_value=0, max_value=3)),
    column('restecg', dtype=int, elements=strategies.integers(min_value=0, max_value=2)),
    column('exang', dtype=int, elements=strategies.integers(min_value=0, max_value=1)),
    column('slope', dtype=int, elements=strategies.integers(min_value=0, max_value=2)),
    column('chol', dtype=float, elements=strategies.floats(min_value=100, max_value=900)),
    column('thalach', dtype=int, elements=strategies.integers(min_value=30, max_value=250)),
    column('oldpeak', dtype=float, elements=strategies.floats(min_value=0, max_value=7)),
    column('trestbps', dtype=int, elements=strategies.integers(min_value=80, max_value=250)),
    column('fbs', dtype=int, elements=strategies.integers(min_value=0, max_value=1)),
    column('ca', dtype=int, elements=strategies.integers(min_value=0, max_value=4)),
    column('thal', dtype=int, elements=strategies.integers(min_value=1, max_value=3)),
]))
def test_whole_pipeline(data_processing_model, data):
    data_processor, classifier = data_processing_model
    transformed_data = data_processor.transform(data)
    classifier.predict(transformed_data)
    assert True
    
