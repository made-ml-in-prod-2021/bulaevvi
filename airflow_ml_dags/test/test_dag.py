import sys
import pytest
from airflow.models import DagBag

sys.path.append("dags")


@pytest.fixture()
def dag_bag():
    return DagBag(dag_folder="dags/", include_examples=False)


def test_dag_bag_import(dag_bag):
    assert dag_bag.dags is not None
    assert dag_bag.import_errors == {}


def test_dag_generate_data_load(dag_bag):
    assert "1_generate_data" in dag_bag.dags
    assert len(dag_bag.dags["1_generate_data"].tasks) == 3


def test_dag_train_pipeline_load(dag_bag):
    assert "2_train_pipeline" in dag_bag.dags
    assert len(dag_bag.dags["2_train_pipeline"].tasks) == 8


def test_dag_predict_pipeline_load(dag_bag):
    assert "3_predict_pipeline" in dag_bag.dags
    assert len(dag_bag.dags["3_predict_pipeline"].tasks) == 6
    
  
def test_dag_generate_data_structure(dag_bag):
    structure = {
        "Begin": ["Generate_data"],
        "Generate_data": ["End"],
        "End": [],
    }
    dag = dag_bag.dags["1_generate_data"]
    for name, task in dag.task_dict.items():
        assert set(structure[name]) == task.downstream_task_ids  
 

def test_dag_train_pipeline_structure(dag_bag):
    structure = {
        "Begin": ["Wait_for_data", "Wait_for_target"],
        "Wait_for_data": ["Data_preprocess"],
        "Wait_for_target": ["Data_preprocess"],
        "Data_preprocess": ["Split_data"],
        "Split_data": ["Train_model"],
        "Train_model": ["Validate_model"],
        "Validate_model": ["End"],
        "End": [],
    }
    dag = dag_bag.dags["2_train_pipeline"]
    for name, task in dag.task_dict.items():
        assert set(structure[name]) == task.downstream_task_ids


def test_dag_predict_pipeline_structure(dag_bag):
    structure = {
        "Begin": ["Wait_for_data", "Wait_for_scaler", "Wait_for_model"],
        "Wait_for_data": ["Prediction"],
        "Wait_for_scaler": ["Prediction"],
        "Wait_for_model": ["Prediction"],
        "Prediction": ["End"],
        "End": [],
    }
    dag = dag_bag.dags["3_predict_pipeline"]
    for name, task in dag.task_dict.items():
        assert set(structure[name]) == task.downstream_task_ids