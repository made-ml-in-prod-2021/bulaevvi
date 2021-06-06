from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
from airflow.sensors.filesystem import FileSensor

from utils import default_args, VOLUME


with DAG(
        "3_predict_pipeline",
        default_args=default_args,
        schedule_interval="@daily",
        start_date=days_ago(5),
) as dag:
    
    # Считываем значение переменной PROD_DIR из Airflow Variables
    prod_dir = "{{ var.value.PROD_DIR }}"
    
    start = DummyOperator(task_id = "Begin")
    
    data_sensor = FileSensor(
        task_id = "Wait_for_data",
        poke_interval = 10,
        retries = 100,
        filepath = "data/raw/{{ ds }}/data.csv"
    )
    
    scaler_sensor = FileSensor(
        task_id = "Wait_for_scaler",
        poke_interval = 10,
        retries = 100,
        filepath = prod_dir + "/scaler.pkl"
    )
    
    model_sensor = FileSensor(
        task_id = "Wait_for_model",
        poke_interval = 10,
        retries = 100,
        filepath = prod_dir + "/model.pkl"
    )
   
    predict = DockerOperator(
        task_id = "Prediction",
        image = "airflow-predict",
        command = "/data/raw/{{ ds }} " + prod_dir + " /data/predictions/{{ ds }}",
        network_mode = "bridge",
        do_xcom_push = False,
        volumes = [VOLUME],
    )
    
    finish = DummyOperator(task_id = "End")

    start >> [data_sensor, scaler_sensor, model_sensor] >> predict >> finish
    
