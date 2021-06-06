from datetime import timedelta

VOLUME = 'C:/MADE/ML_in_prod/HW/airflow_ml_dags/data/:/data'

default_args = {
    "owner": "Vladimir_Bulaev",
    "email": ["bulaevvi@gmail.com"],
    "email_on_failure": True, # Alert в случае падения дага
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


