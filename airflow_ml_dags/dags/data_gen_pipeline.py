import datetime
from datetime import timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator

default_args = {
    "owner": "airflow",
    "email": ["airflow@example.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
        dag_id="Generate_wine_data_from_sklearn.datasets",
        default_args=default_args,
        schedule_interval="*/1 * * * *",
        start_date=datetime.datetime.now(),
) as dag:
    download = DockerOperator(
        image="airflow-download",
        command="/data/raw/{{ ds }}",
        network_mode="bridge",
        task_id="docker-airflow-download-wine-data",
        do_xcom_push=False,
        # !!! HOST folder(NOT IN CONTAINER) replace with yours !!!
        # volumes=["/tmp:/data"]
        volumes=["/home/agysar/made_2/ml_in_prod/airflow_ml_dags/data:/data"]
    )

    download
