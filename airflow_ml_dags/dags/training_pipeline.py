from datetime import timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago

default_args = {
    "owner": "airflow",
    "email": ["airflow@example.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
        dag_id="Training_pipeline",
        default_args=default_args,
        schedule_interval="@weekly",
        start_date=days_ago(7),
) as dag:
    preprocess = DockerOperator(
        image="airflow-preprocess",
        command="--input-dir /data/raw/{{ ds }} --output-dir /data/processed/{{ ds }}",
        task_id="docker-airflow-preprocess",
        do_xcom_push=False,
        # !!! HOST folder(NOT IN CONTAINER) replace with yours !!!
        volumes=["/home/agysar/made_2/ml_in_prod/airflow_ml_dags/data:/data"]
        # volumes=["/tmp:/data"]
    )

    split_data = DockerOperator(
        image="airflow-split-train-val",
        command="--input-dir /data/processed/{{ ds }} --train-data-dir /data/train_data/{{ ds }} "
                "--val-data-dir /data/val_data/{{ ds }}",
        task_id="docker-airflow-split-train-val",
        do_xcom_push=False,
        # !!! HOST folder(NOT IN CONTAINER) replace with yours !!!
        # volumes=["/tmp:/data"]
        volumes=["/home/agysar/made_2/ml_in_prod/airflow_ml_dags/data:/data"]
    )

    train = DockerOperator(
        image="airflow-train",
        command="--input-dir /data/train_data/{{ ds }} --output-dir /data/models/{{ ds }}",
        task_id="docker-airflow-train",
        do_xcom_push=False,
        # !!! HOST folder(NOT IN CONTAINER) replace with yours !!!
        # volumes=["/tmp:/data"]
        volumes=["/home/agysar/made_2/ml_in_prod/airflow_ml_dags/data:/data"]
    )

    evaluate = DockerOperator(
        image="airflow-evaluate",
        command="--path-to-model /data/models/{{ ds }} --path-to-val-data /data/val_data/{{ ds }}"
                " --output-dir /data/metrics/{{ ds }}",
        task_id="docker-airflow-evaluate",
        do_xcom_push=False,
        # !!! HOST folder(NOT IN CONTAINER) replace with yours !!!
        # volumes=["/tmp:/data"]
        volumes=["/home/agysar/made_2/ml_in_prod/airflow_ml_dags/data:/data"]
    )

    preprocess >> split_data >> train >> evaluate
