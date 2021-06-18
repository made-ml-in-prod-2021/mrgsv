import datetime
import os
from datetime import timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sensors.python import PythonSensor
from airflow.models import Variable

default_args = {
    "owner": "airflow",
    'email_on_failure': True,
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}


def _wait_for_data():
    return os.path.exists(os.path.join(os.getcwd(), "data/raw/2021-06-18/data.csv")) \
        and os.path.exists(os.path.join(os.getcwd(), "data/raw/2021-06-18/target.csv"))


def _wait_for_model():
    return os.path.exists(os.path.join(os.getcwd(), Variable.get("PATH_TO_MODEL")))


with DAG(
        dag_id="Predict",
        default_args=default_args,
        schedule_interval="@daily",
        start_date=datetime.datetime.now(),
) as dag:
    wait_for_data = PythonSensor(
        task_id="wait_for_data",
        python_callable=_wait_for_data,
        timeout=6000,
        poke_interval=10,
        retries=10,
        mode="poke",
    )

    wait_for_model = PythonSensor(
        task_id="wait_for_model",
        python_callable=_wait_for_model,
        timeout=6000,
        poke_interval=10,
        retries=10,
        mode="poke",
    )

    predict = DockerOperator(
        image="airflow-predict",
        command="--input-dir /data/raw/{{ ds }} --output-dir /data/predictions/{{ ds }} "
                "--path-to-model /data/models/{{ ds }}/random_forest_classifier.pickle",
        task_id="docker-airflow-predict",
        do_xcom_push=False,
        # !!! HOST folder(NOT IN CONTAINER) replace with yours !!!
        # volumes=["/tmp:/data"]
        volumes=["/home/agysar/made_2/ml_in_prod/airflow_ml_dags/data:/data"]
    )

    [wait_for_data, wait_for_model] >> predict
