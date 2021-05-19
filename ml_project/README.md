# ML in production: Homework 1

Installation:  

    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

### Data
Dataset is [Heart Disease UCI](https://www.kaggle.com/ronitf/heart-disease-uci). Download file and put it to data/raw, then rename to train.csv.


### Testing
We use [pytest](https://docs.pytest.org/) as a framework for testing. You can run tests with the following command:

    pytest tests

To run tests with coverage information use command:

    pytest tests --cov=src -v


### Code linting
One can use [pylint](https://www.pylint.org/) as a framework for linting.

One can run pylint:

    pylint *.py

### Run pipeline

To run end-to-end pipeline one can use command:

    python train_pipeline.py configs/train_config_log_reg.yaml

### Project structure

    ├── README.md          <- The top-level README for developers using this project.
    │
    ├── configs            <- Configuration files.
    │
    ├── data               <- Predictions from the model.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries.
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         and a short `-` delimited description, e.g. `1.0-eda`.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment.
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- code to download or generate data
    │   │
    │   ├── features       <- code to turn raw data into features for modeling
    │   │
    │   ├── model          <- code to train models and then use trained models to make predictions
    │   │
    │   └── data_types     <- configuration dataclasses for type checking
    │
    └── tests              <- Unit tests for project modules.


Python version used: 3.8


### Самооценка


- [X] Назовите ветку homework1 (1 балл)
- [X] положите код в папку ml_project
- [X] В описании к пулл реквесту описаны основные "архитектурные" и тактические решения, которые сделаны в вашей работе. В общем, описание что именно вы сделали и для чего, чтобы вашим ревьюерам было легче понять ваш код. (2 балла)

- [X] Выполнение EDA, закоммитьте ноутбук в папку с ноутбуками (2 баллов)
Вы так же можете построить в ноутбуке прототип(если это вписывается в ваш стиль работы)
Можете использовать не ноутбук, а скрипт, который сгенерит отчет, закоммитьте и скрипт и отчет (за это + 1 балл)

- [X] Проект имеет модульную структуру(не все в одном файле =) ) (2 баллов)

- [X] использованы логгеры (2 балла)

- [X] написаны тесты на отдельные модули и на прогон всего пайплайна(3 баллов)

- [X] Для тестов генерируются синтетические данные, приближенные к реальным (3 баллов)

- [X] Обучение модели конфигурируется с помощью конфигов в json или yaml, закоммитьте как минимум 2 корректные конфигурации, с помощью которых можно обучить модель (разные модели, стратегии split, preprocessing) (3 балла)

- [X] Используются датаклассы для сущностей из конфига, а не голые dict (3 балла) 

- [X] Используйте кастомный трансформер(написанный своими руками) и протестируйте его(3 балла)

- [X] Обучите модель, запишите в readme как это предлагается (3 балла)

- [X] Проведите самооценку, опишите, в какое количество баллов по вашему мнению стоит оценить вашу работу и почему (1 балл доп баллы) 

Получилось баллов: 28