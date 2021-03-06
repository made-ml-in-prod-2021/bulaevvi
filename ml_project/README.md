# ДЗ № 1 "Машинное обучение в продакшене", MADE, весна 2021.

## Установка

Создать виртуальное окружение:
```
conda create -n $env_name python=3.8
conda activate $env_name
```
Установить необходимые пакеты
```
pip install -r requirements.txt
```
## Краткое описание проекта

Исходные данные для построения модели https://www.kaggle.com/ronitf/heart-disease-uci <br>

Для предобработки данных используется:<br>
1) Кастомный MinMaxScaler для числовых признаков
2) OneHotEncoding для категориальных признаков 

Непосредственно для классификации используются две модели:
1) Baseline на основе Logistic Regression
2) Random Forest Classifier

## Описание команд

### Построение отчета по исходным данным:
```
python reports/make_report.py
```
### Обучение моделей

Обучение логистической регрессии:
```
python ml_example/train_pipeline.py configs/config_lr.yml train
```
Обучение случайного леса:
```
python ml_example/train_pipeline.py configs/config_rf.yml train
```
Обученные модели сохраняются в папке models

### Предсказания моделей

Для предсказания используется входной файл data/input/data_to_predict.csv

Предсказание с помощью обученной логистической регрессии:
```
python ml_example/train_pipeline.py configs/config_lr.yml predict
```
Предсказание с помощью обученного случайного леса:
```
python ml_example/train_pipeline.py configs/config_rf.yml predict
```
Предсказания записываются в data/pred/predicts.csv

Тестирование проекта осуществляется с помощью команд:
```
cd ml_example
pytest ../tests --cov
```

Пакетный последовательный запуск всех команд (для удобства):
```
commands.bat
```

## Project Organization

    ├── configs                <- Configuration files.
    │
    ├── data
    │   ├── input              <- Input data for predictions (data_to_predict.csv).
    │   ├── pred               <- Model predictions (predicts.csv).
    │   └── raw                <- The original, immutable data dump (heart.csv).
    │
    ├── ml_example             <- Source code for use in this project.
    │   ├── data               <- Code to generate data.
    │   ├── entities           <- Configuration dataclasses for type checking.
    │   ├── model_pipiline     <- Code to train models and then use trained models to make predictions.
    │   │
    │   └── train_pipeline.py  <- Train pipeline CLI.
    │
    ├── models                 <- Trained and serialized models.
    │
    ├── notebooks              <- Jupyter notebooks (EDA.ipynb).
    │
    ├── reports                <- Script for generation EDA report and resulting report-file (report.html).
    │
    ├── tests                  <- Unit tests for project modules and e2e tests.
    │
    ├── commands.bat           <- Sequence of all commands for fast and convenient testing.
    │
    ├── README.md              <- The top-level README for developers using this project.
    │
    ├── requirements.txt       <- The requirements file for reproducing the analysis environment.
    │
    └── setup.py               <- Setup file.

## Самооценка

+ Назовите ветку homework1 (1 балл) 
+ Положите код в папку ml_project
+ В описании к пулл-реквесту описаны основные "архитектурные" и тактические решения, которые сделаны в вашей работе. В общем, описание что именно вы сделали и для чего, чтобы вашим ревьюерам было легче понять ваш код. (2 балла)
+ Выполнение EDA, закоммитьте ноутбук в папку с ноутбуками (2 балла)
+ Вы так же можете построить в ноутбуке прототип (если это вписывается в ваш тиль работы). Можете использовать не ноутбук, а скрипт, который сгенерит отчет, закоммитьте и скрипт и отчет (1 балл)
+ Проект имеет модульную структуру (2 балла)
+ Использованы логгеры (2 балла)
+ Написаны тесты на отдельные модули и на прогон всего пайплайна (3 балла)
+ Для тестов генерируются синтетические данные, приближенные к реальным (3 балла)
+ Обучение модели конфигурируется с помощью конфигов в json или yaml, закоммитьте как минимум 2 корректные конфигурации, с помощью которых можно обучить модель (разные модели, стратегии split, preprocessing) (3 балла)
+ Используются датаклассы для сущностей из конфига, а не голые dict (3 балла)
+ Используйте кастомный трансформер (написанный своими руками) и протестируйте его (3 балла)
+ Обучите модель, запишите в readme как это предполагается (3 балла)
+ Напишите функцию predict, которая примет на вход артефакт/ы от обучения, тестовую выборку (без меток) и запишет предикт, напишите в readme как это сделать (3 балла)
+ Проведите самооценку, опишите, в какое колво баллов по вашему мнению стоит оценить вашу работу и почему (1 доп. балл)
+ Итого: 32 балла
