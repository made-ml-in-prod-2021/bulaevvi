import os
import pickle
from typing import List, Union, Optional

import pandas as pd
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier


# При загрузке моделей с кастомными трансформерами FastAPI выдает ошибку
# Для фикса этой ошибки пришлось явно прописать используемые функции
cat_features = ['sex', 'cp', 'restecg', 'exang', 'slope', 'ca', 'thal']
num_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
def get_num_features(x):
    return x[num_features]
def get_cat_features(x):
    return x[cat_features]


PATH_TO_MODEL = './model.pkl'


model: Optional[Pipeline] = None


def load_object(path: str) -> Pipeline:
    with open(path, "rb") as f:
        return pickle.load(f)


class InputData(BaseModel):
    id: int
    chol: float
    thalach: int
    oldpeak: float
    trestbps: int
    age: int
    sex: int
    cp: int
    restecg: int
    exang: int
    slope: int
    ca: int
    thal: int


class ModelResponse(BaseModel):
    id: str
    target: int


def make_predict(request: List[InputData]) -> List[ModelResponse]:
    # Составляем DataFrame из данных запроса
    df = pd.DataFrame(columns = InputData.__fields__.keys())
    for row in request:
        df = df.append(row.__dict__, ignore_index=True)
    # Рассчитываем предсказания в столбец 'target'
    df['target'] = model.predict(df)
    # Формируем ответ
    response_list = []
    for row in df.loc[:, ['id', 'target']].itertuples():
        response_list.append(ModelResponse(
            id=row.id,
            target=row.target,
        ))
    return response_list


app = FastAPI()


@app.on_event("startup")
def load_model():
    global model
    model = load_object(PATH_TO_MODEL)


@app.get("/")
def main():
    return "This is an entry point of our predictor"


@app.get("/health")
def health() -> bool:
    return not (model is None)


@app.get("/predict/", response_model=List[ModelResponse])
def predict(request: List[InputData]):
    return make_predict(request)


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=os.getenv("PORT", 8000))
