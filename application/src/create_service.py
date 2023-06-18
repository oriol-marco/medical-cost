import bentoml
import numpy as np
import pandas as pd
from bentoml.io import JSON, NumpyNdarray
from hydra import compose, initialize
from patsy import dmatrix
from pydantic import BaseModel

with initialize(version_base=None, config_path="../../config"):
    config = compose(config_name="main")
    FEATURES = config.process.features
    MODEL_NAME = config.model.name


class Patient(BaseModel):
    age: int = 25
    sex: str = "female"
    bmi: float = 27.9
    children: int = 0
    smoker: str = "yes"
    region: str = "southwest"


def add_dummy_data(df: pd.DataFrame):
    """Add dummy rows so that patsy can create features similar to the train dataset"""
    rows = {
        "age": [25, 35, 51],
        "sex": ["female", "male", "female"],
        "bmi": [33.9, 28.2, 21.8],
        "children": [0, 3, 2],
        "smoker": ["yes", "no", "no"],
        "region": ["southwest", "southeast", "northeast"],
    }

    rows1 = {
        "age": 25,
        "sex": "female",
        "bmi": 33.9,
        "children": 0,
        "smoker": "yes",
        "region": "southwest",
    }

    dummy_df = pd.DataFrame(rows)
    return pd.concat([df, dummy_df])


"""
def rename_columns(X: pd.DataFrame):
    X.columns = X.columns.str.replace("[", "_", regex=True).str.replace(
        "]", "", regex=True
    )
    return X
"""


def transform_data(df: pd.DataFrame):
    """Transform the data"""
    dummy_df = add_dummy_data(df)
    feature_str = " + ".join(FEATURES)
    dummy_X = dmatrix(f"{feature_str} - 1", dummy_df, return_type="dataframe")
    # dummy_X = rename_columns(dummy_X)
    return dummy_X.iloc[0, :].values.reshape(1, -1)


model = bentoml.xgboost.get("xgboost:latest").to_runner()
# model = bentoml.xgboost.load_runner(f"{MODEL_NAME}:latest")
# Create service with the model
service = bentoml.Service("medical_cost", runners=[model])


@service.api(input=JSON(pydantic_model=Patient), output=NumpyNdarray())
def predict(patient: Patient) -> np.ndarray:
    """Transform the data then make predictions"""
    df = pd.DataFrame(patient.dict(), index=[0])
    df = transform_data(df)
    result = model.run(df)[0]
    return np.array(result)
