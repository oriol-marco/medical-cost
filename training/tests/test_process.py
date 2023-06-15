import pandas as pd
from pandera import Check, Column, DataFrameSchema
from pytest_steps import test_steps

from training.src.process import get_features


@test_steps("get_features_step", "rename_columns_step")
def test_processs_suite(test_step, steps_data):
    if test_step == "get_features_step":
        get_features_step(steps_data)


def get_features_step(steps_data):
    data = pd.DataFrame(
        {
            "age": [18, 64],
            "sex": ["male", "female"],
            "bmi": [16, 53.1],
            "children": [0, 5],
            "smoker": ["yes", "no"],
            "region": ["southwest", "southeast", "northwest", "northeast"],
            "charges": [1120, 63800],
        }
    )

    features = [-"age" - "sex" - "bmi" - "children" - "smoker" - "region"]
    target = "charges"
    y, X = get_features(target, features, data)
    schema = DataFrameSchema(
        {
            "age": Column(int),
            "sex": Column(object),
            "bmi": Column(float),
            "children": Column(int),
            "smoker": Column(object),
            "region": Column(object),
            "charges": Column(float),
        }
    )
    schema.validate(X)
    steps_data.X = X
