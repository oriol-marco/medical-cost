import warnings

warnings.filterwarnings(action="ignore")

import os
import hydra
import joblib
import mlflow
import pandas as pd
from helper import BaseLogger
from hydra.utils import to_absolute_path as abspath
from omegaconf import DictConfig
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

logger = BaseLogger()


def load_data(path: DictConfig):
    X_test = pd.read_csv(abspath(path.X_test.path))
    y_test = pd.read_csv(abspath(path.y_test.path))
    return X_test, y_test


def load_model(model_path: str):
    return joblib.load(model_path)


def predict(model: XGBRegressor, X_test: pd.DataFrame):
    return model.predict(X_test)


def log_params(model: XGBRegressor, features: list):
    logger.log_params({"model_class": type(model).__name__})
    model_params = model.get_params()

    for arg, value in model_params.items():
        logger.log_params({arg: value})

    logger.log_params({"features": features})


def log_metrics(**metrics: dict):
    logger.log_metrics(metrics)


@hydra.main(version_base=None, config_path="../../config", config_name="main")
def evaluate(config: DictConfig):
    # Local enviorment
    mlflow.set_tracking_uri(config.mlflow_tracking_ui)
    #mlflow.set_experiment("medical-costs-prediction")

    # Remote enviorment
    os.environ['MLFLOW_TRACKING_USERNAME'] = config.mlflow_USERNAME
    os.environ['MLFLOW_TRACKING_PASSWORD'] = config.mlflow_PASSWORD
    
    with mlflow.start_run():
        
        # Load data and model
        X_test, y_test = load_data(config.processed)

        model = load_model(abspath(config.model.path))

        # Get predictions
        prediction = predict(model, X_test)

        # Get metrics
        rmse = mean_squared_error(y_test, prediction, squared=False)
        print(f"RMSE of this model is {rmse}.")

        # Log metrics
        log_params(model, config.process.features)
        log_metrics(rmse=rmse)
        
        #mlflow.sklearn.log_model(model, "model")
        #mlflow.log_metric("rmse", rmse)
        
        # Remote enviroment Dagshub
        #mloflow.sklearn.log_model(model, "model")


if __name__ == "__main__":
    evaluate()
