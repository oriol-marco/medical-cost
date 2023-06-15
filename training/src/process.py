import hydra
import pandas as pd
from hydra.utils import to_absolute_path as abspath
from omegaconf import DictConfig
from patsy import dmatrices
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def get_data(raw_path: str):
    data = pd.read_csv(raw_path)
    return data


def drop_duplicates(data: pd.DataFrame):
    data = data.drop_duplicates()
    return data


def get_features(target: str, features: list, data: pd.DataFrame):
    X = data[features]
    y = data[target]
    return y, X


def label_encoder(data: pd.DataFrame, cat_features: list):
    le = LabelEncoder()
    for feature in cat_features:
        data[feature] = le.fit_transform(data[feature])
    return data


"""
def standardize(data: pd.DataFrame, num_features: list):
    scaler = StandardScaler()
    for feature in num_features:
        data[feature] = scaler.fit_transform(data[feature])
    return data
"""

@hydra.main(version_base=None, config_path="../../config", config_name="main") 
def process_data(config: DictConfig):
    """Function to process the data"""

    data = get_data(abspath(config.raw.path))
    
    data = drop_duplicates(data)

    y, X = get_features(config.process.target, config.process.features, data)

    X = label_encoder(X, config.process.cat_features)
    
    # X = standardize(X, config.process.num_features)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Save data
    X_train.to_csv(abspath(config.processed.X_train.path), index=False)
    X_test.to_csv(abspath(config.processed.X_test.path), index=False)
    y_train.to_csv(abspath(config.processed.y_train.path), index=False)
    y_test.to_csv(abspath(config.processed.y_test.path), index=False)


if __name__ == "__main__":
    process_data()
