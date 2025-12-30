import pandas as pd
from sklearn.model_selection import train_test_split
from prefect import task
import yaml

@task
def preprocess_data(df: pd.DataFrame, config_path: str = "config.yaml"):
    """
    Preprocess data for training.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Convert target to binary classification (0 = no disease, >0 = disease)
    # The original dataset has 0-4 classes. We treat 0 as negative, 1-4 as positive.
    df["target"] = df["target"].apply(lambda x: 1 if x > 0 else 0)
    
    X = df.drop("target", axis=1)
    y = df["target"]
    
    test_size = config["data"]["test_size"]
    random_state = config["data"]["random_state"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"Data split into train ({X_train.shape}) and test ({X_test.shape}) sets")
    
    return X_train, X_test, y_train, y_test
