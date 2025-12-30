from prefect import flow
from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.train import train_model

@flow(name="Heart Disease Prediction Pipeline")
def training_pipeline(config_path: str = "config.yaml"):
    df = load_data(config_path)
    X_train, X_test, y_train, y_test = preprocess_data(df, config_path)
    model = train_model(X_train, X_test, y_train, y_test, config_path)
    return model

if __name__ == "__main__":
    training_pipeline()
