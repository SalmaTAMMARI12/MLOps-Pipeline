from zenml import pipeline
import mlflow
from datetime import datetime

from steps.ingest_data import ingest_data
from steps.clean_data import clean_data
from steps.feature_engineer import feature_engineer
from steps.split_and_scale import split_and_scale
from steps.train_model import train_model
from steps.evaluate_model import evaluate_model

from zenml.config import DockerSettings

# Configuration Docker pour ZenML
docker_settings = DockerSettings(
    required_integrations=["mlflow", "sklearn"], 
    requirements=["pandas", "numpy"]             
)

@pipeline(settings={"docker": docker_settings}, enable_cache=False)
def training_pipeline(data_path: str, model_type: str = "logistic"):
    mlflow.set_experiment("first-mlflow-experiment")
    mlflow.set_tag("mlflow.runName", f"run-{model_type}-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    mlflow.log_param("data_path", data_path)
    mlflow.log_param("selected_model", model_type)
    df = ingest_data(data_path)
    df, encoders = clean_data(df)
    df = feature_engineer(df)
    X_train, X_test, y_train, y_test, scaler = split_and_scale(df)
    model = train_model(X_train, y_train, model_type=model_type)
    _ = evaluate_model(model, X_test, y_test)  

    return None