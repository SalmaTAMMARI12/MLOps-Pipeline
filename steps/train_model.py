from src.training_model import LogisticRegressionStrategy, RandomForestStrategy 
import pandas as pd
from zenml import step
from typing import Annotated # <--- AJOUT
from sklearn.base import ClassifierMixin # Pour le typage
import mlflow
import mlflow.sklearn

@step
def train_model(
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    model_type: str = "logistic"
) -> Annotated[ClassifierMixin, "model"]: # <--- FORCE LE NOM ICI
    
    # 1. Choix de la stratégie
    if model_type == "logistic":
        strategy = LogisticRegressionStrategy()
    elif model_type == "random_forest":
        strategy = RandomForestStrategy()
    else:
        raise ValueError(f"Modèle {model_type} non supporté")

    # 2. Entraînement
    model = strategy.train(X_train, y_train)
    
    # 3. Logging MLflow
    mlflow.log_param("model_type", model_type)
    
    mlflow.sklearn.log_model(
        sk_model=model, 
        artifact_path="model",
        registered_model_name="MyBestModel" 
    )
    
    return model