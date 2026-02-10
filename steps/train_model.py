from src.training_model import TrainingStrategy
from zenml import step
import mlflow
import mlflow.sklearn


@step
def train_model(X_train, y_train):
    model = TrainingStrategy().train(X_train, y_train)
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model"
    )
    mlflow.register_model(
        model_uri=f"runs:/{mlflow.active_run().info.run_id}/model",
        name="MyClassifier"
    )

    return model
