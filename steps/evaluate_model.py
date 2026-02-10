from typing import Dict, Annotated, Tuple, List
import pandas as pd
from zenml import step
import mlflow
import json
import tempfile

from src.evaluation_model import EvaluationStrategy


@step
def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[
    Annotated[Dict[str, float], "Model metrics"],
    Annotated[List[List[int]], "Confusion matrix"],
    Annotated[str, "Classification report"],
]:
    """
    Evaluate model using EvaluationStrategy (accuracy, f1, confusion matrix, report)
    and log everything to MLflow.
    """
    try:
        results = EvaluationStrategy().evaluate(model, X_test, y_test)
        metrics: Dict[str, float] = {
            "accuracy": float(results["accuracy"]),
            "f1": float(results["f1"]),
        }
        for name, score in metrics.items():
            mlflow.log_metric(name, score)

        confusion_matrix = results["confusion_matrix"]  
        report = results["report"] 
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump(confusion_matrix, f)
            cm_path = f.name
        mlflow.log_artifact(cm_path, artifact_path="evaluation")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write(report)
            rep_path = f.name
        mlflow.log_artifact(rep_path, artifact_path="evaluation")
        return metrics, confusion_matrix, report

    except Exception as e:
        logging.error(f"Failed to evaluate model: {str(e)}")
        raise
