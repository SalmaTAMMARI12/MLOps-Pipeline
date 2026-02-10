import logging
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

class EvaluationStrategy:
    """Evaluate classification model."""
    def evaluate(self, model, X_test, y_test):
        try:
            preds = model.predict(X_test)

            results = {
                "accuracy": float(accuracy_score(y_test, preds)),
                "f1": float(f1_score(y_test, preds)),
                "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
                "report": classification_report(y_test, preds),
            }
            return results
        except Exception as e:
            logging.error(f"Evaluation failed: {e}")
            raise
