import logging
from sklearn.linear_model import LogisticRegression

class TrainingStrategy:
    """Train a simple baseline model (good for small datasets)."""
    def train(self, X_train, y_train):
        try:
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)
            return model
        except Exception as e:
            logging.error(f"Training failed: {e}")
            raise
