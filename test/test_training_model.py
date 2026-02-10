import pandas as pd
from src.training_model import TrainingStrategy

def test_training_returns_fitted_model():
    X_train = pd.DataFrame({
        "age": [0.1, -0.3, 1.2, -1.0],
        "study_hours_per_day": [0.2, -0.2, 1.0, -1.0],
        "grades_before_ai": [0.0, 0.5, -0.5, 1.0],
        "grades_after_ai": [0.1, 0.4, -0.4, 0.9],
        "daily_screen_time_hours": [0.0, 1.0, -1.0, 0.3],
        "improvement": [0.2, -0.1, 0.3, -0.2],
        "relative_improvement": [0.1, -0.05, 0.12, -0.08],
        "productivity": [0.2, -0.1, 0.4, -0.3],
        "uses_ai": [0, 1, 0, 1],
        "screen_overload": [0, 1, 0, 0],
    })
    y_train = pd.Series([0, 1, 0, 1])

    model = TrainingStrategy().train(X_train, y_train)

    # Model should be fitted and usable
    assert hasattr(model, "predict")
    preds = model.predict(X_train)
    assert len(preds) == len(X_train)
