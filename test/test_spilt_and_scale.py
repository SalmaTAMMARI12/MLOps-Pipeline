import pandas as pd
import numpy as np
from src.data_splitting_scaling import TrainTestSplitWithScaling

def test_split_and_scale_returns_expected_shapes_and_no_target_in_X():
    df = pd.DataFrame({
        "age": [19, 18, 20, 17, 21, 22],  # 6 lignes au lieu de 5
        "study_hours_per_day": [2.0, 4.0, 1.0, 3.0, 2.5, 3.5],
        "grades_before_ai": [60, 50, 70, 40, 65, 45],
        "grades_after_ai": [61, 70, 71, 55, 66, 60],
        "daily_screen_time_hours": [3, 7, 2, 5, 4, 6],
        "uses_ai": [0, 1, 0, 1, 0, 1],
        "screen_overload": [0, 1, 0, 0, 0, 1],
        "improvement": [1, 20, 1, 15, 1, 15],
        "relative_improvement": [1/60, 20/50, 1/70, 15/40, 1/65, 15/45],
        "productivity": [61/2, 70/4, 71/1, 55/3, 66/2.5, 60/3.5],
        "student_at_risk": [0, 0, 0, 1, 0, 1],  # Au moins 2 instances de classe 1
    })

    splitter = TrainTestSplitWithScaling(test_size=0.4, random_state=42)
    
    X_train, X_test, y_train, y_test, *_ = splitter.split(df)

    total_samples = len(df)

    # Vérifier que le total des échantillons est correct
    assert X_train.shape[0] + X_test.shape[0] == total_samples
    assert len(y_train) + len(y_test) == total_samples
    
    # Vérifier que les dimensions correspondent entre X et y
    assert X_train.shape[0] == len(y_train)
    assert X_test.shape[0] == len(y_test)

    # Vérifier que la target n'est pas dans X
    assert "student_at_risk" not in X_train.columns
    assert "student_at_risk" not in X_test.columns

    # Vérifier que les données sont bien normalisées (moyenne ~0, std ~1)
    assert np.abs(X_train.mean().mean()) < 1.0  # Moyenne proche de 0
    assert np.abs(X_train.std().mean() - 1.0) < 0.5  # Std proche de 1