import pandas as pd
from src.feature_engineering import FeatureEngineeringStrategy

def test_feature_engineering_creates_expected_columns():
    df = pd.DataFrame({
        "age": [19, 18],
        "education_level": ["Bachelor", "High School"],
        "study_hours_per_day": [2.0, 4.0],
        "uses_ai": [0, 1],
        "ai_tools_used": ["None", "Copilot"],
        "purpose_of_ai": ["None", "Homework"],
        "grades_before_ai": [60, 50],
        "grades_after_ai": [61, 70],
        "daily_screen_time_hours": [3, 7],
    })

    out = FeatureEngineeringStrategy().add_features(df)

    for col in ["improvement", "relative_improvement", "productivity", "screen_overload", "student_at_risk"]:
        assert col in out.columns

    # check a couple of values
    assert out.loc[0, "improvement"] == 1
    assert out.loc[1, "screen_overload"] == 1
