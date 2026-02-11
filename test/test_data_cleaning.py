import pandas as pd
from src.data_cleaning import MissingValuesStrategy

def test_missing_values_strategy_sets_none_for_ai_fields_when_uses_ai_no():
    df = pd.DataFrame({
        "uses_ai": ["No", "Yes", "No"],
        "ai_tools_used": [None, "Copilot", None],
        "purpose_of_ai": [None, "Homework", None],
        "education_level": ["Bachelor", "Bachelor", None],
        "age": [19, 18, 20],
        "study_hours_per_day": [2.0, 3.0, 1.0],
        "grades_before_ai": [60, 50, 70],
        "grades_after_ai": [61, 60, 71],
        "daily_screen_time_hours": [3, 4, 2],
    })

    out = MissingValuesStrategy().handle_data(df)

    assert set(out["uses_ai"].unique()).issubset({"No", "Yes"})
    mask = out["uses_ai"] == "No"
    assert (out.loc[mask, "ai_tools_used"] == "None").all()
    assert (out.loc[mask, "purpose_of_ai"] == "None").all()

def test_cleaning_fills_unknown_for_remaining_categoricals():
    df = pd.DataFrame({
        "uses_ai": ["Yes"],
        "ai_tools_used": [None],
        "purpose_of_ai": [None],
        "education_level": [None],
        "age": [19],
        "study_hours_per_day": [2.0],
        "grades_before_ai": [60],
        "grades_after_ai": [61],
        "daily_screen_time_hours": [3],
    })
    out = MissingValuesStrategy().handle_data(df)
    assert out.loc[0, "ai_tools_used"] == "Unknown"
    assert out.loc[0, "purpose_of_ai"] == "Unknown"
    assert out.loc[0, "education_level"] == "Unknown"