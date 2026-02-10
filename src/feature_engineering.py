import pandas as pd
import logging

class FeatureEngineeringStrategy:
    """
    Creates meaningful features for the Students AI dataset.
    """

    def add_features(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            df = data.copy()
            df["improvement"] = df["grades_after_ai"] - df["grades_before_ai"]
            df["relative_improvement"] = df["improvement"] / df["grades_before_ai"].replace(0, 1)
            df["productivity"] = df["grades_after_ai"] / df["study_hours_per_day"].replace(0, 1)
            df["screen_overload"] = (df["daily_screen_time_hours"] > 6).astype(int)
            df["student_at_risk"] = (df["grades_after_ai"] < 60).astype(int)

            return df

        except Exception as e:
            logging.error(f"Feature engineering failed: {e}")
            raise
