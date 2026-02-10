import pandas as pd
from zenml import step
from src.feature_engineering import FeatureEngineeringStrategy
@step
def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    return FeatureEngineeringStrategy().add_features(df)
