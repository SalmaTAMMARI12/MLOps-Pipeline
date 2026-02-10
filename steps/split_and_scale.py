import pandas as pd
from typing import Tuple, Annotated
from zenml import step
from src.data_splitting_scaling import TrainTestSplitWithScaling

@step
def split_and_scale(
    df: pd.DataFrame
) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
]:
    """
    Split data into train/test sets and scale features.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    splitter = TrainTestSplitWithScaling(test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = splitter.split(df)
    return X_train, X_test, y_train, y_test