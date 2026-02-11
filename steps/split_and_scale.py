import pandas as pd
from typing import Tuple, Annotated
from zenml import step
from src.data_splitting_scaling import TrainTestSplitWithScaling
from sklearn.preprocessing import StandardScaler # Import nécessaire pour le type de retour

@step
def split_and_scale(
    df: pd.DataFrame
) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
    Annotated[StandardScaler, "fitted_scaler"], # On ajoute le scaler ici
]:
    """
    Split data into train/test sets and scale features.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, fitted_scaler)
    """
    splitter = TrainTestSplitWithScaling(test_size=0.2, random_state=42)
    
    # On récupère maintenant 5 éléments
    X_train, X_test, y_train, y_test, fitted_scaler = splitter.split(df)
    
    return X_train, X_test, y_train, y_test, fitted_scaler