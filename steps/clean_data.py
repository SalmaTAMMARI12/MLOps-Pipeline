# steps/clean_data.py (CORRIGÉ)
import pandas as pd
from zenml import step
from src.data_cleaning import DataCleaning, MissingValuesStrategy, EncodingStrategy

@step
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean data by handling missing values and encoding categorical variables.
    """
    
    df = DataCleaning(df, MissingValuesStrategy()).handle_data()
    
    df = DataCleaning(df, EncodingStrategy()).handle_data()
    
    return df