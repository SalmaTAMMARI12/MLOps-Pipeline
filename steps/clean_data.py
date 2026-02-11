import pandas as pd
from typing import Tuple, Annotated, Dict
from zenml import step
from src.data_cleaning import DataCleaning, MissingValuesStrategy, EncodingStrategy

@step
def clean_data(
    df: pd.DataFrame
) -> Tuple[
    Annotated[pd.DataFrame, "cleaned_df"],
    Annotated[Dict, "fitted_encoders"], # On ajoute la sortie pour les encodeurs
]:
    """
    Nettoie les données et encode les variables catégorielles.
    """
    # 1. Gestion des valeurs manquantes
    cleaning_pipeline = DataCleaning(df, MissingValuesStrategy())
    df = cleaning_pipeline.handle_data()

    # 2. Encodage (Récupération des encodeurs)
    encoding_strategy = EncodingStrategy()
    df, fitted_encoders = encoding_strategy.handle_data(df)

    return df, fitted_encoders