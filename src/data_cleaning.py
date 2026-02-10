import logging
from abc import ABC, abstractmethod
from typing import Union, Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class DataStrategy(ABC):
    """
    Abstract class definining strategy for handling data
    """
    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass

class MissingValuesStrategy(DataStrategy):
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            df = data.copy()
            df.loc[df["uses_ai"] == "No", ["ai_tools_used", "purpose_of_ai"]] = "None"
            for col in df.select_dtypes(include="object"):
                df[col] = df[col].fillna("Unknown")
            for col in df.select_dtypes(include="number"):
                df[col] = df[col].fillna(df[col].median())
            return df

        except Exception as e:
            logging.error(f"Missing value handling failed: {e}")
            raise   


class EncodingStrategy(DataStrategy):
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            df = data.copy()
            for col in df.select_dtypes(include="object"):
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
            return df
        except Exception as e:
            logging.error(f"Encoding failed: {e}")
            raise

class DataCleaning:
    """
    Executes a data cleaning strategy
    """

    def __init__(self, data: pd.DataFrame, strategy: DataStrategy):
        self.df = data
        self.strategy = strategy

    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        try:
            return self.strategy.handle_data(self.df)
        except Exception as e:
            logging.error(f"Error in data cleaning process: {e}")
            raise e

