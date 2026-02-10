import logging
import pandas as pd
import os
from abc import ABC, abstractmethod

class IngestionStrategy(ABC):
    @abstractmethod
    def load_data(self, file_path: str) -> pd.DataFrame:
        pass

class CSVIngestion(IngestionStrategy):
    def load_data(self, file_path: str):
        return pd.read_csv(file_path)

class ParquetIngestion(IngestionStrategy):
    def load_data(self, file_path: str):
        return pd.read_parquet(file_path)

# Other types can be added if necessary

class DataIngestionFactory:
    """Decides which strategy to use."""
    @staticmethod
    def get_strategy(file_path: str):
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.csv':
            return CSVIngestion()
        elif ext == '.parquet':
            return ParquetIngestion()
        else:
            raise ValueError(f"Unsupported file format: {ext}")
      
class DataIngestor:
    def __init__(self, strategy: IngestionStrategy):
        self.strategy = strategy

    def execute(self, file_path: str):
        logging.info(f"Starting ingestion for: {file_path}")
        try:
            df = self.strategy.load_data(file_path)
            logging.info(f"Successfully loaded {len(df)} rows.")
            return df
        except Exception as e:
            logging.error(f"Failed to ingest {file_path}: {e}")
            raise