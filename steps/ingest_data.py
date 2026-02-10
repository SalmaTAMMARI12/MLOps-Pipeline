import logging
import pandas as pd
from typing import Annotated
from zenml import step
import mlflow



from src.data_ingestion import DataIngestionFactory, DataIngestor

@step
def ingest_data(data_path: str) -> Annotated[pd.DataFrame, "Raw data"]:
    try:
        strategy_choice = DataIngestionFactory.get_strategy(data_path)
        ingestor = DataIngestor(strategy_choice)
        df = ingestor.execute(data_path)
        mlflow.log_param("data_path", data_path)
        return df
    except Exception as e:
        logging.error(f"Data ingestion failed for path: {data_path}. Error: {e}")
        raise