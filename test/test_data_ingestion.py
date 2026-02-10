import pytest
import pandas as pd
from unittest.mock import MagicMock
from src.data_ingestion import (
    DataIngestionFactory, 
    CSVIngestion, 
    ParquetIngestion, 
    DataIngestor
)

# Unit tests

def test_factory_returns_correct_strategies():
    """Verify the factory selects the right class based on extension."""
    assert isinstance(DataIngestionFactory.get_strategy("data.csv"), CSVIngestion)
    assert isinstance(DataIngestionFactory.get_strategy("data.parquet"), ParquetIngestion)

def test_factory_raises_value_error():
    """Verify factory rejects unsupported formats."""
    with pytest.raises(ValueError, match="Unsupported file format"):
        DataIngestionFactory.get_strategy("data.txt")

def test_ingestor_executes_strategy():
    """Verify Ingestor delegates the call to the underlying strategy."""
    mock_strategy = MagicMock()
    ingestor = DataIngestor(mock_strategy)
    ingestor.execute("dummy_path.csv")
    
    mock_strategy.load_data.assert_called_once_with("dummy_path.csv")

# Integration tests 

@pytest.fixture
def sample_csv(tmp_path):
    """Creates a real temporary CSV file."""
    file_path = tmp_path / "test.csv"
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    df.to_csv(file_path, index=False)
    return str(file_path)

def test_csv_ingestion_integration(sample_csv):
    """Test the full flow from factory to dataframe for a CSV."""
    strategy = DataIngestionFactory.get_strategy(sample_csv)
    ingestor = DataIngestor(strategy)
    df = ingestor.execute(sample_csv)
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert list(df.columns) == ["a", "b"]

def test_ingestor_re_raises_exception():
    """Verify that failures are logged and then raised (no silent failure)."""
    ingestor = DataIngestor(CSVIngestion())
    with pytest.raises(FileNotFoundError):
        ingestor.execute("non_existent_file.csv")