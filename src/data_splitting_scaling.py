import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from abc import ABC, abstractmethod
from typing import Tuple, Any

class SplittingStrategy(ABC):
    @abstractmethod
    def split(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Any]:
        pass


class TrainTestSplitWithScaling(SplittingStrategy):
    """
    Splits data into train/test and applies StandardScaler.
    Returns the datasets AND the fitted scaler object.
    """
    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.cols_to_scale = None

    def split(self, data: pd.DataFrame):
        try:
            if "student_at_risk" not in data.columns:
                raise KeyError("Target 'student_at_risk' not found.")

            y = data["student_at_risk"]
            X = data.drop(columns=["student_at_risk"])

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=y if y.nunique() == 2 else None,
            )
            
            # Identification des colonnes numériques à scaler
            self.cols_to_scale = X_train.select_dtypes(include="number").columns.tolist()
            # On exclut les colonnes déjà binaires (0/1) pour ne pas fausser le scaling
            self.cols_to_scale = [c for c in self.cols_to_scale if c not in ["uses_ai", "screen_overload"]]
            
            # FIT + TRANSFORM
            self.scaler.fit(X_train[self.cols_to_scale])
            X_train[self.cols_to_scale] = self.scaler.transform(X_train[self.cols_to_scale])
            X_test[self.cols_to_scale] = self.scaler.transform(X_test[self.cols_to_scale])

            # IMPORTANT : On retourne l'objet scaler à la fin
            return X_train, X_test, y_train, y_test, self.scaler

        except Exception as e:
            logging.error(f"Split+Scaling failed: {e}")
            raise