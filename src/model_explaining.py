from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, List, Tuple

import io
import numpy as np
import pandas as pd
import shap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


class ExplanationStrategy(ABC):
    @abstractmethod
    def explain(self, model: Any, data: pd.DataFrame) -> shap.Explanation:
        pass


def _ensure_dataframe(data: Any) -> pd.DataFrame:
    if not isinstance(data, pd.DataFrame):
        raise ValueError("data doit être un pandas.DataFrame (pour les colonnes et l'index).")
    if data.shape[0] == 0:
        raise ValueError("data est vide.")
    return data


def _to_png_image(fig=None) -> Image.Image:
    buf = io.BytesIO()
    if fig is None:
        plt.savefig(buf, format="png", bbox_inches="tight", dpi=150)
        plt.close()
    else:
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
        plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


def _select_class_explanation(
    expl: shap.Explanation,
    class_index: Optional[int],
) -> Tuple[shap.Explanation, Optional[int]]:
    """
    Rend l'explication 2D (n_samples, n_features).
    Si multi-classe, sélectionne class_index (par défaut 1 si possible).
    """
    values = expl.values

    if values is None:
        raise RuntimeError("SHAP Explanation invalide (values manquants).")

    # Cas multi-classe fréquent: (n, f, c)
    if isinstance(values, np.ndarray) and values.ndim == 3:
        n_classes = values.shape[2]
        if class_index is None:
            class_index = 1 if n_classes > 1 else 0
        if not (0 <= class_index < n_classes):
            raise ValueError(f"class_index doit être entre 0 et {n_classes-1}")

        base_values = expl.base_values
        # base_values peut être (n, c) ou (c,) selon versions
        if isinstance(base_values, np.ndarray) and base_values.ndim == 2:
            base_sel = base_values[:, class_index]
        elif isinstance(base_values, np.ndarray) and base_values.ndim == 1 and base_values.shape[0] == n_classes:
            base_sel = np.full((values.shape[0],), base_values[class_index])
        else:
            base_sel = base_values

        expl_sel = shap.Explanation(
            values=values[:, :, class_index],
            base_values=base_sel,
            data=expl.data,
            feature_names=expl.feature_names,
        )
        return expl_sel, class_index

    # Déjà 2D: (n, f)
    return expl, class_index


class SHAPExplanationStrategy(ExplanationStrategy):
    """
    SHAP strategy: choisit un explainer selon le type de modèle.
    IMPORTANT: force_plot/waterfall sont à la demande.
    """

    def __init__(self, class_index: Optional[int] = None, max_background: int = 200):
        """
        class_index: pour multi-classe (None => auto)
        max_background: limite d'échantillons pour KernelExplainer / background
        """
        self.class_index = class_index
        self.max_background = max_background
        self.explainer = None

    def explain(self, model: Any, data: pd.DataFrame) -> shap.Explanation:
        data = _ensure_dataframe(data)

        try:
            # Background pour Kernel/Linear si besoin (évite temps énorme)
            background = data
            if len(data) > self.max_background:
                background = data.sample(self.max_background, random_state=42)

            # KNN: KernelExplainer (coûteux)
            if isinstance(model, KNeighborsClassifier):
                # predict_proba si dispo (meilleur pour classification)
                if hasattr(model, "predict_proba"):
                    f = model.predict_proba
                else:
                    f = model.predict

                explainer = shap.KernelExplainer(f, background)
                shap_values = explainer(data)

            # Tree models
            elif isinstance(model, RandomForestClassifier):
                explainer = shap.TreeExplainer(model)
                shap_values = explainer(data)

            # LogisticRegression: LinearExplainer ou Explainer auto
            elif isinstance(model, LogisticRegression):
                # LinearExplainer est fait pour ce cas
                explainer = shap.LinearExplainer(model, background)
                shap_values = explainer(data)

            # Fallback
            else:
                explainer = shap.Explainer(model, background)
                shap_values = explainer(data)

            self.explainer = explainer

            # normaliser multi-classe si besoin
            shap_values_sel, used_class = _select_class_explanation(shap_values, self.class_index)
            self.class_index = used_class
            return shap_values_sel

        except Exception as e:
            raise RuntimeError(f"Error calculating SHAP values for {type(model).__name__}: {e}") from e

    # ✅ Global summary (safe en prod)
    def plot_summary(self, shap_values: shap.Explanation, data: pd.DataFrame) -> Image.Image:
        data = _ensure_dataframe(data)
        try:
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, data, show=False)
            return _to_png_image()
        except Exception as e:
            raise RuntimeError(f"Error generating Summary Plot: {e}") from e

    def plot_summary_bar(self, shap_values: shap.Explanation, data: pd.DataFrame) -> Image.Image:
        data = _ensure_dataframe(data)
        try:
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, data, plot_type="bar", show=False)
            return _to_png_image()
        except Exception as e:
            raise RuntimeError(f"Error generating Summary Bar Plot: {e}") from e

    # ✅ On-demand ONLY (production)
    def plot_force_on_demand(
        self,
        shap_values: shap.Explanation,
        data: pd.DataFrame,
        sample_index: int = 0
    ) -> Optional[Image.Image]:
        """
        Force plot: à appeler uniquement quand l'utilisateur le demande (API/UI).
        Retourne PNG (matplotlib=True). Si ça échoue => None.
        """
        data = _ensure_dataframe(data)

        if sample_index < 0 or sample_index >= data.shape[0]:
            raise IndexError(f"sample_index {sample_index} out of bounds [0, {data.shape[0]})")

        try:
            base_val = shap_values.base_values[sample_index]
            sv = shap_values.values[sample_index]

            # scalars
            if isinstance(base_val, np.ndarray):
                base_val = float(base_val.reshape(-1)[0])

            # si sv est 2D (rare) -> aplatir
            sv = np.array(sv).reshape(-1)

            plt.figure(figsize=(12, 3))
            shap.force_plot(
                base_val,
                sv,
                data.iloc[sample_index, :],
                feature_names=data.columns.tolist(),
                matplotlib=True,
                show=False
            )
            return _to_png_image()

        except Exception as e:
            print(f"Error generating Force Plot for index {sample_index}: {e}")
            return None

    # ✅ On-demand (utile aussi)
    def plot_waterfall_on_demand(self, shap_values: shap.Explanation, sample_index: int = 0) -> Optional[Image.Image]:
        try:
            plt.figure(figsize=(10, 6))
            shap.plots.waterfall(shap_values[sample_index], show=False)
            return _to_png_image()
        except Exception as e:
            print(f"Waterfall plot failed: {e}")
            return None

    # ✅ JSON pour dashboard
    def get_shap_values(
        self,
        shap_values: shap.Explanation,
        data: pd.DataFrame,
        sample_index: int = 0,
        top_k: int = 10
    ) -> Dict[str, Any]:
        data = _ensure_dataframe(data)

        if sample_index < 0 or sample_index >= data.shape[0]:
            raise IndexError(f"sample_index {sample_index} out of bounds [0, {data.shape[0]})")

        try:
            base_val = shap_values.base_values[sample_index]
            sv = shap_values.values[sample_index]

            if isinstance(base_val, np.ndarray):
                base_val = float(base_val.reshape(-1)[0])

            sv = np.array(sv).reshape(-1)
            feature_names = data.columns.tolist()

            rows: List[Dict[str, Any]] = []
            for i, feat in enumerate(feature_names):
                val = data.iloc[sample_index, i]
                s = float(sv[i])
                rows.append({
                    "feature": feat,
                    "feature_value": float(val) if np.isscalar(val) else str(val),
                    "shap_value": s,
                    "impact": "positive" if s > 0 else "negative"
                })

            rows_sorted = sorted(rows, key=lambda x: abs(x["shap_value"]), reverse=True)

            return {
                "base_value": float(base_val),
                "class_index": self.class_index,
                "shap_values": rows_sorted,
                "top_features": rows_sorted[:top_k]
            }

        except Exception as e:
            raise RuntimeError(f"Error extracting SHAP values: {e}") from e


class ModelExplainer:
    def __init__(self, model: Any, data: pd.DataFrame, strategy: ExplanationStrategy) -> None:
        self.model = model
        self.data = data
        self.strategy = strategy

    def explain(self) -> shap.Explanation:
        try:
            return self.strategy.explain(self.model, self.data)
        except Exception as e:
            raise RuntimeError(f"Error in model explanation process: {e}") from e
