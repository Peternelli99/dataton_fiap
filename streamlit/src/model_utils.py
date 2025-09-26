# streamlit/src/model_utils.py
from pathlib import Path
import numpy as np
import pandas as pd

def _resolve(p: str | Path) -> Path:
    p = Path(p)
    if p.exists():
        return p
    # tenta caminhos relativos comuns (src -> raiz do projeto)
    here = Path(__file__).resolve().parent      # .../streamlit/src
    candidates = [
        here / p,
        here.parent / p,                        # .../streamlit/<p>
        here.parent.parent / p,                 # .../<p>
    ]
    for c in candidates:
        if c.exists():
            return c
    return p  # deixamos falhar adiante, se realmente não existir

def load_model(model_path):
    path = _resolve(model_path)
    # tenta joblib; se não tiver instalado, cai para pickle
    try:
        import joblib
        return joblib.load(path)
    except Exception:
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)

def predict_ranking(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    raise ValueError("Modelo não suporta predict_proba")

def get_feature_importance(model, feature_names):
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
        return (pd.DataFrame({"feature": feature_names, "importance": imp})
                  .sort_values("importance", ascending=False))
    return None
