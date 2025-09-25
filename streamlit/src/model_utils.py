import pickle
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

def load_model(model_path):
    """Carrega o modelo serializado."""
    path = Path(model_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
    
    try:
        model = joblib.load(path)
        return model
    except Exception as e:
        raise ValueError(f"Erro ao carregar modelo: {str(e)}")

def predict_ranking(model, X):
    """Gera probabilidades de contratação para ranking."""
    try:
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X)[:, 1]
        else:
            raise ValueError("Modelo não suporta predict_proba")
        
        return probabilities
    except Exception as e:
        raise ValueError(f"Erro na predição: {str(e)}")
