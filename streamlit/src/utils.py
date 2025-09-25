import pickle
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# ======= FUNÇÕES DO MODELO =======
def load_model(model_path):
    """Carrega o modelo serializado."""
    path = Path(model_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
    
    try:
        if path.suffix == '.pkl':
            # Tentar com joblib primeiro (mais comum para sklearn/lightgbm)
            try:
                model = joblib.load(path)
            except:
                # Se falhar, tentar com pickle
                with open(path, 'rb') as f:
                    model = pickle.load(f)
        else:
            raise ValueError(f"Formato não suportado: {path.suffix}")
        
        return model
    except Exception as e:
        raise ValueError(f"Erro ao carregar modelo: {str(e)}")

def predict_ranking(model, X):
    """Gera probabilidades de contratação para ranking."""
    try:
        # Para modelos LGBMClassifier (sklearn-like)
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X)[:, 1]
        else:
            raise ValueError("Modelo não suporta predict_proba")
        
        return probabilities
    except Exception as e:
        raise ValueError(f"Erro na predição: {str(e)}")

def get_feature_importance(model, feature_names):
    """Retorna importância das features."""
    try:
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            return pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
        else:
            return None
    except Exception as e:
        print(f"Erro ao obter importância: {str(e)}")
        return None

# ======= FUNÇÕES DE DADOS E FORMATAÇÃO =======
def load_data(data_path):
    """Carrega dados do CSV processado."""
    path = Path(data_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Dataset não encontrado: {data_path}")
    
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        raise ValueError(f"Erro ao carregar dados: {str(e)}")

def format_probability(prob):
    """Formata probabilidade para exibição."""
    return f"{prob*100:.1f}%"

def get_status_color(situacao_ord):
    """Retorna cor baseada no status."""
    colors = {
        0: "gray",    # Cadastrado
        1: "blue",    # Contato  
        2: "orange",  # Encaminhado
        3: "purple",  # Entrevista
        4: "green",   # Aprovado
        5: "gold",    # Contratado
        6: "red"      # Reprovado
    }
    return colors.get(situacao_ord, "gray")

def calculate_metrics_summary(df):
    """Calcula métricas resumo do dataset."""
    total_candidatos = len(df)
    total_vagas = df['vaga_id'].nunique()
    taxa_contratacao = (df['situacao_ord'] == 5).sum() / total_candidatos * 100
    
    return {
        'total_candidatos': total_candidatos,
        'total_vagas': total_vagas,
        'taxa_contratacao': taxa_contratacao
    }
