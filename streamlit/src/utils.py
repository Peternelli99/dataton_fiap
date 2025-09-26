# streamlit/src/utils.py
from pathlib import Path
import pandas as pd

def _resolve(p: str | Path) -> Path:
    p = Path(p)
    if p.exists():
        return p
    here = Path(__file__).resolve().parent
    for c in [here / p, here.parent / p, here.parent.parent / p]:
        if c.exists():
            return c
    return p

def load_data(data_path):
    path = _resolve(data_path)
    return pd.read_csv(path)

def format_probability(prob):
    return f"{float(prob) * 100:.1f}%"

def get_status_color(situacao_ord: int) -> str:
    colors = {0: "gray", 1: "blue", 2: "orange", 3: "purple", 4: "green", 5: "gold", 6: "red"}
    return colors.get(situacao_ord, "gray")

def calculate_metrics_summary(df: pd.DataFrame):
    total_candidatos = len(df)
    total_vagas = df["vaga_id"].nunique()
    taxa_contratacao = (df["situacao_ord"] == 5).mean() * 100
    return {
        "total_candidatos": total_candidatos,
        "total_vagas": total_vagas,
        "taxa_contratacao": taxa_contratacao,
    }
