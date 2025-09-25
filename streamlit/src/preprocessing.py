"""
Módulo de pré-processamento de dados para o projeto Decision AI.
Contém funções para limpeza e preparação dos dados brutos.
"""

import pandas as pd
import numpy as np
import json
import re
from pathlib import Path
from sklearn.preprocessing import StandardScaler

def load_json(path: Path):
    """Carrega arquivo JSON com encoding UTF-8."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def flatten_vagas(raw_vagas):
    """Normaliza dados de vagas JSON para DataFrame."""
    df_vagas = pd.json_normalize(list(raw_vagas.values()), sep='.')
    df_vagas["vaga_id"] = list(map(str, raw_vagas.keys()))
    
    cols_vagas = [
        "vaga_id",
        "informacoes_basicas.titulo_vaga",
        "informacoes_basicas.vaga_sap",
        "informacoes_basicas.cliente",
        "perfil_vaga.nivel_profissional",
        "perfil_vaga.nivel_ingles",
        "perfil_vaga.nivel_espanhol",
        "perfil_vaga.cidade",
        "perfil_vaga.estado",
        "perfil_vaga.pais",
        "perfil_vaga.principais_atividades",
        "perfil_vaga.competencia_tecnicas_e_comportamentais",
    ]
    
    return df_vagas[[c for c in cols_vagas if c in df_vagas.columns]]

def flatten_prospects(raw_prospects):
    """Normaliza dados de prospects JSON para DataFrame."""
    rows = []
    for vaga_id, payload in raw_prospects.items():
        for p in payload.get("prospects", []):
            r = p.copy()
            r["vaga_id"] = str(vaga_id)
            rows.append(r)
    
    df_prospects = pd.DataFrame(rows).rename(
        columns={"codigo":"codigo_candidato","situacao_candidado":"situacao_candidato"}
    )
    return df_prospects

def flatten_applicants(raw_applicants):
    """Normaliza dados de candidatos JSON para DataFrame."""
    df_app = pd.json_normalize(list(raw_applicants.values()), sep='.')
    df_app["codigo_candidato"] = list(map(str, raw_applicants.keys()))
    
    if "infos_basicas.codigo_profissional" in df_app.columns:
        df_app["codigo_candidato"] = df_app["infos_basicas.codigo_profissional"].fillna(
            df_app["codigo_candidato"]
        ).astype(str)

    cols_app = [
        "codigo_candidato",
        "infos_basicas.nome",
        "informacoes_profissionais.titulo_profissional",
        "informacoes_profissionais.area_atuacao",
        "formacao_e_idiomas.nivel_academico",
        "formacao_e_idiomas.nivel_ingles",
        "formacao_e_idiomas.nivel_espanhol",
        "informacoes_profissionais.conhecimentos_tecnicos",
        "cv_pt",
    ]
    
    return df_app[[c for c in cols_app if c in df_app.columns]]

def clean_dataframe(df):
    """Aplica limpeza básica no DataFrame."""
    # Remove duplicatas
    df = df.drop_duplicates(subset=["vaga_id","codigo_candidato"], keep="first")
    
    # Remove espaços em branco
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype(str).str.strip()
    
    # Normaliza valores SAP
    if "informacoes_basicas.vaga_sap" in df.columns:
        df["informacoes_basicas.vaga_sap"] = df["informacoes_basicas.vaga_sap"].str.lower().map({
            "sim":"Sim","yes":"Sim","true":"Sim","não":"Não","nao":"Não","no":"Não"
        }).fillna("Não")
    
    return df

def create_basic_features(df):
    """Cria features básicas a partir dos dados."""
    # Tamanho do CV
    df["len_cv_pt"] = df.get("cv_pt","").astype(str).str.len()
    df["len_cv_pt"] = pd.to_numeric(df["len_cv_pt"], errors="coerce").fillna(0)
    
    # Normalização Z-score
    scaler = StandardScaler()
    df["len_cv_pt_z"] = scaler.fit_transform(df[["len_cv_pt"]]).astype(np.float32)
    
    return df.drop(columns=["len_cv_pt"], errors="ignore")

def preprocess_data(vagas_path, prospects_path, applicants_path):
    """Pipeline completo de pré-processamento."""
    # Carregar dados
    raw_vagas = load_json(Path(vagas_path))
    raw_prospects = load_json(Path(prospects_path))
    raw_applicants = load_json(Path(applicants_path))
    
    # Flatten JSONs
    df_vagas = flatten_vagas(raw_vagas)
    df_prospects = flatten_prospects(raw_prospects)
    df_app = flatten_applicants(raw_applicants)
    
    # Join tables
    df = df_prospects.merge(df_vagas, on="vaga_id", how="left").merge(df_app, on="codigo_candidato", how="left")
    
    # Aplicar limpeza
    df = clean_dataframe(df)
    df = create_basic_features(df)
    
    return df

if __name__ == "__main__":
    # Exemplo de uso
    df = preprocess_data("../data/vagas.json", "../data/prospects.json", "../data/applicants.json")
    print(f"Dataset pré-processado: {df.shape}")
    print(df.head())
