"""
Módulo de engenharia de features para o projeto Decision AI.
Cria features especializadas para o problema de recrutamento.
"""

import pandas as pd
import numpy as np
import re
from datetime import datetime

# Constantes globais
TECH_TERMS = [
    "sap","abap","hana","sql","python","java",".net","c#","node",
    "aws","azure","gcp","linux","docker","kubernetes","oracle",
    "mysql","postgres","power bi","tableau","react","angular","vue"
]

LANG_MAP = {
    "basico":1,"básico":1,"a1":1,"a2":2,"intermediario":3,
    "intermediário":3,"b1":3,"b2":4,"avancado":5,"avançado":5,
    "fluente":6,"c1":6,"c2":7
}

def get_series(df, col, default=""):
    """Obtém série de uma coluna, retornando default se não existir."""
    return df[col] if col in df.columns else pd.Series([default] * len(df), index=df.index)

def norm_txt(s):
    """Normaliza texto para processamento."""
    s = "" if pd.isna(s) else str(s).lower()
    s = re.sub(r"[^a-z0-9áâãàéêíóôõúç\+\#\.\- ]"," ",s)
    return re.sub(r"\s+"," ",s).strip()

def extract_terms(text):
    """Extrai termos técnicos do texto."""
    t = norm_txt(text)
    return {tkn for tkn in TECH_TERMS if tkn in t}

def create_technical_features(df):
    """Cria features de compatibilidade técnica."""
    # Textos de vagas e candidatos
    vaga_txt = (get_series(df, "perfil_vaga.principais_atividades").astype(str) + " " +
                get_series(df, "perfil_vaga.competencia_tecnicas_e_comportamentais").astype(str))
    
    cand_txt = (get_series(df, "informacoes_profissionais.conhecimentos_tecnicos").astype(str) + " " +
                get_series(df, "cv_pt").astype(str))
    
    # Overlap técnico
    vt = vaga_txt.map(extract_terms)
    ct = cand_txt.map(extract_terms)
    df["tech_overlap_count"] = [len(a.intersection(b)) for a,b in zip(vt,ct)]
    
    # Features SAP
    df["is_sap_vaga"] = get_series(df, "informacoes_basicas.vaga_sap","Não").eq("Sim").astype(np.int8)
    df["cand_has_sap"] = cand_txt.str.contains(r"\bsap\b", regex=True, na=False).astype(np.int8)
    df["sap_pair"] = (df["is_sap_vaga"] & df["cand_has_sap"]).astype(np.int8)
    
    return df

def lang_rank(s):
    """Mapeia nível de idioma para valor ordinal."""
    s = norm_txt(s)
    m = re.search(r"\b([abc][12])\b", s)
    if m:
        return LANG_MAP.get(m.group(1),0)
    for k,v in LANG_MAP.items():
        if k in s:
            return v
    return 0

def create_language_features(df):
    """Cria features de compatibilidade de idiomas."""
    df["vaga_ing_rank"] = get_series(df, "perfil_vaga.nivel_ingles").astype(str).apply(lang_rank).astype("Int8")
    df["vaga_esp_rank"] = get_series(df, "perfil_vaga.nivel_espanhol").astype(str).apply(lang_rank).astype("Int8")
    df["cand_ing_rank"] = get_series(df, "formacao_e_idiomas.nivel_ingles").astype(str).apply(lang_rank).astype("Int8")
    df["cand_esp_rank"] = get_series(df, "formacao_e_idiomas.nivel_espanhol").astype(str).apply(lang_rank).astype("Int8")
    
    # Compatibilidade
    df["ingles_ok"] = (df["cand_ing_rank"] >= df["vaga_ing_rank"]).astype(np.int8)
    df["espanhol_ok"] = (df["cand_esp_rank"] >= df["vaga_esp_rank"]).astype(np.int8)
    
    return df

def sen_vaga(s):
    """Extrai senioridade da vaga."""
    s = norm_txt(s)
    return 3 if "sen" in s else 2 if "plen" in s else 1 if "jun" in s else 0

def sen_cand_from_title(s):
    """Extrai senioridade do título profissional."""
    s = norm_txt(s)
    return 3 if "sen" in s else 2 if "plen" in s else 1 if "jun" in s else 0

def create_seniority_features(df):
    """Cria features de senioridade."""
    df["vaga_sen_rank"] = get_series(df, "perfil_vaga.nivel_profissional").astype(str).apply(sen_vaga).astype("Int8")
    df["cand_sen_rank"] = get_series(df, "informacoes_profissionais.titulo_profissional").astype(str).apply(sen_cand_from_title).astype("Int8")
    
    df["senioridade_gap"] = (df["cand_sen_rank"] - df["vaga_sen_rank"]).astype("Int8")
    df["senioridade_ok"] = (df["cand_sen_rank"] >= df["vaga_sen_rank"]).astype(np.int8)
    
    return df

def map_situacao_ordinal(situacao_str):
    """Mapeia situação do candidato para valor ordinal."""
    if pd.isna(situacao_str):
        return 2
    
    situacao_clean = str(situacao_str).lower().strip()
    
    if "contratado" in situacao_clean:
        return 5
    elif "aprovado" in situacao_clean:
        return 4
    elif "entrevista" in situacao_clean:
        return 3
    elif "encaminhado" in situacao_clean:
        return 2
    elif "contato" in situacao_clean:
        return 1
    elif "cadastrado" in situacao_clean:
        return 0
    elif "reprovado" in situacao_clean:
        return 6
    else:
        return 2

def create_funnel_features(df):
    """Cria features do funil temporal."""
    def to_date(s): 
        return pd.to_datetime(s, dayfirst=True, errors="coerce")
    
    df["dt_cand"] = get_series(df, "data_candidatura").apply(to_date)
    df["dt_ult"] = get_series(df, "ultima_atualizacao").apply(to_date)
    df["days_update"] = (df["dt_ult"] - df["dt_cand"]).dt.days.fillna(0).astype(np.int16)
    
    # Situação ordinal
    df["situacao_ord"] = get_series(df, "situacao_candidato").apply(map_situacao_ordinal).astype("Int8")
    
    return df

def create_interaction_features(df):
    """Cria features de interação."""
    # Binning de CV
    len_cv_pt_raw = get_series(df, "cv_pt").astype(str).str.len()
    df["len_cv_bin"] = pd.qcut(len_cv_pt_raw.rank(method="first"), q=4, labels=False, duplicates="drop").astype("Int8")
    
    # Interação inglês + senioridade
    df["ok_eng_sen"] = (df["ingles_ok"] & df["senioridade_ok"]).astype(np.int8)
    
    return df

def engineer_features(df):
    """Pipeline completo de engenharia de features."""
