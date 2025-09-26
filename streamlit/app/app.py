# streamlit/app/app.py
import sys
from pathlib import Path

# --- Bootstrapping dos caminhos ---
APP_DIR = Path(__file__).resolve().parent        # .../streamlit/app
ROOT    = APP_DIR.parent                         # .../streamlit
SRC_DIR = ROOT / "src"

# garanta que o Python enxergue a pasta que contém "src" e a própria "src"
for p in (str(ROOT), str(SRC_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

# agora os imports funcionam
import streamlit as st
import pandas as pd
import numpy as np

from src.model_utils import load_model, predict_ranking, get_feature_importance
from src.utils import load_data, format_probability, calculate_metrics_summary, get_status_color

# ---------------------------------
# Configuração da página
# ---------------------------------
st.set_page_config(
    page_title="Decision AI - Recrutamento Inteligente",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🎯 Decision AI - Recrutamento Inteligente")
st.markdown("""
**Solução de IA para encontrar o candidato ideal para cada vaga em tempo hábil**

Desenvolvido para otimizar o processo de match candidato-vaga da Decision usando Machine Learning.
""")

# ---------------------------------
# Paths robustos (independem do WD)
# ---------------------------------
APP_DIR = Path(__file__).resolve().parent          # .../streamlit/app
ROOT    = APP_DIR.parent                           # .../streamlit
DATA_PATH  = ROOT / "data" / "df_clean.csv"
MODEL_PATH = ROOT / "models" / "model_lgbm.pkl"

# ---------------------------------
# Cache dos dados e do modelo
# ---------------------------------
@st.cache_data(show_spinner="Carregando dataset…")
def load_cached_data(path: Path) -> pd.DataFrame:
    return load_data(path)

@st.cache_resource(show_spinner="Carregando modelo…")
def load_cached_model(path: Path):
    return load_model(path)

# ---------------------------------
# Carregar dados e modelo
# ---------------------------------
try:
    df = load_cached_data(DATA_PATH)
    model = load_cached_model(MODEL_PATH)
    st.sidebar.success("✅ Modelo e dados carregados")
except Exception as e:
    st.error(f"❌ Erro ao carregar dados/modelo: {e}")
    st.stop()

# ---------------------------------
# Sidebar – Filtros
# ---------------------------------
st.sidebar.title("🔧 Filtros")
st.sidebar.markdown("---")

vagas_disponiveis = sorted(df["vaga_id"].unique())
vaga_selecionada = st.sidebar.selectbox(
    "🎯 Selecionar Vaga",
    vagas_disponiveis,
    help="Escolha uma vaga para ver o ranking de candidatos",
)

st.sidebar.markdown("### 🎓 Filtros de Qualificação")
filtro_ingles = st.sidebar.checkbox("✅ Inglês OK", value=False)
filtro_senioridade = st.sidebar.checkbox("✅ Senioridade OK", value=False)
filtro_sap = st.sidebar.checkbox("🔶 Conhecimento SAP", value=False)

min_tech_overlap = st.sidebar.slider(
    "🔧 Mínimo Match Técnico",
    min_value=0,
    max_value=20,
    value=0,
    help="Número mínimo de tecnologias em comum",
)

# ---------------------------------
# Aplicar filtros
# ---------------------------------
df_filtrado = df[df["vaga_id"] == vaga_selecionada].copy()

if filtro_ingles:
    df_filtrado = df_filtrado[df_filtrado["ingles_ok"] == 1]
if filtro_senioridade:
    df_filtrado = df_filtrado[df_filtrado["senioridade_ok"] == 1]
if filtro_sap:
    df_filtrado = df_filtrado[df_filtrado["cand_has_sap"] == 1]
if min_tech_overlap > 0:
    df_filtrado = df_filtrado[df_filtrado["tech_overlap_count"] >= min_tech_overlap]

# ---------------------------------
# Métricas da vaga
# ---------------------------------
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("📋 Vaga Selecionada", str(vaga_selecionada))
with col2:
    st.metric("👥 Candidatos Filtrados", len(df_filtrado))
with col3:
    if len(df_filtrado) > 0:
        contratados = (df_filtrado["situacao_ord"] == 5).sum()
        st.metric("✅ Taxa Sucesso Histórica", f"{contratados / len(df_filtrado) * 100:.1f}%")

# ---------------------------------
# Ranking de candidatos
# ---------------------------------
if len(df_filtrado) > 0:
    st.markdown("---")
    st.subheader("🏆 Ranking de Candidatos - IA")

    # Preparar features (remove colunas não preditoras)
    drop_cols = ["vaga_id", "codigo_candidato", "situacao_ord"]
    X = df_filtrado.drop(columns=[c for c in drop_cols if c in df_filtrado.columns])

    # Reordenar/alinhar colunas se o modelo expõe feature_names_in_
    if hasattr(model, "feature_names_in_"):
        X = X.reindex(columns=model.feature_names_in_, fill_value=0)

    try:
        probabilidades = predict_ranking(model, X)
    except Exception as e:
        st.error(f"Erro ao gerar predições: {e}")
        st.stop()

    # Montar ranking
    cols_keep = [
        "codigo_candidato", "vaga_id", "situacao_ord",
        "tech_overlap_count", "ingles_ok", "senioridade_ok",
        "cand_has_sap", "days_update",
    ]
    cols_keep = [c for c in cols_keep if c in df_filtrado.columns]

    df_ranking = df_filtrado[cols_keep].copy()
    df_ranking["probabilidade_contratacao"] = probabilidades
    df_ranking = df_ranking.sort_values("probabilidade_contratacao", ascending=False)

    # Top 10 com cartões
    st.markdown("### 🥇 Top 10 Candidatos Recomendados")
    for idx, (_, cand) in enumerate(df_ranking.head(10).iterrows(), start=1):
        with st.container():
            c1, c2, c3, c4 = st.columns([1, 2, 2, 1])

            with c1:
                st.markdown(f"**#{idx}**")
                st.markdown(f"🆔 {cand['codigo_candidato']}")

            with c2:
                prob_fmt = format_probability(cand["probabilidade_contratacao"])
                st.markdown(f"**🎯 Prob. Contratação: {prob_fmt}**")

                status_map = {2: "📋 Encaminhado", 3: "🎤 Entrevista",
                              4: "✅ Aprovado", 5: "🎉 Contratado", 6: "❌ Reprovado"}
                status = status_map.get(int(cand.get("situacao_ord", -1)), "❓ Desconhecido")
                st.markdown(f"Status: {status}")

            with c3:
                badges = []
                if cand.get("ingles_ok", 0) == 1:
                    badges.append("🇺🇸 Inglês OK")
                if cand.get("senioridade_ok", 0) == 1:
                    badges.append("🎓 Senioridade OK")
                if cand.get("cand_has_sap", 0) == 1:
                    badges.append("🔶 SAP")
                if cand.get("tech_overlap_count", 0) > 0:
                    badges.append(f"🔧 Tech: {int(cand['tech_overlap_count'])}")

                for b in badges:
                    st.markdown(f"`{b}`")

            with c4:
                days = int(cand.get("days_update", 0))
                st.markdown(f"⏱️ {days} dias" if days > 0 else "🆕 Novo")

            st.markdown("---")

    # Tabela completa opcional
    if len(df_ranking) > 10:
        if st.button(f"📋 Ver todos os {len(df_ranking)} candidatos"):
            mostrar_cols = [
                "codigo_candidato", "probabilidade_contratacao",
                "tech_overlap_count", "ingles_ok", "senioridade_ok",
                "cand_has_sap", "days_update",
            ]
            mostrar_cols = [c for c in mostrar_cols if c in df_ranking.columns]
            st.dataframe(df_ranking[mostrar_cols], use_container_width=True)
else:
    st.warning("⚠️ Nenhum candidato encontrado com os filtros aplicados.")

# ---------------------------------
# Footer
# ---------------------------------
st.markdown("---")
st.markdown(
    """
<div style='text-align: center'>
  <p><strong>Decision AI</strong> - Powered by LightGBM | ROC AUC: 0.87 | Precision: 0.31</p>
  <p><em>Otimizando o recrutamento com Inteligência Artificial</em></p>
</div>
""",
    unsafe_allow_html=True,
)
