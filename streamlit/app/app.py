import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Adicionar src ao path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from model_utils import load_model, predict_ranking  # ← CORRETO
from utils import load_data, format_probability      # ← AGORA FUNCIONA


# Configuração da página
st.set_page_config(
    page_title="Decision AI - Recrutamento Inteligente",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Header principal
st.title("🎯 Decision AI - Recrutamento Inteligente")
st.markdown("""
**Solução de IA para encontrar o candidato ideal para cada vaga em tempo hábil**

Desenvolvido para otimizar o processo de match candidato-vaga da Decision usando Machine Learning.
""")

# Sidebar com filtros
st.sidebar.title("🔧 Filtros")
st.sidebar.markdown("---")

# Cache dos dados
@st.cache_data
def load_cached_data():
    return load_data("data/df_clean.csv")

# Cache do modelo
@st.cache_resource
def load_cached_model():
    return load_model("models/model_lgbm.pkl")

# Carregar dados e modelo
try:
    df = load_cached_data()
    model = load_cached_model()
    st.sidebar.success("✅ Modelo carregado com sucesso")
except Exception as e:
    st.error(f"❌ Erro ao carregar dados/modelo: {str(e)}")
    st.stop()

# Filtros interativos
vagas_disponiveis = sorted(df['vaga_id'].unique())
vaga_selecionada = st.sidebar.selectbox(
    "🎯 Selecionar Vaga",
    vagas_disponiveis,
    help="Escolha uma vaga para ver o ranking de candidatos"
)

# Filtros de qualificação
st.sidebar.markdown("### 🎓 Filtros de Qualificação")
filtro_ingles = st.sidebar.checkbox("✅ Inglês OK", value=False)
filtro_senioridade = st.sidebar.checkbox("✅ Senioridade OK", value=False)
filtro_sap = st.sidebar.checkbox("🔶 Conhecimento SAP", value=False)

# Filtros de compatibilidade técnica
min_tech_overlap = st.sidebar.slider(
    "🔧 Mínimo Match Técnico",
    min_value=0,
    max_value=20,
    value=0,
    help="Número mínimo de tecnologias em comum"
)

# Aplicar filtros
df_filtrado = df[df['vaga_id'] == vaga_selecionada].copy()

if filtro_ingles:
    df_filtrado = df_filtrado[df_filtrado['ingles_ok'] == 1]
if filtro_senioridade:
    df_filtrado = df_filtrado[df_filtrado['senioridade_ok'] == 1]
if filtro_sap:
    df_filtrado = df_filtrado[df_filtrado['cand_has_sap'] == 1]
if min_tech_overlap > 0:
    df_filtrado = df_filtrado[df_filtrado['tech_overlap_count'] >= min_tech_overlap]

# Mostrar informações da vaga
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("📋 Vaga Selecionada", vaga_selecionada)
with col2:
    st.metric("👥 Candidatos Filtrados", len(df_filtrado))
with col3:
    if len(df_filtrado) > 0:
        contratados = (df_filtrado['situacao_ord'] == 5).sum()
        st.metric("✅ Taxa Sucesso Histórica", f"{contratados/len(df_filtrado)*100:.1f}%")

# Ranking de candidatos
if len(df_filtrado) > 0:
    st.markdown("---")
    st.subheader("🏆 Ranking de Candidatos - IA")
    
    # Fazer predições
    X = df_filtrado.drop(columns=['vaga_id', 'codigo_candidato', 'situacao_ord'])
    probabilidades = predict_ranking(model, X)
    
    # Criar ranking
    df_ranking = df_filtrado[['codigo_candidato', 'vaga_id', 'situacao_ord', 
                            'tech_overlap_count', 'ingles_ok', 'senioridade_ok', 
                            'cand_has_sap', 'days_update']].copy()
    df_ranking['probabilidade_contratacao'] = probabilidades
    df_ranking = df_ranking.sort_values('probabilidade_contratacao', ascending=False)
    
    # Exibir top 10
    st.markdown("### 🥇 Top 10 Candidatos Recomendados")
    
    for idx, (_, candidato) in enumerate(df_ranking.head(10).iterrows()):
        with st.container():
            col1, col2, col3, col4 = st.columns([1, 2, 2, 1])
            
            with col1:
                st.markdown(f"**#{idx+1}**")
                st.markdown(f"🆔 {candidato['codigo_candidato']}")
            
            with col2:
                prob_formatada = format_probability(candidato['probabilidade_contratacao'])
                st.markdown(f"**🎯 Prob. Contratação: {prob_formatada}**")
                
                # Status atual
                status_map = {2: "📋 Encaminhado", 3: "🎤 Entrevista", 
                            4: "✅ Aprovado", 5: "🎉 Contratado", 6: "❌ Reprovado"}
                status = status_map.get(candidato['situacao_ord'], "❓ Desconhecido")
                st.markdown(f"Status: {status}")
            
            with col3:
                # Badges de qualificação
                badges = []
                if candidato['ingles_ok'] == 1:
                    badges.append("🇺🇸 Inglês OK")
                if candidato['senioridade_ok'] == 1:
                    badges.append("🎓 Senioridade OK")
                if candidato['cand_has_sap'] == 1:
                    badges.append("🔶 SAP")
                if candidato['tech_overlap_count'] > 0:
                    badges.append(f"🔧 Tech: {candidato['tech_overlap_count']}")
                
                for badge in badges:
                    st.markdown(f"`{badge}`")
            
            with col4:
                if candidato['days_update'] > 0:
                    st.markdown(f"⏱️ {candidato['days_update']} dias")
                else:
                    st.markdown("🆕 Novo")
            
            st.markdown("---")
    
    # Botão para ver todos
    if len(df_ranking) > 10:
        if st.button(f"📋 Ver todos os {len(df_ranking)} candidatos"):
            st.dataframe(
                df_ranking[['codigo_candidato', 'probabilidade_contratacao', 
                          'tech_overlap_count', 'ingles_ok', 'senioridade_ok',
                          'cand_has_sap', 'days_update']],
                use_container_width=True
            )

else:
    st.warning("⚠️ Nenhum candidato encontrado com os filtros aplicados.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p><strong>Decision AI</strong> - Powered by LightGBM | ROC AUC: 0.87 | Precision: 0.31</p>
    <p><em>Otimizando o recrutamento com Inteligência Artificial</em></p>
</div>
""", unsafe_allow_html=True)
