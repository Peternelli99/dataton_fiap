# 🎯 Decision AI - Recrutamento Inteligente

Solução de Inteligência Artificial para otimizar o processo de recrutamento da Decision, focada em encontrar o candidato ideal para cada vaga em tempo hábil.

## 🚀 Demonstração
- **App Streamlit**: [Link do deploy]
- **Repositório**: [Link do GitHub]
- **Vídeo Pitch**: [Link do vídeo]

## 📊 Performance do Modelo
- **ROC AUC**: 0.8655 ± 0.0066 (86.55% de capacidade discriminativa)
- **Average Precision**: 0.3131 ± 0.0206 (31.31% em dados desbalanceados)
- **Algoritmo**: LightGBM com validação por GroupKFold

## 🏗️ Arquitetura da Solução

### Stack Tecnológica
- **Frontend**: Streamlit
- **Machine Learning**: LightGBM, Scikit-learn
- **Data Processing**: Pandas, NumPy
- **Deployment**: Streamlit Cloud
- **Versionamento**: Git/GitHub

### Pipeline de ML
1. **Pré-processamento**: Limpeza, normalização, encoding
2. **Feature Engineering**: 19 features especializadas em recrutamento
3. **Treinamento**: GroupKFold por vaga, early stopping
4. **Validação**: Métricas adequadas para dados desbalanceados

## ⚡ Como Executar Localmente

### 1. Clonar Repositório
