# 🎯 Decision AI - Recrutamento Inteligente

Solução de Inteligência Artificial para otimizar o processo de recrutamento da Decision, focada em encontrar o candidato ideal para cada vaga em tempo hábil.

## 🚀 Demonstração
- **App Streamlit**: [[Link do deploy](https://datatonfiap-fqy7r6qyggeequgnytu9b4.streamlit.app/)]
- **Repositório**: [\[Link do GitHub\]](https://github.com/Peternelli99/dataton_fiap)
- **Vídeo Pitch**: [[Link do vídeo](https://drive.google.com/file/d/1Y845r0mfg-4f2QnY2Kinym5pSyoQCRvz/view?usp=sharing)]

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

1. **Navegue até a pasta streamlit:
cd C:\Users\Doug_\Downloads\datathon_2.0\datathon\streamlit
2. **Execute o Streamlit:
streamlit run app\app.py
3. **O navegador será aberto automaticamente em http://localhost:8501.
Use a sidebar para filtrar vagas, idiomas, senioridade, SAP e match técnico.

### Treinando modelo novamente

1. **Carregue os dados (df_clean.csv) e separe features/target.
2. **Divida em treino/teste (train_test_split).
3. **Treine o LightGBM (LGBMClassifier) e avalie (ROC AUC).
4. **Salve o modelo em models/model_lgbm.pkl com joblib.dump.
