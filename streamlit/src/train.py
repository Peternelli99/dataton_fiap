"""
Módulo de treinamento do modelo para o projeto Decision AI.
Executa o pipeline completo de treinamento com validação cruzada.
"""

import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from lightgbm.callback import early_stopping, log_evaluation
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score
import joblib
from pathlib import Path

from preprocessing import preprocess_data
from feature_engineering import engineer_features, get_final_features

def load_and_prepare_data(vagas_path, prospects_path, applicants_path):
    """Carrega e prepara dados para treinamento."""
    print("Carregando e processando dados...")
    
    # Pré-processamento
    df = preprocess_data(vagas_path, prospects_path, applicants_path)
    
    # Engenharia de features
    df = engineer_features(df)
    
    # Selecionar features finais
    keys = ["vaga_id", "codigo_candidato"]
    features = get_final_features()
    
    df_final = pd.concat([df[keys], df[features]], axis=1)
    
    print(f"Dataset preparado: {df_final.shape}")
    return df_final

def train_model(df, n_folds=5):
    """Treina modelo com validação cruzada."""
    print("Iniciando treinamento...")
    
    # Preparar dados
    X = df.drop(columns=["vaga_id", "codigo_candidato", "situacao_ord"])
    y = (df["situacao_ord"] == 5).astype(int)
    
    print(f"Balanceamento: {y.sum()} positivos de {len(y)} total ({y.mean()*100:.2f}%)")
    
    # Configurar validação cruzada
    gkf = GroupKFold(n_splits=n_folds)
    
    auc_scores = []
    pr_scores = []
    models = []
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=df["vaga_id"])):
        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]
        
        print(f"\nFold {fold + 1} - Positives in train: {sum(y_train)}, Positives in val: {sum(y_val)}")
        
        # Pular fold se não houver positivos
        if sum(y_train) == 0 or sum(y_val) == 0:
            print(f"Skipping fold {fold + 1} due to lack of positive class")
            continue
        
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        
        # Treinar modelo
        model = LGBMClassifier(
            objective="binary",
            n_estimators=1000,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
        )
        
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric=["auc", "average_precision"],
            callbacks=[early_stopping(stopping_rounds=50), log_evaluation(period=50)],
        )
        
        # Avaliar
        y_pred = model.predict_proba(X_val, num_iteration=model.best_iteration_)[:, 1]
        
        auc = roc_auc_score(y_val, y_pred)
        pr = average_precision_score(y_val, y_pred)
        
        print(f"ROC AUC: {auc:.4f}, Average Precision: {pr:.4f}")
        
        auc_scores.append(auc)
        pr_scores.append(pr)
        models.append(model)
    
    # Métricas finais
    print(f"\nResultados finais:")
    print(f"Mean ROC AUC: {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")
    print(f"Mean Average Precision: {np.mean(pr_scores):.4f} ± {np.std(pr_scores):.4f}")
    
    # Retornar melhor modelo
    best_idx = np.argmax(auc_scores)
    best_model = models[best_idx]
    
    return best_model, auc_scores, pr_scores

def save_model(model, model_path):
    """Salva modelo treinado."""
    Path(model_path).parent.mkdir(exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Modelo salvo em: {model_path}")

def main():
    """Pipeline principal de treinamento."""
    # Caminhos
    vagas_path = "data/vagas.json"
    prospects_path = "data/prospects.json"
    applicants_path = "data/applicants.json"
    model_path = "models/model_lgbm.pkl"
    
    # Executar pipeline
    df = load_and_prepare_data(vagas_path, prospects_path, applicants_path)
    model, auc_scores, pr_scores = train_model(df)
    save_model(model, model_path)
    
    print("\n✅ Treinamento concluído com sucesso!")
    
    return model, auc_scores, pr_scores

if __name__ == "__main__":
    model, auc_scores, pr_scores = main()
