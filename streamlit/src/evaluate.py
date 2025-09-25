"""
Módulo de avaliação do modelo para o projeto Decision AI.
Gera métricas detalhadas e relatórios de performance.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

def load_model(model_path):
    """Carrega modelo treinado."""
    return joblib.load(model_path)

def evaluate_model(model, X_test, y_test):
    """Avalia modelo com métricas detalhadas."""
    print("Avaliando modelo...")
    
    # Predições
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # Métricas básicas
    print("\n=== RELATÓRIO DE CLASSIFICAÇÃO ===")
    print(classification_report(y_test, y_pred))
    
    # Matriz de confusão
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n=== MATRIZ DE CONFUSÃO ===")
    print(cm)
    
    # ROC AUC
    from sklearn.metrics import roc_auc_score, average_precision_score
    auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)
    
    print(f"\n=== MÉTRICAS DE RANKING ===")
    print(f"ROC AUC: {auc:.4f}")
    print(f"PR AUC: {pr_auc:.4f}")
    
    return {
        'auc': auc,
        'pr_auc': pr_auc,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }

def plot_roc_curve(y_test, y_pred_proba, save_path=None):
    """Plota curva ROC."""
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Decision AI Model')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_precision_recall_curve(y_test, y_pred_proba, save_path=None):
    """Plota curva Precision-Recall."""
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'PR Curve (AUC = {pr_auc:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve - Decision AI Model')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def feature_importance_analysis(model, feature_names, top_k=15):
    """Analisa importância das features."""
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    else:
        print("Modelo não suporta análise de importância")
        return None
    
    # Criar DataFrame
    df_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.barplot(data=df_importance.head(top_k), x='importance', y='feature')
    plt.title(f'Top {top_k} Features - Decision AI Model')
    plt.xlabel('Importância')
    plt.tight_layout()
    plt.show()
    
    print(f"\n=== TOP {top_k} FEATURES MAIS IMPORTANTES ===")
    for idx, row in df_importance.head(top_k).iterrows():
        print(f"{row['feature']}: {row['importance']:.4f}")
    
    return df_importance

def evaluate_by_segments(model, X_test, y_test, df_test):
    """Avalia performance por segmentos."""
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    results = {}
    
    # Por status SAP
    if 'is_sap_vaga' in X_test.columns:
        for sap_status in [0, 1]:
            mask = X_test['is_sap_vaga'] == sap_status
            if mask.sum() > 10:  # Mínimo de samples
                auc = roc_auc_score(y_test[mask], y_pred_proba[mask])
                results[f'sap_vaga_{sap_status}'] = auc
    
    # Por nível de match técnico
    if 'tech_overlap_count' in X_test.columns:
        for tech_level in ['baixo', 'medio', 'alto']:
            if tech_level == 'baixo':
                mask = X_test['tech_overlap_count'] == 0
            elif tech_level == 'medio':
                mask = (X_test['tech_overlap_count'] > 0) & (X_test['tech_overlap_count'] <= 3)
            else:
                mask = X_test['tech_overlap_count'] > 3
            
            if mask.sum() > 10:
                auc = roc_auc_score(y_test[mask], y_pred_proba[mask])
                results[f'tech_match_{tech_level}'] = auc
    
    print("\n=== PERFORMANCE POR SEGMENTOS ===")
    for segment, auc in results.items():
        print(f"{segment}: {auc:.4f}")
    
    return results

def main():
    """Pipeline principal de avaliação."""
    from train import load_and_prepare_data
    from feature_engineering import get_final_features
    from sklearn.model_selection import train_test_split
    
    print("=== AVALIAÇÃO DO MODELO DECISION AI ===")
    
    # Carregar dados
    df = load_and_prepare_data("data/vagas.json", "data/prospects.json", "data/applicants.json")
    
    # Preparar dados
    X = df.drop(columns=["vaga_id", "codigo_candidato", "situacao_ord"])
    y = (df["situacao_ord"] == 5).astype(int)
    
    # Split simples para avaliação
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Carregar modelo
    model = load_model("models/model_lgbm.pkl")
    
    # Avaliar
    results = evaluate_model(model, X_test, y_test)
    
    # Plots
    plot_roc_curve(y_test, results['probabilities'])
    plot_precision_recall_curve(y_test, results['probabilities'])
    
    # Importância das features
    feature_importance_analysis(model, X.columns.tolist())
    
    # Análise por segmentos
    evaluate_by_segments(model, X_test, y_test, df)
    
    print("\n✅ Avaliação concluída!")

if __name__ == "__main__":
    main()
