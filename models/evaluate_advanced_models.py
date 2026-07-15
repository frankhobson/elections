import os
# Prevent OpenMP conflict issues on macOS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# PyTorch must be imported BEFORE XGBoost and CatBoost to avoid segmentation faults
import torch
import torch.nn as nn
import torch.optim as optim

import sqlite3
import numpy as np
import pandas as pd
import time
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss
from sklearn.base import BaseEstimator, ClassifierMixin

# Import other ML models
import xgboost as xgb
import catboost as cb

# Add parent directory to path so python can find the correlation_models module
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.correlation_models import build_modeling_dataset, FEATURES, calibrate_probability

# Setup PyTorch MLP
class PyTorchMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout=0.3):
        super(PyTorchMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.net(x)

class MLPClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden_dim=64, dropout=0.3, lr=0.003, weight_decay=1e-4, epochs=80, batch_size=64):
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.classes_ = np.array([0, 1])
        
    def fit(self, X, y):
        # Convert X, y to tensors
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y.values if isinstance(y, pd.Series) else y, dtype=torch.float32).unsqueeze(1)
        
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model = PyTorchMLP(X.shape[1], self.hidden_dim, self.dropout)
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.BCELoss()
        
        self.model.train()
        for epoch in range(self.epochs):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                pred = self.model(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
        return self
        
    def predict_proba(self, X):
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            preds = self.model(X_t).numpy()
        return np.hstack([1 - preds, preds])
        
    def predict(self, X):
        prob = self.predict_proba(X)[:, 1]
        return (prob >= 0.5).astype(int)

def evaluate_models():
    # Connect from the workspace root (elections.db is there)
    conn = sqlite3.connect("elections.db")
    
    print("Building modeling dataset...")
    df = build_modeling_dataset(conn, include_upcoming=True)
    conn.close()
    
    # Only train/test on rows with known outcomes
    df_known = df[df["target_outcome"].notna()].copy()
    
    origins = [2010, 2014, 2018, 2022]
    
    # Track metrics
    results = {
        "Baseline Ensemble": {"acc": [], "loss": []},
        "XGBoost (Baseline)": {"acc": [], "loss": []},
        "CatBoost": {"acc": [], "loss": []},
        "PyTorch MLP": {"acc": [], "loss": []},
        "Hybrid Ensemble (Ridge+XGB+Cat+MLP)": {"acc": [], "loss": []}
    }
    
    for y in origins:
        train_df = df_known[df_known["year"] <= y]
        test_df = df_known[(df_known["year"] > y) & (df_known["year"] <= y + 4)]
        
        if train_df.empty or test_df.empty:
            continue
            
        print(f"\nOrigin Year <= {y} -> Testing on {y+1} to {y+4} ({len(test_df)} elections)...")
        
        X_train, y_train = train_df[FEATURES], train_df["target_outcome"]
        X_test, y_test = test_df[FEATURES], test_df["target_outcome"]
        
        # 1. Fill NaNs with 0 (mean) for linear/NN models
        X_train_filled = X_train.fillna(0)
        X_test_filled = X_test.fillna(0)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_filled)
        X_test_scaled = scaler.transform(X_test_filled)
        
        # --- Evaluate Baseline Ensemble ---
        print("Fitting Ridge 0.2...")
        from sklearn.linear_model import LogisticRegression
        lr02 = LogisticRegression(penalty='l2', C=0.2, random_state=42).fit(X_train_scaled, y_train)
        prob_lr02 = lr02.predict_proba(X_test_scaled)[:, 1]
        
        print("Fitting Ridge 1.0...")
        lr10 = LogisticRegression(penalty='l2', C=1.0, random_state=42).fit(X_train_scaled, y_train)
        prob_lr10 = lr10.predict_proba(X_test_scaled)[:, 1]
        
        print("Fitting XGBoost (Baseline)...")
        xgb_model = xgb.XGBClassifier(
            max_depth=3, learning_rate=0.06, n_estimators=150,
            eval_metric='logloss', random_state=42
        )
        xgb_model.fit(X_train, y_train)
        prob_xgb = xgb_model.predict_proba(X_test)[:, 1]
        
        # Combine Baseline Ensemble
        prob_baseline = 0.25 * prob_lr02 + 0.25 * prob_lr10 + 0.50 * prob_xgb
        prob_baseline_cal = np.array([calibrate_probability(p, T=0.6) for p in prob_baseline])
        acc_baseline = accuracy_score(y_test, (prob_baseline_cal >= 0.50).astype(int))
        loss_baseline = log_loss(y_test, prob_baseline_cal)
        
        results["Baseline Ensemble"]["acc"].append(acc_baseline)
        results["Baseline Ensemble"]["loss"].append(loss_baseline)
        
        # --- XGBoost ---
        acc_xgb = accuracy_score(y_test, (prob_xgb >= 0.51).astype(int))
        loss_xgb = log_loss(y_test, prob_xgb)
        results["XGBoost (Baseline)"]["acc"].append(acc_xgb)
        results["XGBoost (Baseline)"]["loss"].append(loss_xgb)
        
        # --- CatBoost ---
        print("Fitting CatBoost...")
        cb_model = cb.CatBoostClassifier(
            iterations=200, depth=4, learning_rate=0.05, verbose=0, random_seed=42
        )
        cb_model.fit(X_train, y_train)
        prob_cb = cb_model.predict_proba(X_test)[:, 1]
        acc_cb = accuracy_score(y_test, (prob_cb >= 0.50).astype(int))
        loss_cb = log_loss(y_test, prob_cb)
        results["CatBoost"]["acc"].append(acc_cb)
        results["CatBoost"]["loss"].append(loss_cb)
        
        # --- PyTorch MLP ---
        print("Fitting PyTorch MLP...")
        mlp_model = MLPClassifier(epochs=80, lr=0.003)
        mlp_model.fit(X_train_scaled, y_train)
        prob_mlp = mlp_model.predict_proba(X_test_scaled)[:, 1]
        acc_mlp = accuracy_score(y_test, (prob_mlp >= 0.50).astype(int))
        loss_mlp = log_loss(y_test, prob_mlp)
        results["PyTorch MLP"]["acc"].append(acc_mlp)
        results["PyTorch MLP"]["loss"].append(loss_mlp)
        
        # --- Hybrid Ensemble (Ridge + XGBoost + CatBoost + MLP) ---
        prob_hybrid = 0.15 * prob_lr02 + 0.15 * prob_lr10 + 0.35 * prob_xgb + 0.20 * prob_cb + 0.15 * prob_mlp
        acc_hybrid = accuracy_score(y_test, (prob_hybrid >= 0.50).astype(int))
        loss_hybrid = log_loss(y_test, prob_hybrid)
        results["Hybrid Ensemble (Ridge+XGB+Cat+MLP)"]["acc"].append(acc_hybrid)
        results["Hybrid Ensemble (Ridge+XGB+Cat+MLP)"]["loss"].append(loss_hybrid)
        
        print(f"  Accuracy: Baseline Ens: {acc_baseline:.4f} | XGB: {acc_xgb:.4f} | CatBoost: {acc_cb:.4f} | MLP: {acc_mlp:.4f} | Hybrid Ens: {acc_hybrid:.4f}")
        
    print("\n" + "="*60)
    print("FINAL BENCHMARK COMPARISON REPORT (AVERAGE OUT-OF-SAMPLE)")
    print("="*60)
    print(f"{'Model Name':35s} | {'Accuracy':10s} | {'Log Loss':10s}")
    print("-"*60)
    for model_name, metrics in results.items():
        avg_acc = np.mean(metrics["acc"])
        avg_loss = np.mean(metrics["loss"])
        print(f"{model_name:35s} | {avg_acc:8.2%} | {avg_loss:8.4f}")
    print("="*60)

if __name__ == "__main__":
    evaluate_models()
