import os
import sys
import pickle
import sqlite3
import pandas as pd
import numpy as np
import xgboost as xgb
import catboost as cb
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

BASE_DIR = "/Users/frankhobson/Documents/Antigravity Workspace/Election Correlations and Predictions"
sys.path.insert(0, BASE_DIR)

from models.correlation_models import (
    build_modeling_dataset,
    predict_with_cascading,
    FEATURES,
    EXEC_SPECIFIC,
    LEG_SPECIFIC,
    TUNED_PARAMS
)

DB_PATH = os.path.join(BASE_DIR, "elections.db")
CACHE_PATH = os.path.join(BASE_DIR, "dashboard", "cached_dashboard_data.pkl")

def get_all_predictions(df, preds):
    df_known = df[df["target_outcome"].notna()].copy()
    
    country_dummies = [col for col in df.columns if col.startswith("country_") and len(col) == 11 and col[8:].isupper()]
    exec_features = [f for f in FEATURES if f not in LEG_SPECIFIC and not (f.startswith("country_") and len(f) == 11 and f[8:].isupper())] + country_dummies
    leg_features = [f for f in FEATURES if f not in EXEC_SPECIFIC and not (f.startswith("country_") and len(f) == 11 and f[8:].isupper())] + country_dummies
    
    exec_params = TUNED_PARAMS.get("executive_model", {"xgb_weight": 0.5, "T": 1.15, "classification_threshold": 0.515})
    leg_params = TUNED_PARAMS.get("legislative_model", {"xgb_weight": 0.5, "T": 1.15, "classification_threshold": 0.515})
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    df_known["predicted_prob"] = np.nan
    df_known["predicted_outcome"] = np.nan
    
    # Calibrate probability locally
    def cal_prob_val(pr, T_val):
        pr = np.clip(pr, 1e-5, 1.0 - 1e-5)
        logit = np.log(pr / (1.0 - pr))
        return 1.0 / (1.0 + np.exp(-logit / T_val))
    
    # 1. Cross-validate Executive Model
    df_known_exec = df_known[df_known["is_presidential"] == 1].copy()
    if not df_known_exec.empty:
        X_ex = df_known_exec[exec_features]
        y_ex = df_known_exec["target_outcome"]
        
        for train_idx, val_idx in kf.split(df_known_exec):
            X_tr = X_ex.iloc[train_idx]
            X_val = X_ex.iloc[val_idx]
            y_tr = y_ex.iloc[train_idx]
            
            ex_cb = cb.CatBoostClassifier(
                iterations=200, depth=TUNED_PARAMS["model_parameters"]["cb_depth"],
                learning_rate=TUNED_PARAMS["model_parameters"]["cb_lr"], verbose=0, random_seed=42
            ).fit(X_tr, y_tr)
            
            ex_xgb = xgb.XGBClassifier(
                max_depth=TUNED_PARAMS["model_parameters"]["xgb_max_depth"],
                learning_rate=TUNED_PARAMS["model_parameters"]["xgb_lr"],
                n_estimators=150, eval_metric='logloss', random_state=42
            ).fit(X_tr, y_tr)
            
            prob_cb = ex_cb.predict_proba(X_val)[:, 1]
            prob_xgb = ex_xgb.predict_proba(X_val)[:, 1]
            
            w_ex = exec_params["xgb_weight"]
            p_ens = w_ex * prob_xgb + (1.0 - w_ex) * prob_cb
            p_ens_cal = np.array([cal_prob_val(p, exec_params["T"]) for p in p_ens])
            
            # Map back to group indices
            val_indices_global = df_known_exec.index[val_idx]
            df_known.loc[val_indices_global, "predicted_prob"] = p_ens_cal
            df_known.loc[val_indices_global, "predicted_outcome"] = (p_ens_cal >= exec_params["classification_threshold"]).astype(int)
            
    # 2. Cross-validate Legislative Model
    df_known_leg = df_known[df_known["is_presidential"] == 0].copy()
    if not df_known_leg.empty:
        X_leg = df_known_leg[leg_features]
        y_leg = df_known_leg["target_outcome"]
        
        for train_idx, val_idx in kf.split(df_known_leg):
            X_tr = X_leg.iloc[train_idx]
            X_val = X_leg.iloc[val_idx]
            y_tr = y_leg.iloc[train_idx]
            
            leg_cb = cb.CatBoostClassifier(
                iterations=200, depth=TUNED_PARAMS["model_parameters"]["cb_depth"],
                learning_rate=TUNED_PARAMS["model_parameters"]["cb_lr"], verbose=0, random_seed=42
            ).fit(X_tr, y_tr)
            
            leg_xgb = xgb.XGBClassifier(
                max_depth=TUNED_PARAMS["model_parameters"]["xgb_max_depth"],
                learning_rate=TUNED_PARAMS["model_parameters"]["xgb_lr"],
                n_estimators=150, eval_metric='logloss', random_state=42
            ).fit(X_tr, y_tr)
            
            prob_cb = leg_cb.predict_proba(X_val)[:, 1]
            prob_xgb = leg_xgb.predict_proba(X_val)[:, 1]
            
            w_lg = leg_params["xgb_weight"]
            p_ens = w_lg * prob_xgb + (1.0 - w_lg) * prob_cb
            p_ens_cal = np.array([cal_prob_val(p, leg_params["T"]) for p in p_ens])
            
            # Map back to group indices
            val_indices_global = df_known_leg.index[val_idx]
            df_known.loc[val_indices_global, "predicted_prob"] = p_ens_cal
            df_known.loc[val_indices_global, "predicted_outcome"] = (p_ens_cal >= leg_params["classification_threshold"]).astype(int)
            
    df_known["confidence"] = df_known["predicted_prob"].apply(lambda p: max(p, 1 - p))
    
    # Map columns to match upcoming predictions dataframe
    df_known_mapped = pd.DataFrame()
    df_known_mapped["election_id"] = df_known["election_id"]
    df_known_mapped["source"] = df_known["source"]
    df_known_mapped["country_code"] = df_known["country_code"]
    df_known_mapped["region"] = df_known["region"]
    df_known_mapped["year"] = df_known["year"]
    df_known_mapped["election_type"] = df_known["is_presidential"].map({1: "Executive", 0: "Legislative"})
    df_known_mapped["is_scheduled"] = df_known["is_scheduled"]
    df_known_mapped["clean_index"] = df_known["clean_index"]
    df_known_mapped["gdp_growth"] = df_known.get("gdp_growth", np.nan)
    df_known_mapped["raw_probability"] = df_known["predicted_prob"]
    df_known_mapped["predicted_outcome"] = df_known["predicted_outcome"]
    df_known_mapped["predicted_winner"] = df_known["predicted_outcome"].map({1: "Incumbent", 0: "Challenger"})
    df_known_mapped["raw_confidence"] = df_known["confidence"]
    df_known_mapped["data_completeness"] = df_known["data_completeness"]
    df_known_mapped["adjusted_confidence"] = df_known["confidence"]
    df_known_mapped["data_source_flags"] = df_known["data_source_flags"]
    df_known_mapped["target_outcome"] = df_known["target_outcome"]
    df_known_mapped["election_date"] = df_known.get("election_date", None)
    
    # Combine with upcoming cascading predictions
    upcoming_part = preds.copy()
    
    # Combine
    all_preds = pd.concat([df_known_mapped, upcoming_part], ignore_index=True)
    return all_preds

def evaluate_national_models_live(df):
    df_known = df[df["target_outcome"].notna()].copy()
    
    origins = [2010, 2014, 2018, 2022]
    cv_rows = []
    cb_all, xgb_all, ens_all = [], [], []
    
    country_dummies = [col for col in df.columns if col.startswith("country_") and len(col) == 11 and col[8:].isupper()]
    exec_features = [f for f in FEATURES if f not in LEG_SPECIFIC and not (f.startswith("country_") and len(f) == 11 and f[8:].isupper())] + country_dummies
    leg_features = [f for f in FEATURES if f not in EXEC_SPECIFIC and not (f.startswith("country_") and len(f) == 11 and f[8:].isupper())] + country_dummies
    
    exec_params = TUNED_PARAMS.get("executive_model", {"xgb_weight": 0.5, "T": 1.15, "classification_threshold": 0.515})
    leg_params = TUNED_PARAMS.get("legislative_model", {"xgb_weight": 0.5, "T": 1.15, "classification_threshold": 0.515})
    
    for y in origins:
        train_df = df_known[df_known["year"] <= y]
        test_df = df_known[(df_known["year"] > y) & (df_known["year"] <= y + 4)]
        
        if train_df.empty or test_df.empty:
            continue
            
        train_exec = train_df[train_df["is_presidential"] == 1]
        train_leg = train_df[train_df["is_presidential"] == 0]
        test_exec = test_df[test_df["is_presidential"] == 1]
        test_leg = test_df[test_df["is_presidential"] == 0]
        
        probs_cb = np.zeros(len(test_df))
        probs_xgb = np.zeros(len(test_df))
        probs_ens = np.zeros(len(test_df))
        test_indices = test_df.index.tolist()
        
        # Executive
        if not train_exec.empty and not test_exec.empty:
            X_tr_ex, y_tr_ex = train_exec[exec_features], train_exec["target_outcome"]
            X_te_ex, y_te_ex = test_exec[exec_features], test_exec["target_outcome"]
            
            ex_cb = cb.CatBoostClassifier(
                iterations=200, depth=TUNED_PARAMS["model_parameters"]["cb_depth"],
                learning_rate=TUNED_PARAMS["model_parameters"]["cb_lr"], verbose=0, random_seed=42
            ).fit(X_tr_ex, y_tr_ex)
            
            ex_xgb = xgb.XGBClassifier(
                max_depth=TUNED_PARAMS["model_parameters"]["xgb_max_depth"],
                learning_rate=TUNED_PARAMS["model_parameters"]["xgb_lr"],
                n_estimators=150, eval_metric='logloss', random_state=42
            ).fit(X_tr_ex, y_tr_ex)
            
            p_cb_ex = ex_cb.predict_proba(X_te_ex)[:, 1]
            p_xgb_ex = ex_xgb.predict_proba(X_te_ex)[:, 1]
            p_ens_ex = exec_params["xgb_weight"] * p_xgb_ex + (1.0 - exec_params["xgb_weight"]) * p_cb_ex
            
            T_ex = exec_params["T"]
            def cal_ex(p):
                p = np.clip(p, 1e-5, 1.0 - 1e-5)
                logit = np.log(p / (1.0 - p))
                return 1.0 / (1.0 + np.exp(-logit / T_ex))
            p_ens_ex_cal = np.array([cal_ex(p) for p in p_ens_ex])
            
            for idx_in_test, original_idx in enumerate(test_exec.index):
                pos = test_indices.index(original_idx)
                probs_cb[pos] = p_cb_ex[idx_in_test]
                probs_xgb[pos] = p_xgb_ex[idx_in_test]
                probs_ens[pos] = p_ens_ex_cal[idx_in_test]
                
        # Legislative
        if not train_leg.empty and not test_leg.empty:
            X_tr_lg, y_tr_lg = train_leg[leg_features], train_leg["target_outcome"]
            X_te_lg, y_te_lg = test_leg[leg_features], test_leg["target_outcome"]
            
            lg_cb = cb.CatBoostClassifier(
                iterations=200, depth=TUNED_PARAMS["model_parameters"]["cb_depth"],
                learning_rate=TUNED_PARAMS["model_parameters"]["cb_lr"], verbose=0, random_seed=42
            ).fit(X_tr_lg, y_tr_lg)
            
            lg_xgb = xgb.XGBClassifier(
                max_depth=TUNED_PARAMS["model_parameters"]["xgb_max_depth"],
                learning_rate=TUNED_PARAMS["model_parameters"]["xgb_lr"],
                n_estimators=150, eval_metric='logloss', random_state=42
            ).fit(X_tr_lg, y_tr_lg)
            
            p_cb_lg = lg_cb.predict_proba(X_te_lg)[:, 1]
            p_xgb_lg = lg_xgb.predict_proba(X_te_lg)[:, 1]
            p_ens_lg = leg_params["xgb_weight"] * p_xgb_lg + (1.0 - leg_params["xgb_weight"]) * p_cb_lg
            
            T_lg = leg_params["T"]
            def cal_lg(p):
                p = np.clip(p, 1e-5, 1.0 - 1e-5)
                logit = np.log(p / (1.0 - p))
                return 1.0 / (1.0 + np.exp(-logit / T_lg))
            p_ens_lg_cal = np.array([cal_lg(p) for p in p_ens_lg])
            
            for idx_in_test, original_idx in enumerate(test_leg.index):
                pos = test_indices.index(original_idx)
                probs_cb[pos] = p_cb_lg[idx_in_test]
                probs_xgb[pos] = p_xgb_lg[idx_in_test]
                probs_ens[pos] = p_ens_lg_cal[idx_in_test]
                
        y_test = test_df["target_outcome"].values
        is_pres_test = test_df["is_presidential"].values
        
        preds_cb = np.zeros(len(test_df))
        preds_xgb = np.zeros(len(test_df))
        preds_ens = np.zeros(len(test_df))
        
        for i in range(len(test_df)):
            t = exec_params["classification_threshold"] if is_pres_test[i] == 1 else leg_params["classification_threshold"]
            preds_cb[i] = 1 if probs_cb[i] >= t else 0
            preds_xgb[i] = 1 if probs_xgb[i] >= t else 0
            preds_ens[i] = 1 if probs_ens[i] >= t else 0
            
        acc_cb = accuracy_score(y_test, preds_cb)
        acc_xgb = accuracy_score(y_test, preds_xgb)
        acc_ens = accuracy_score(y_test, preds_ens)
        
        cv_rows.append({
            "Origin": f"<= {y} (Test {y+1}-{y+4})",
            "CatBoost Classifier": acc_cb,
            "XGBoost Classifier": acc_xgb,
            "Voting Ensemble": acc_ens
        })
        cb_all.append(acc_cb)
        xgb_all.append(acc_xgb)
        ens_all.append(acc_ens)
        
    return pd.DataFrame(cv_rows), np.mean(cb_all), np.mean(xgb_all), np.mean(ens_all)

def main():
    print("Connecting to database...")
    conn = sqlite3.connect(DB_PATH)
    df = build_modeling_dataset(conn, include_upcoming=True)
    cursor = conn.cursor()
    cursor.execute("SELECT country_code, country_name, region FROM countries ORDER BY country_name;")
    c_names = pd.DataFrame(cursor.fetchall(), columns=["country_code", "country_name", "region"])
    
    print("Generating historical V-Dem map data cache...")
    query_map_all = """
        SELECT c.country_code, c.country_name, c.latitude, c.longitude,
               v.clean_elections_index, v.polyarchy_index, v.regime_type, v.year
        FROM countries c
        LEFT JOIN vdem_indicators v ON c.country_code = v.country_code
        WHERE v.year >= 1990 AND v.year <= 2025;
    """
    cursor.execute(query_map_all)
    vdem_map_data = pd.DataFrame(
        cursor.fetchall(), 
        columns=["country_code", "country_name", "latitude", "longitude", "clean_elections_index", "polyarchy_index", "regime_type", "year"]
    )
    conn.close()
    
    print("Running cascading forecasts (predict_with_cascading)...")
    preds, df_modeling = predict_with_cascading(df, return_df=True)
    
    print("Generating all predictions dataset (get_all_predictions)...")
    all_preds = get_all_predictions(df, preds)
    
    print("Evaluating live national models validation (evaluate_national_models_live)...")
    cv_df, mean_ridge, mean_xgb, mean_ensemble = evaluate_national_models_live(df)
    
    print("Training final full-database models...")
    df_known = df[df["target_outcome"].notna()].copy()
    country_dummies = [col for col in df.columns if col.startswith("country_") and len(col) == 11 and col[8:].isupper()]
    exec_features = [f for f in FEATURES if f not in LEG_SPECIFIC and not (f.startswith("country_") and len(f) == 11 and f[8:].isupper())] + country_dummies
    leg_features = [f for f in FEATURES if f not in EXEC_SPECIFIC and not (f.startswith("country_") and len(f) == 11 and f[8:].isupper())] + country_dummies
    
    train_exec = df_known[df_known["is_presidential"] == 1]
    train_leg = df_known[df_known["is_presidential"] == 0]
    
    ex_xgb = xgb.XGBClassifier(
        max_depth=TUNED_PARAMS["model_parameters"]["xgb_max_depth"],
        learning_rate=TUNED_PARAMS["model_parameters"]["xgb_lr"],
        n_estimators=150, eval_metric='logloss', random_state=42
    ).fit(train_exec[exec_features], train_exec["target_outcome"])
    
    ex_cb = cb.CatBoostClassifier(
        iterations=200, depth=TUNED_PARAMS["model_parameters"]["cb_depth"],
        learning_rate=TUNED_PARAMS["model_parameters"]["cb_lr"], verbose=0, random_seed=42
    ).fit(train_exec[exec_features], train_exec["target_outcome"])
    
    leg_xgb = xgb.XGBClassifier(
        max_depth=TUNED_PARAMS["model_parameters"]["xgb_max_depth"],
        learning_rate=TUNED_PARAMS["model_parameters"]["xgb_lr"],
        n_estimators=150, eval_metric='logloss', random_state=42
    ).fit(train_leg[leg_features], train_leg["target_outcome"])
    
    leg_cb = cb.CatBoostClassifier(
        iterations=200, depth=TUNED_PARAMS["model_parameters"]["cb_depth"],
        learning_rate=TUNED_PARAMS["model_parameters"]["cb_lr"], verbose=0, random_seed=42
    ).fit(train_leg[leg_features], train_leg["target_outcome"])
    
    print("Packaging and writing cached data dictionary to pickle...")
    cache_data = {
        "all_preds": all_preds,
        "df_modeling": df_modeling,
        "cv_df": cv_df,
        "mean_ridge": mean_ridge,
        "mean_xgb": mean_xgb,
        "mean_ensemble": mean_ensemble,
        "ex_xgb": ex_xgb,
        "ex_cb": ex_cb,
        "leg_xgb": leg_xgb,
        "leg_cb": leg_cb,
        "preds": preds,
        "c_names": c_names,
        "vdem_map_data": vdem_map_data
    }
    
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    print(f"Precompute cache generated successfully! File size: {os.path.getsize(CACHE_PATH)/1024/1024:.2f} MB")

if __name__ == "__main__":
    main()
