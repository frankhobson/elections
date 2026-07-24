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

def cal_prob_val(pr, T_val):
    pr = np.clip(pr, 1e-5, 1.0 - 1e-5)
    logit = np.log(pr / (1.0 - pr))
    return 1.0 / (1.0 + np.exp(-logit / T_val))

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
    
    # Cross-validate hybrid models
    for train_idx, val_idx in kf.split(df_known):
        df_tr = df_known.iloc[train_idx]
        df_val = df_known.iloc[val_idx]
        
        from models.correlation_models import HybridElectionsModel
        model = HybridElectionsModel(exec_features, leg_features, exec_params, leg_params)
        model.fit(df_tr)
        
        p_ens_list = []
        for _, row in df_val.iterrows():
            prob = model.predict_row_probability(row)
            is_clean = pd.notna(row["clean_index"]) and row["clean_index"] >= 0.65
            T_val = exec_params["T"] if row["is_presidential"] == 1 else leg_params["T"]
            prob_cal = prob if is_clean else cal_prob_val(prob, T_val)
            p_ens_list.append(prob_cal)
            
        val_indices_global = df_known.index[val_idx]
        df_known.loc[val_indices_global, "predicted_prob"] = p_ens_list

    # Map predicted outcomes using threshold rules
    exec_clean_t = exec_params.get("classification_threshold_clean", 0.50)
    exec_unclean_t = exec_params.get("classification_threshold_unclean", 0.515)
    leg_clean_t = leg_params.get("classification_threshold_clean", 0.50)
    leg_unclean_t = leg_params.get("classification_threshold_unclean", 0.515)
    
    preds_outcome = []
    for _, row in df_known.iterrows():
        is_pres = row["is_presidential"]
        is_clean = pd.notna(row["clean_index"]) and row["clean_index"] >= 0.65
        prob = row["predicted_prob"]
        if is_pres == 1:
            t = exec_clean_t if is_clean else exec_unclean_t
        else:
            t = leg_clean_t if is_clean else leg_unclean_t
        preds_outcome.append(1 if prob >= t else 0)
        
    df_known["predicted_outcome"] = preds_outcome
            
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
            
        from models.correlation_models import HybridElectionsModel
        model = HybridElectionsModel(exec_features, leg_features, exec_params, leg_params)
        model.fit(train_df)
        
        probs_cb = np.zeros(len(test_df))
        probs_xgb = np.zeros(len(test_df))
        probs_ens = np.zeros(len(test_df))
        
        for idx, (_, row) in enumerate(test_df.iterrows()):
            p_xgb, p_cb, p_ens = model.predict_row_probability_split(row)
            is_clean = pd.notna(row["clean_index"]) and row["clean_index"] >= 0.65
            T_val = exec_params["T"] if row["is_presidential"] == 1 else leg_params["T"]
            p_ens_cal = p_ens if is_clean else cal_prob_val(p_ens, T_val)
            
            probs_xgb[idx] = p_xgb
            probs_cb[idx] = p_cb
            probs_ens[idx] = p_ens_cal
            
        y_test = test_df["target_outcome"].values
        is_pres_test = test_df["is_presidential"].values
        
        preds_cb = np.zeros(len(test_df))
        preds_xgb = np.zeros(len(test_df))
        preds_ens = np.zeros(len(test_df))
        
        for i in range(len(test_df)):
            row = test_df.iloc[i]
            is_pres = row["is_presidential"]
            is_clean = pd.notna(row["clean_index"]) and row["clean_index"] >= 0.65
            
            if is_pres == 1:
                t_cb_xgb = exec_params.get("classification_threshold_unclean", 0.515)
                t_ens = exec_params.get("classification_threshold_clean", 0.50) if is_clean else t_cb_xgb
            else:
                t_cb_xgb = leg_params.get("classification_threshold_unclean", 0.515)
                t_ens = leg_params.get("classification_threshold_clean", 0.50) if is_clean else t_cb_xgb
                
            preds_cb[i] = 1 if probs_cb[i] >= t_cb_xgb else 0
            preds_xgb[i] = 1 if probs_xgb[i] >= t_cb_xgb else 0
            preds_ens[i] = 1 if probs_ens[i] >= t_ens else 0
            
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
    
    print("Training final full-database HybridElectionsModel...")
    df_known = df[df["target_outcome"].notna()].copy()
    country_dummies = [col for col in df.columns if col.startswith("country_") and len(col) == 11 and col[8:].isupper()]
    exec_features = [f for f in FEATURES if f not in LEG_SPECIFIC and not (f.startswith("country_") and len(f) == 11 and f[8:].isupper())] + country_dummies
    leg_features = [f for f in FEATURES if f not in EXEC_SPECIFIC and not (f.startswith("country_") and len(f) == 11 and f[8:].isupper())] + country_dummies
    
    exec_params = TUNED_PARAMS.get("executive_model", {"xgb_weight": 0.5, "T": 1.15, "classification_threshold": 0.515})
    leg_params = TUNED_PARAMS.get("legislative_model", {"xgb_weight": 0.5, "T": 1.15, "classification_threshold": 0.515})
    
    from models.correlation_models import HybridElectionsModel
    hybrid_model = HybridElectionsModel(exec_features, leg_features, exec_params, leg_params)
    hybrid_model.fit(df_known)
    
    print("Packaging and writing cached data dictionary to pickle...")
    cache_data = {
        "all_preds": all_preds,
        "df_modeling": df_modeling,
        "cv_df": cv_df,
        "mean_ridge": mean_ridge,
        "mean_xgb": mean_xgb,
        "mean_ensemble": mean_ensemble,
        "hybrid_model": hybrid_model,
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
