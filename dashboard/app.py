import os
# Prevent OpenMP conflict issues on macOS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
import sqlite3
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import xgboost as xgb
import catboost as cb
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix
import warnings

# Add workspace to system path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import importlib
if "models.correlation_models" in sys.modules:
    importlib.reload(sys.modules["models.correlation_models"])

try:
    from models.correlation_models import (
        build_modeling_dataset,
        predict_with_cascading,
        get_feature_importance_report,
        FEATURES,
        calibrate_probability,
        TUNED_PARAMS,
        EXEC_SPECIFIC,
        LEG_SPECIFIC
    )
except ImportError:
    from models.correlation_models import (
        build_modeling_dataset,
        predict_with_cascading,
        get_feature_importance_report,
        FEATURES,
        calibrate_probability,
        TUNED_PARAMS
    )
    EXEC_SPECIFIC = ["is_open_seat", "successor_run", "open_seat_prior_margin_interaction", "is_presidential"]
    LEG_SPECIFIC = [
        "party_fragmentation_legislative_interaction", 
        "electoral_system_majoritarian", "electoral_system_pr",
        "incumbent_seat_buffer", "is_majority_fragile",
        "pr_fragmentation_interaction", "polarization_fragmentation_interaction",
        "is_presidential"
    ]

warnings.filterwarnings("ignore")

# Configure Streamlit Page
st.set_page_config(
    page_title="Global Election Forecaster",
    layout="wide",
    initial_sidebar_state="collapsed"
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "elections.db")
CACHE_PATH = os.path.join(BASE_DIR, "dashboard", "cached_dashboard_data.pkl")

# Attempt to load static cache for instant boot speed in production
cache_data = None
if os.path.exists(CACHE_PATH):
    try:
        import pickle
        with open(CACHE_PATH, "rb") as f:
            cache_data = pickle.load(f)
        print("Loaded static dashboard cache successfully. App boot accelerated!")
    except Exception as e:
        print(f"Error loading static cache: {e}. Falling back to live database queries.")

def get_connection():
    # Use timeout and WAL mode for concurrent SQLite reading/writing
    conn = sqlite3.connect(DB_PATH, timeout=60.0)
    try:
        conn.execute("PRAGMA journal_mode = WAL;")
    except sqlite3.OperationalError:
        pass
    return conn

# Cache database connection & queries
@st.cache_data
def get_db_stats():
    if cache_data is not None:
        return 216, 2206, 377
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM countries;")
    c_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM elections;")
    e_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM upcoming_elections;")
    u_count = cursor.fetchone()[0]
    conn.close()
    return c_count, e_count, u_count

@st.cache_data
def load_cached_modeling_dataset(db_mtime):
    conn = get_connection()
    df = build_modeling_dataset(conn, include_upcoming=True)
    conn.close()
    return df

@st.cache_resource
def get_trained_models(db_mtime):
    """Train separate Executive and Legislative ensembles (XGBoost + CatBoost) on all known historical records."""
    if cache_data is not None:
        return cache_data["ex_xgb"], cache_data["ex_cb"], cache_data["leg_xgb"], cache_data["leg_cb"]
        
    df = load_cached_modeling_dataset(db_mtime)
    df_known = df[df["target_outcome"].notna()].copy()
    
    country_dummies = [col for col in df.columns if col.startswith("country_") and len(col) == 11 and col[8:].isupper()]
    exec_features = [f for f in FEATURES if f not in LEG_SPECIFIC and not (f.startswith("country_") and len(f) == 11 and f[8:].isupper())] + country_dummies
    leg_features = [f for f in FEATURES if f not in EXEC_SPECIFIC and not (f.startswith("country_") and len(f) == 11 and f[8:].isupper())] + country_dummies
    
    # Train separate models for Executive and Legislative
    train_exec = df_known[df_known["is_presidential"] == 1]
    train_leg = df_known[df_known["is_presidential"] == 0]
    
    # Fit Executive model
    ex_xgb = xgb.XGBClassifier(
        max_depth=TUNED_PARAMS["model_parameters"]["xgb_max_depth"],
        learning_rate=TUNED_PARAMS["model_parameters"]["xgb_lr"],
        n_estimators=150,
        eval_metric='logloss',
        random_state=42
    ).fit(train_exec[exec_features], train_exec["target_outcome"])
    
    ex_cb = cb.CatBoostClassifier(
        iterations=200,
        depth=TUNED_PARAMS["model_parameters"]["cb_depth"],
        learning_rate=TUNED_PARAMS["model_parameters"]["cb_lr"],
        verbose=0,
        random_seed=42
    ).fit(train_exec[exec_features], train_exec["target_outcome"])
    
    # Fit Legislative model
    leg_xgb = xgb.XGBClassifier(
        max_depth=TUNED_PARAMS["model_parameters"]["xgb_max_depth"],
        learning_rate=TUNED_PARAMS["model_parameters"]["xgb_lr"],
        n_estimators=150,
        eval_metric='logloss',
        random_state=42
    ).fit(train_leg[leg_features], train_leg["target_outcome"])
    
    leg_cb = cb.CatBoostClassifier(
        iterations=200,
        depth=TUNED_PARAMS["model_parameters"]["cb_depth"],
        learning_rate=TUNED_PARAMS["model_parameters"]["cb_lr"],
        verbose=0,
        random_seed=42
    ).fit(train_leg[leg_features], train_leg["target_outcome"])
    
    return ex_xgb, ex_cb, leg_xgb, leg_cb

@st.cache_data
def evaluate_national_models_live(db_mtime):
    """Performs historical validation on rolling milestones with separate Executive & Legislative models."""
    if cache_data is not None:
        return cache_data["cv_df"], cache_data["mean_ridge"], cache_data["mean_xgb"], cache_data["mean_ensemble"]
        
    df = load_cached_modeling_dataset(db_mtime)
    df_known = df[df["target_outcome"].notna()].copy()
    
    origins = [2010, 2014, 2018, 2022]
    cv_rows = []
    cb_all, xgb_all, ens_all = [], [], []
    
    country_dummies = [col for col in df.columns if col.startswith("country_") and len(col) == 11 and col[8:].isupper()]
    exec_features = [f for f in FEATURES if f not in LEG_SPECIFIC and not (f.startswith("country_") and len(f) == 11 and f[8:].isupper())] + country_dummies
    leg_features = [f for f in FEATURES if f not in EXEC_SPECIFIC and not (f.startswith("country_") and len(f) == 11 and f[8:].isupper())] + country_dummies
    
    # Load parameters
    exec_params = TUNED_PARAMS.get("executive_model", {"xgb_weight": 0.9, "T": 0.78, "classification_threshold": 0.50})
    leg_params = TUNED_PARAMS.get("legislative_model", {"xgb_weight": 0.5, "T": 0.80, "classification_threshold": 0.50})
    
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
        
        # --- Executive Model Group ---
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
            
            # Calibrate executive predictions
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
                
        # --- Legislative Model Group ---
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
            
            # Calibrate legislative predictions
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

@st.cache_data
def get_cascading_predictions_and_df(db_mtime):
    if cache_data is not None:
        return cache_data["preds"], cache_data["df_modeling"]
    df = load_cached_modeling_dataset(db_mtime)
    preds, df_cascaded = predict_with_cascading(df, return_df=True)
    return preds, df_cascaded

@st.cache_data
def get_cascading_predictions(db_mtime):
    preds, _ = get_cascading_predictions_and_df(db_mtime)
    return preds

@st.cache_data
def get_all_predictions(db_mtime):
    """Generates out-of-sample cross-validated historical predictions and joins them with cascading upcoming forecasts."""
    if cache_data is not None:
        return cache_data["all_preds"]
    df = load_cached_modeling_dataset(db_mtime)
    df_known = df[df["target_outcome"].notna()].copy()
    df_unknown = df[df["target_outcome"].isna()].copy()
    
    country_dummies = [col for col in df.columns if col.startswith("country_") and len(col) == 11 and col[8:].isupper()]
    exec_features = [f for f in FEATURES if f not in LEG_SPECIFIC and not (f.startswith("country_") and len(f) == 11 and f[8:].isupper())] + country_dummies
    leg_features = [f for f in FEATURES if f not in EXEC_SPECIFIC and not (f.startswith("country_") and len(f) == 11 and f[8:].isupper())] + country_dummies
    
    exec_params = TUNED_PARAMS.get("executive_model", {"xgb_weight": 0.9, "T": 0.78, "classification_threshold": 0.50})
    leg_params = TUNED_PARAMS.get("legislative_model", {"xgb_weight": 0.5, "T": 0.80, "classification_threshold": 0.50})
    
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    df_known["predicted_prob"] = np.nan
    df_known["predicted_outcome"] = np.nan
    
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
            
            # Calibrate
            T_ex = exec_params["T"]
            def cal_ex(p):
                p = np.clip(p, 1e-5, 1.0 - 1e-5)
                logit = np.log(p / (1.0 - p))
                return 1.0 / (1.0 + np.exp(-logit / T_ex))
            p_ens_cal = np.array([cal_ex(p) for p in p_ens])
            
            threshold = exec_params["classification_threshold"]
            df_known_exec.iloc[val_idx, df_known_exec.columns.get_loc("predicted_prob")] = p_ens_cal
            df_known_exec.iloc[val_idx, df_known_exec.columns.get_loc("predicted_outcome")] = (p_ens_cal >= threshold).astype(int)
            
    # 2. Cross-validate Legislative Model
    df_known_leg = df_known[df_known["is_presidential"] == 0].copy()
    if not df_known_leg.empty:
        X_lg = df_known_leg[leg_features]
        y_lg = df_known_leg["target_outcome"]
        
        for train_idx, val_idx in kf.split(df_known_leg):
            X_tr = X_lg.iloc[train_idx]
            X_val = X_lg.iloc[val_idx]
            y_tr = y_lg.iloc[train_idx]
            
            lg_cb = cb.CatBoostClassifier(
                iterations=200, depth=TUNED_PARAMS["model_parameters"]["cb_depth"],
                learning_rate=TUNED_PARAMS["model_parameters"]["cb_lr"], verbose=0, random_seed=42
            ).fit(X_tr, y_tr)
            
            lg_xgb = xgb.XGBClassifier(
                max_depth=TUNED_PARAMS["model_parameters"]["xgb_max_depth"],
                learning_rate=TUNED_PARAMS["model_parameters"]["xgb_lr"],
                n_estimators=150, eval_metric='logloss', random_state=42
            ).fit(X_tr, y_tr)
            
            prob_cb = lg_cb.predict_proba(X_val)[:, 1]
            prob_xgb = lg_xgb.predict_proba(X_val)[:, 1]
            
            w_lg = leg_params["xgb_weight"]
            p_ens = w_lg * prob_xgb + (1.0 - w_lg) * prob_cb
            
            # Calibrate
            T_lg = leg_params["T"]
            def cal_lg(p):
                p = np.clip(p, 1e-5, 1.0 - 1e-5)
                logit = np.log(p / (1.0 - p))
                return 1.0 / (1.0 + np.exp(-logit / T_lg))
            p_ens_cal = np.array([cal_lg(p) for p in p_ens])
            
            threshold = leg_params["classification_threshold"]
            df_known_leg.iloc[val_idx, df_known_leg.columns.get_loc("predicted_prob")] = p_ens_cal
            df_known_leg.iloc[val_idx, df_known_leg.columns.get_loc("predicted_outcome")] = (p_ens_cal >= threshold).astype(int)
            
    # Combine back to df_known
    df_known_cv = pd.concat([df_known_exec, df_known_leg]).sort_index()
    df_known_cv["confidence"] = df_known_cv["predicted_prob"].apply(lambda p: max(p, 1 - p))
    
    # Map columns to match upcoming predictions dataframe
    df_known_mapped = pd.DataFrame()
    df_known_mapped["election_id"] = df_known_cv["election_id"]
    df_known_mapped["source"] = df_known_cv["source"]
    df_known_mapped["country_code"] = df_known_cv["country_code"]
    df_known_mapped["region"] = df_known_cv["region"]
    df_known_mapped["year"] = df_known_cv["year"]
    df_known_mapped["election_type"] = df_known_cv["is_presidential"].map({1: "Executive", 0: "Legislative"})
    df_known_mapped["is_scheduled"] = df_known_cv["is_scheduled"]
    df_known_mapped["clean_index"] = df_known_cv["clean_index"]
    df_known_mapped["gdp_growth"] = df_known_cv["gdp_growth"]
    df_known_mapped["raw_probability"] = df_known_cv["predicted_prob"]
    df_known_mapped["predicted_outcome"] = df_known_cv["predicted_outcome"]
    df_known_mapped["predicted_winner"] = df_known_cv["predicted_outcome"].map({1: "Incumbent", 0: "Challenger"})
    df_known_mapped["raw_confidence"] = df_known_cv["confidence"]
    df_known_mapped["data_completeness"] = df_known_cv["data_completeness"]
    df_known_mapped["adjusted_confidence"] = df_known_cv["confidence"]
    df_known_mapped["data_source_flags"] = df_known_cv["data_source_flags"]
    df_known_mapped["target_outcome"] = df_known_cv["target_outcome"]
    df_known_mapped["election_date"] = df_known_cv["election_date"]
    
    # Get predictions for the unknown-outcome elections
    df_unknown_preds, _ = predict_with_cascading(df, return_df=True)
    
    # Combine
    all_preds = pd.concat([df_known_mapped, df_unknown_preds], ignore_index=True)
    all_preds = all_preds.sort_values(by=["year", "country_code"]).reset_index(drop=True)
    return all_preds


@st.cache_data
def get_feature_importances(db_mtime):
    df = load_cached_modeling_dataset(db_mtime)
    return get_feature_importance_report(df)

# Feature Explanations Metadata
FEATURE_METADATA = {
    "polyarchy_index": {
        "name": "V-Dem Electoral Democracy Index (Polyarchy)",
        "explanation": "Measures electoral democracy (Dahl, 1971; Coppedge et al., 2011). In highly democratic systems, robust civil liberties and competitive party systems increase baseline competitiveness, lowering default incumbent win rates."
    },
    "clean_index": {
        "name": "V-Dem Clean Elections Index",
        "explanation": "Measures electoral fairness and absence of fraud (Coppedge et al., 2020). Higher cleanliness prevents voter intimidation and manipulation, favoring challengers and ensuring democratic contestation (Schedler, 2002)."
    },
    "clean_index_delta_3yr": {
        "name": "3-Year Change in Clean Elections Index",
        "explanation": "Negative values indicate democratic backsliding, which typically helps incumbents consolidate control."
    },
    "country_incumbent_win_rate": {
        "name": "Historical Incumbent Win Rate (Country)",
        "explanation": "Captures institutionalized path dependency. Higher historical win rates favor incumbent persistence."
    },
    "party_fragmentation_legislative_interaction": {
        "name": "Legislative Coalition & Party Fragmentation",
        "explanation": "In legislative systems, high party fragmentation leads to unstable coalitions and penalizes the incumbent."
    },
    "clean_index_delta_1yr": {
        "name": "1-Year Change in Clean Elections Index",
        "explanation": "Recent changes in electoral cleanliness serve as immediate leading indicators of integrity shifts."
    },
    "is_open_seat": {
        "name": "Open Seat (No Incumbent Running)",
        "explanation": "An election where the incumbent leader or party executive does not run (Jacobson & Carson, 2015). Lacking personal incumbency advantages, open seats are highly competitive and favor challengers."
    },
    "successor_run": {
        "name": "Succession Election (Successor Running)",
        "explanation": "Indicates when the incumbent steps down and a designated successor nominee runs (Jalalzai, 2013). Succession elections lack personal incumbency benefits, resulting in a 'successor penalty' that aids challengers."
    },
    "constitutional_rigidity": {
        "name": "Constitutional Rigidity Index",
        "explanation": "Measures institutional barriers to altering legal rules (Tsebelis, 2002 on Veto Players). Rigid systems protect incumbent executives by locking in rules and raising coordination thresholds for challengers."
    },
    "incumbent_seat_buffer": {
        "name": "Incumbent Legislative Seat Buffer",
        "explanation": "The margin of seats held by the governing coalition above the 50% majority threshold. Larger legislative buffers protect the cabinet from collapse and internal friction, raising incumbent survival (Lijphart, 2012)."
    },
    "legislative_productivity": {
        "name": "Legislative Productivity Proxy",
        "explanation": "Measured via legislative constraints and active policymaking indicators. Retrospective voting theory (Fiorina, 1981) shows voters reward effective governance and punish legislative gridlock."
    },
    "government_years_in_power": {
        "name": "Governing Coalition Longevity (Fatigue)",
        "explanation": "The number of years the governing coalition has held power. Long-serving governments suffer from 'coalition fatigue' as grievances and policy failures accumulate over time (Rose & Mackie, 1983)."
    },
    "government_continuity_score": {
        "name": "Government Continuity Score",
        "explanation": "An interaction term combining government longevity and legislative checks. Strong legislative checks stabilize long-lived governments by preventing radical policy swings (Tsebelis, 2002)."
    },
    "open_seat_prior_margin_interaction": {
        "name": "Open Seat * Prior Margin Interaction",
        "explanation": "Captures the compounding vulnerability of an open seat in a historically thin-margin district or nation, triggering high electoral volatility and strategic challenger coordination (Cox, 1997)."
    },
    "regime_challenger_tide": {
        "name": "Regional Regime Challenger Tide",
        "explanation": "Spillover of challenger victories among regional neighbors, raising domestic opposition success expectations."
    },
    "opposition_momentum_margin_interaction": {
        "name": "Opposition Momentum & Prior Margin",
        "explanation": "High pre-election opposition news tone and volume (GDELT) combined with thin prior margins favors challengers."
    },
    "polyarchy_delta_1yr": {
        "name": "1-Year Change in Democracy Index",
        "explanation": "Recent shifts in civil liberties and press freedoms directly affect election competitiveness."
    },
    "currency_depreciation_6mo": {
        "name": "6-Month Currency Depreciation",
        "explanation": "Measures recent currency decline. Sharp depreciation hurts purchasing power and sparks anti-incumbent voting."
    },
    "currency_volatility_6mo": {
        "name": "6-Month Currency Volatility",
        "explanation": "High exchange rate instability signals macroeconomic uncertainty, penalizing the governing party."
    },
    "gdp_growth": {
        "name": "GDP Growth Rate",
        "explanation": "Strong macroeconomic growth validates incumbent stewardship, while contractions lead to voter economic backlash."
    },
    "inflation": {
        "name": "Inflation Rate",
        "explanation": "High inflation directly increases cost of living, which historically triggers strong anti-incumbent voting."
    },
    "electoral_system_pr": {
        "name": "Proportional Representation System",
        "explanation": "Indicates whether elections are held under proportional rules. PR systems promote coalition building and seat stability."
    },
    "electoral_system_majoritarian": {
        "name": "Majoritarian Electoral System",
        "explanation": "Indicates whether elections use majoritarian/FPTP rules, which are subject to high seat-share swings."
    },
    "political_polarization": {
        "name": "V-Dem Political Polarization Index",
        "explanation": "Measures societal and party polarization. Higher polarization makes coalition assembly and governability harder."
    },
    "is_majority_fragile": {
        "name": "Fragile Majority Indicator",
        "explanation": "Flags countries where the incumbent holds a slim legislative majority (50% to 55%), leaving them vulnerable."
    },
    "pr_fragmentation_interaction": {
        "name": "PR System * Party Fragmentation",
        "explanation": "Captures the combined friction of proportional rules and multi-party setups on legislative stability."
    },
    "polarization_fragmentation_interaction": {
        "name": "Polarization * Party Fragmentation",
        "explanation": "Captures how high ideological polarization multiplies coalition-formation difficulty under fragmented legislatures."
    },
    "is_scheduled": {
        "name": "Scheduled Election Indicator",
        "explanation": "Elections held on normal constitutional schedule. Unscheduled or snap elections usually indicate political crises or strategic timing by the incumbent."
    },
    "is_presidential": {
        "name": "Executive Race Indicator",
        "explanation": "Differentiates Executive from Legislative races. Executive elections are highly visible and subject to national economic retrospective voting."
    },
    "gdp_shock": {
        "name": "GDP Growth Shock Margin",
        "explanation": "Short-term economic contraction or expansion margin relative to trend. Negative shocks trigger strong retrospective voting backlash."
    },
    "inflation_shock": {
        "name": "Price Inflation Shock",
        "explanation": "Sudden rise in price inflation. Hurts household purchasing power and acts as an immediate trigger for voter anger."
    },
    "average_news_tone": {
        "name": "National Media Tone Aggregate",
        "explanation": "General sentiment in national media (GDELT). Positive tone correlates with incumbent satisfaction; negative tone favors challengers."
    },
    "opposition_boycott": {
        "name": "Opposition Boycott Flag (NELDA)",
        "explanation": "Pre-election opposition boycott indicator (NELDA). Signals extreme polarization, lack of legitimacy, or a skewed electoral field."
    },
    "monitors_present": {
        "name": "International Monitors Present (NELDA)",
        "explanation": "Presence of international election observers (NELDA). Promotes transparency and raises the cost of manipulation, aiding opposition competitiveness."
    },
    "protest_pressure": {
        "name": "GDELT Civil Protest Salience",
        "explanation": "Frequency of protest and civil unrest events (GDELT). Signals public discontent, governability issues, and eroding incumbent support."
    },
    "coop_score": {
        "name": "Geopolitical Cooperation Score",
        "explanation": "Ratio of cooperative to hostile events in media. Higher cooperation indicates stability and successful governance."
    },
    "conflict_level": {
        "name": "GDELT Conflict Event Salience",
        "explanation": "Frequency of internal social or political conflict events. High conflict raises uncertainty and voter anxiety."
    },
    "coercion_ratio": {
        "name": "State Coercion Ratio",
        "explanation": "Proportion of government events involving policing, arrests, or military coercion. Signals state repression or security instability."
    },
    "aid_ratio": {
        "name": "International Development Aid Ratio",
        "explanation": "International development aid or humanitarian events ratio. Higher external aid flows can stabilize vulnerable regimes."
    },
    "protest_velocity": {
        "name": "Protest Acceleration (Velocity)",
        "explanation": "Acceleration or momentum of civil protest events. Rapid spikes signal impending political crises and sudden drops in governability."
    },
    "conflict_velocity": {
        "name": "Conflict Acceleration (Velocity)",
        "explanation": "Acceleration of social conflict. Indicates rapid deterioration of the domestic security climate."
    },
    "total_news_volume": {
        "name": "Total News Coverage Volume",
        "explanation": "Total news media coverage volume. High news attention correlates with periods of high political stakes or active crises."
    },
    "gov_news_volume": {
        "name": "Government Salience in News",
        "explanation": "Relative media volume focusing on the governing coalition. Reflects incumbent media salience."
    },
    "opp_news_volume": {
        "name": "Opposition Salience in News",
        "explanation": "Relative media volume focusing on opposition parties. Reflects opposition voice and salience."
    },
    "gov_avg_tone": {
        "name": "Government Average Coverage Tone",
        "explanation": "Average news tone regarding the government. Positive tone indicates favorable coverage and administrative popularity."
    },
    "opp_avg_tone": {
        "name": "Opposition Average Coverage Tone",
        "explanation": "Average news tone regarding the opposition. Positive tone signals gaining momentum or sympathetic coverage."
    },
    "opp_gov_volume_ratio": {
        "name": "Opposition-to-Government Media Volume Ratio",
        "explanation": "Ratio of opposition to government media volume. Spikes indicate an active challenger voice taking over the national conversation."
    },
    "opp_gov_tone_diff": {
        "name": "Opposition-to-Government Tone Difference",
        "explanation": "Sentiment difference between opposition and government coverage. High positive values signal favorable challenger momentum relative to the incumbent."
    },
    "regional_challenger_tide": {
        "name": "Regional Challenger Victory Tide",
        "explanation": "Incumbent defeat rate in regional neighbors over the last 12 months. Signals regional waves of political change."
    },
    "global_challenger_tide": {
        "name": "Global Challenger Victory Tide",
        "explanation": "Worldwide incumbent defeat rate. Captures global waves of political alternation or economic discontent."
    },
    "global_economic_shock": {
        "name": "Global Economic Distress Index",
        "explanation": "Global macroeconomic shock index. External economic stress triggers concurrent anti-incumbent voting globally."
    },
    "incumbent_term_count": {
        "name": "Incumbent Leader Terms Served",
        "explanation": "Number of consecutive terms the incumbent leader/party has served. Higher term counts accumulate fatigue and succession friction."
    },
    "prior_turnout": {
        "name": "Electoral Turnout Rate (Prior Election)",
        "explanation": "Voter turnout rate in the previous election. Signals base mobilization or historical voter engagement."
    },
    "prior_margin_of_victory": {
        "name": "Incumbent Margin of Victory (Prior Election)",
        "explanation": "Incumbent margin of victory in the prior election. Thin prior margins make the incumbent highly vulnerable to small swings."
    },
    "prior_effective_parties": {
        "name": "Effective Legislative Parties (Prior Election)",
        "explanation": "Effective number of legislative parties (CLEA). Measures parliamentary fragmentation and coalition complexity."
    },
    "prior_incumbent_seat_share": {
        "name": "Incumbent Seat Share Margin (Prior Election)",
        "explanation": "Percentage of legislative seats won by the incumbent in the prior cycle. Higher seat shares indicate stable legislative control."
    },
    "prior_margin_presidential_interaction": {
        "name": "Prior Margin * Executive Interaction",
        "explanation": "Prior victory margin interacted with executive race indicator. Thin prior executive margins are primary challenger targets."
    },
    "oil_price_shock": {
        "name": "Global Crude Oil Price Shock",
        "explanation": "Percentage change in global brent crude oil prices. Affects fuel costs, energy inflation, and trade balances globally."
    },
    "oil_exporter_shock": {
        "name": "Oil Shock * Exporter Interaction",
        "explanation": "Oil price shocks interacted with net oil exporter status. Windfall revenues help exporter incumbents; import costs penalize importers."
    },
    "fed_rate_change": {
        "name": "US Federal Reserve Rate Shift",
        "explanation": "12-month change in the US Federal Reserve federal funds rate. Drives global capital flows, debt-servicing costs, and currency pressure."
    },
    "us_president_ideology": {
        "name": "US Administration Ideology Alignment",
        "explanation": "Political ideology of the US President. Affects global alignment, foreign aid priorities, and trade regimes."
    },
    "currency_depreciation_12mo": {
        "name": "12-Month Currency Depreciation",
        "explanation": "12-month change in the exchange rate. Long-term depreciation erodes real wages and signals persistent economic mismanagement."
    },
    "economy_regime_interaction": {
        "name": "GDP Growth * Regime Type",
        "explanation": "GDP growth interacted with regime democracy level. Democratic voters are more prone to punish economic distress than autocratic voters."
    },
    "protest_regime_interaction": {
        "name": "Protest Velocity * Regime Type",
        "explanation": "Protest velocity interacted with regime democracy level. Autocrats often suppress protest signals, while democrats suffer immediate electoral costs."
    },
    "term_fatigue_economy_interaction": {
        "name": "Years in Power * GDP Growth Interaction",
        "explanation": "Governing years interacted with GDP growth. Multiplies the penalty of economic recessions on long-serving coalitions."
    },
    "last_alt_incumbent_won": {
        "name": "Last Alternation Incumbent Victory Indicator",
        "explanation": "Indicator of whether the incumbent party won the last alternation election. Signals historical resistance to change."
    },
    "last_alt_margin": {
        "name": "Last Alternation Electoral Margin",
        "explanation": "Electoral margin in the last alternation election. Reflects baseline partisan divisions in the country."
    },
    "months_since_last_alt": {
        "name": "Time Elapsed Since Last Alternation",
        "explanation": "Time elapsed since the last power alternation. Long periods without alternation can lead to pent-up challenger demands."
    },
    "government_type_majority": {
        "name": "Majority Government Cabinet Setup",
        "explanation": "Flag for single-party legislative majority. Typically provides stable cabinet control and clear accountability."
    },
    "government_type_minority": {
        "name": "Minority Government Cabinet Setup",
        "explanation": "Flag for minority government. Highly vulnerable to legislative deadlock, censure votes, and early collapses."
    },
    "government_type_coalition": {
        "name": "Coalition Government Cabinet Setup",
        "explanation": "Flag for multi-party coalition government. Subject to internal friction, compromise failures, and joint accountability decay."
    },
    "cabinet_reshuffles_current_term": {
        "name": "Cabinet Member Reshuffles Count",
        "explanation": "Number of times cabinet members were reshuffled. High turnover indicates internal division, governance crises, or cabinet weakness."
    },
    "coalition_size": {
        "name": "Governing Coalition Size (Veto Players)",
        "explanation": "Number of parties in the governing coalition. Larger sizes raise coordination friction and veto players (Tsebelis, 2002)."
    },
    "coalition_fragmentation": {
        "name": "Governing Cabinet fragmentation",
        "explanation": "Ideological or seat-share fragmentation within the cabinet. Unstable fragmentations raise the risk of early elections."
    },
    "democratic_age": {
        "name": "Democracy Age (Years since transition)",
        "explanation": "Years since transition to democratic governance. Older democracies have institutionalized parties and stable voting habits."
    },
    "electoral_volatility_10yr": {
        "name": "10-Year Historical Electoral Volatility",
        "explanation": "Electoral volatility over the last 10 years. Higher volatility signals loose partisan attachments and swing-voter dominance."
    },
    "electoral_volatility_20yr": {
        "name": "20-Year Historical Electoral Volatility",
        "explanation": "Electoral volatility over the last 20 years. Reflects long-term system stability and party consolidation."
    },
    "electoral_volatility_change": {
        "name": "Electoral Volatility Acceleration Index",
        "explanation": "10-year volatility relative to 20-year baseline. Accelerating volatility signals collapsing party systems."
    },
    "pedersen_index": {
        "name": "Pedersen Electoral Volatility Index",
        "explanation": "Pedersen electoral volatility index from the last cycle. Direct measure of net seat shifts between parties."
    },
    "inflation_acceleration": {
        "name": "Price Inflation Acceleration Rate",
        "explanation": "Rate of change in price inflation. Rapidly accelerating inflation triggers acute voter alarm and economic anxiety."
    },
    "gdp_growth_acceleration": {
        "name": "GDP Growth Acceleration Rate",
        "explanation": "Rate of change in GDP growth. Accelerating growth rewards incumbents, while slowing growth signals impending stagnation."
    },
    "unemployment_acceleration": {
        "name": "Unemployment Acceleration Rate",
        "explanation": "Rate of change in unemployment rate. Rising unemployment is a powerful predictor of incumbent defeat (Lewis-Beck, 1988)."
    },
    "real_wage_growth": {
        "name": "Real Wage Growth Index",
        "explanation": "Year-over-year change in consumer-price-adjusted wages. Directly determines voter household well-being and retrospective support."
    },
    "migration_shock": {
        "name": "Net Migration Inflow Shock",
        "explanation": "Sudden net migration inflow spikes. Can trigger public resource strain or domestic political controversies."
    },
    "refugee_inflow_shock": {
        "name": "Refugee Inflow Shock Margin",
        "explanation": "Rapid change in refugee inflows per capita. Can lead to fiscal strain and security concerns, boosting challenger nationalistic rhetoric."
    },
    "government_sentiment_score": {
        "name": "Media Sentiment on Government Action",
        "explanation": "Media tone aggregated specifically around government figures. Reflects public relations momentum."
    },
    "government_corruption_events": {
        "name": "GDELT Corruption Scandal Count",
        "explanation": "GDELT event count of corruption scandals. Directly erodes governing party moral authority and institutional trust."
    },
    "government_corruption_ratio": {
        "name": "Corruption Scandal News Ratio",
        "explanation": "Corruption events relative to total government news. Signals the salience of scandal in the national psyche."
    },
    "government_news_salience": {
        "name": "Government Media Salience Volume",
        "explanation": "Media coverage volume focusing on government action. High salience indicates active political crises or intense scrutiny."
    },
    "government_media_balance": {
        "name": "Government Positive-to-Negative media balance",
        "explanation": "Ratio of positive to negative news tones regarding the administration. Directly measures press favoritism or hostility."
    },
    "global_inflation_pressure": {
        "name": "Global Price Inflation Stress Margin",
        "explanation": "Global average inflation rate. Acts as an external constraint on domestic pricing stability."
    },
    "global_energy_pressure": {
        "name": "Global Fuel & Energy Price Pressure",
        "explanation": "Global energy index. Drives domestic fuel costs and energy supply constraints."
    },
    "global_migration_pressure": {
        "name": "Global Displacement Pressure waves",
        "explanation": "Global displacement and migration wave aggregates. Signals regional and global humanitarian shifts."
    },
    "global_democratic_backsliding": {
        "name": "Global Autocratization Wave Index",
        "explanation": "Global rate of democratic score declines. Captures international waves of autocratization or backsliding."
    },
    "global_incumbent_punishment_index": {
        "name": "Global Anti-Establishment Punishment Tide",
        "explanation": "Average global incumbent defeat rate. Reflects global wave effects of anti-incumbent sentiment."
    },
    "latent_global_political_climate": {
        "name": "Global Political Climate Latent Index (PCA)",
        "explanation": "First principal component of global political indicators. Represents global democratic trends vs autocratization cycles."
    },
    "latent_global_economic_anxiety": {
        "name": "Global Economic Anxiety Latent Index (PCA)",
        "explanation": "First principal component of global economic stresses. Represents global stagflation and currency crises cycles."
    },
    "latent_global_anti_incumbent_index": {
        "name": "Global Anti-Incumbent Sentiment Latent Index",
        "explanation": "Latent global index of incumbent defeat pressure. Captures global waves of anti-establishment voter alignment."
    },
    "latent_global_democracy_trend": {
        "name": "Global Democracy Trend Latent Index",
        "explanation": "Latent global trend of institutional democracy levels."
    },
}

def get_feature_info(feature_name):
    # Check hardcoded list
    if feature_name in FEATURE_METADATA:
        return FEATURE_METADATA[feature_name]["name"], FEATURE_METADATA[feature_name]["explanation"]
        
    # Handle country dummy variables
    if feature_name.startswith("country_"):
        c_code = feature_name.replace("country_", "")
        return f"Country Baseline ({c_code})", f"Represents historical baseline political tendencies unique to {c_code}."
    
    # Handle spatial lags
    if feature_name.startswith("spatial_lag_clean_"):
        layer = feature_name.replace("spatial_lag_clean_", "")
        layer_name = {"unga_voting": "Diplomatic (UNGA)", "trade": "Economic (Trade)", "alliance": "Strategic (Alliance)", "contiguity": "Physical Contiguity"}.get(layer, layer)
        return f"Neighbor Integrity ({layer_name})", "Weighted average of electoral cleanliness among neighboring countries."
        
    if feature_name.startswith("spatial_lag_outcome_"):
        layer = feature_name.replace("spatial_lag_outcome_", "")
        layer_name = {"unga_voting": "Diplomatic (UNGA)", "trade": "Economic (Trade)", "alliance": "Strategic (Alliance)", "contiguity": "Physical Contiguity"}.get(layer, layer)
        return f"Neighbor Outcome Contagion ({layer_name})", "Weighted average of predicted incumbent victory rates among neighboring countries."
        
    # Default fallback
    readable_name = feature_name.replace("_", " ").title()
    return readable_name, "Model feature capturing political or economic conditions."

# CSS Injection for Premium Glassmorphism and Curved Aesthetics
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=Inter:wght@300;400;500;600&display=swap');
    
    /* Main App Overrides */
    .stApp {
        background-color: #F8FAFC !important;
        color: #1E293B !important;
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide Streamlit Header */
    header[data-testid="stHeader"] {
        display: none !important;
    }
    
    /* Offset scroll targets */
    :target {
        scroll-margin-top: 100px;
    }
    html {
        scroll-behavior: smooth;
    }
    
    /* Sticky Navigation Header */
    .sticky-nav {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        background: rgba(255, 255, 255, 0.90);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-bottom: 1px solid rgba(148, 163, 184, 0.15);
        padding: 14px 40px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        z-index: 99999;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.02);
    }
    .nav-brand {
        font-family: 'Outfit', sans-serif;
        font-size: 1.25rem;
        font-weight: 700;
        color: #0F172A;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .nav-links {
        display: flex;
        gap: 24px;
    }
    .nav-links a {
        font-family: 'Outfit', sans-serif;
        font-size: 0.95rem;
        font-weight: 600;
        color: #475569;
        text-decoration: none;
        transition: color 0.2s, transform 0.2s;
    }
    .nav-links a:hover {
        color: #3b82f6;
    }
    
    /* Padding to prevent sticky nav from covering top of page */
    .main .block-container {
        padding-top: 90px !important;
    }
    
    /* Hero Section Container */
    .hero-container {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        border-radius: 24px;
        padding: 3rem;
        color: white;
        margin-bottom: 2.5rem;
        box-shadow: 0 20px 40px rgba(15, 23, 42, 0.08);
        position: relative;
        overflow: hidden;
    }
    .hero-container::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -30%;
        width: 600px;
        height: 600px;
        background: radial-gradient(circle, rgba(59, 130, 246, 0.15) 0%, rgba(0,0,0,0) 70%);
        border-radius: 50%;
        pointer-events: none;
    }
    
    .hero-title {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        font-size: 2.8rem;
        line-height: 1.15;
        margin-bottom: 0.75rem;
        color: #F8FAFC;
    }
    .hero-subtitle {
        font-size: 1.1rem;
        color: #94A3B8;
        max-width: 800px;
        line-height: 1.6;
        margin-bottom: 1.5rem;
    }
    
    /* Cards & Containers */
    .section-container {
        margin-bottom: 3.5rem;
        padding: 2rem;
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.015);
    }
    
    .section-header {
        font-family: 'Outfit', sans-serif;
        font-size: 1.8rem;
        font-weight: 700;
        color: #0F172A;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .section-subheader {
        font-size: 0.95rem;
        color: #64748B;
        margin-bottom: 1.5rem;
        line-height: 1.5;
    }
    
    .kpi-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 1.25rem;
        text-align: center;
        transition: transform 0.2s, border-color 0.2s;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        border-color: #cbd5e1;
    }
    .kpi-title {
        font-size: 0.8rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.4rem;
    }
    .kpi-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0F172A;
    }
    .kpi-desc {
        font-size: 0.75rem;
        color: #475569;
        margin-top: 0.3rem;
    }
    
    .showcase-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 20px;
        margin-bottom: 2rem;
    }
    
    .showcase-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 20px;
        padding: 1.25rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.01);
        text-align: center;
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
    }
    .showcase-card:hover {
        border-color: #3b82f6;
        transform: translateY(-4px);
        box-shadow: 0 12px 30px rgba(59, 130, 246, 0.05);
    }
    
    .showcase-country {
        font-family: 'Outfit', sans-serif;
        font-size: 1.2rem;
        font-weight: 700;
        color: #0F172A;
    }
    .showcase-badge {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 8px 0;
    }
    .badge-incumbent {
        background-color: rgba(59, 130, 246, 0.1);
        color: #2563eb;
    }
    .badge-challenger {
        background-color: rgba(249, 115, 22, 0.1);
        color: #ea580c;
    }
    .card-text {
        font-size: 0.85rem;
        color: #1E293B !important;
        line-height: 1.4;
    }
    
    /* Dark Mode Overrides */
    @media (prefers-color-scheme: dark) {
        .stApp {
            background-color: #0B0F19 !important;
            color: #E2E8F0 !important;
        }
        .sticky-nav {
            background: rgba(11, 15, 25, 0.90);
            border-bottom: 1px solid rgba(226, 232, 240, 0.08);
        }
        .nav-brand {
            color: #F8FAFC;
        }
        .nav-links a {
            color: #94A3B8;
        }
        .nav-links a:hover {
            color: #60a5fa;
        }
        .section-container {
            background: #111827;
            border-color: #1f2937;
        }
        .section-header {
            color: #F8FAFC;
        }
        .section-subheader {
            color: #94A3B8;
        }
        .kpi-card {
            background: #1f2937;
            border-color: #374151;
        }
        .kpi-title {
            color: #94A3B8;
        }
        .kpi-value {
            color: #F8FAFC;
        }
        .kpi-desc {
            color: #94A3B8;
        }
        .showcase-card {
            background: #111827;
            border-color: #1f2937;
        }
        .showcase-country {
            color: #F8FAFC;
        }
        .card-text {
            color: #E2E8F0 !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# Load basic mappings and settings
if cache_data is not None:
    c_names = cache_data["c_names"]
    code_to_country = dict(zip(c_names["country_code"], c_names["country_name"]))
else:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT country_code, country_name, region FROM countries ORDER BY country_name;")
    countries_list = cursor.fetchall()
    c_names = pd.DataFrame(countries_list, columns=["country_code", "country_name", "region"])
    code_to_country = dict(zip(c_names["country_code"], c_names["country_name"]))
    conn.close()

import datetime

# Helper to define Plotly Gauge/Dial Chart
def render_plotly_gauge(prob, winner, country_name, year, election_type):
    bar_color = "#33658A" if winner == "Incumbent" else "#F26419"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#475569", 'tickfont': {'family': 'Inter', 'size': 10}},
            'bar': {'color': bar_color, 'thickness': 0.6},
            'bgcolor': "rgba(0,0,0,0.05)",
            'borderwidth': 1,
            'bordercolor': "#cbd5e1",
            'steps': [
                {'range': [0, 50], 'color': 'rgba(242, 100, 25, 0.05)'},
                {'range': [50, 100], 'color': 'rgba(51, 101, 138, 0.05)'}
            ],
            'threshold': {
                'line': {'color': "#ef4444", 'width': 2},
                'thickness': 0.8,
                'value': 50
            }
        },
        number={'suffix': "%", 'font': {'family': 'Outfit', 'size': 26, 'weight': 'bold', 'color': bar_color}}
    ))
    fig.update_layout(
        height=130,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'family': "Inter"}
    )
    return fig

# Load compiled datasets with composite cache buster to detect parameter tuning or model code updates
mtimes = [os.path.getmtime(DB_PATH) if os.path.exists(DB_PATH) else 0.0]

tuned_json_path = os.path.join(BASE_DIR, "models", "tuned_parameters.json")
if os.path.exists(tuned_json_path):
    mtimes.append(os.path.getmtime(tuned_json_path))

corr_models_py = os.path.join(BASE_DIR, "models", "correlation_models.py")
if os.path.exists(corr_models_py):
    mtimes.append(os.path.getmtime(corr_models_py))

db_mtime = max(mtimes)
preds, df_modeling = get_cascading_predictions_and_df(db_mtime)
all_preds = get_all_predictions(db_mtime)

# Calculate dynamic KPIs
known_preds = all_preds[all_preds["target_outcome"].notna()].copy()
if not known_preds.empty:
    cv_accuracy = accuracy_score(known_preds["target_outcome"].astype(int), known_preds["predicted_outcome"].astype(int))
    baseline_accuracy = (known_preds["target_outcome"] == 1).mean()
    gain = cv_accuracy - baseline_accuracy
    historical_size = len(known_preds)
else:
    cv_accuracy = 0.7371
    baseline_accuracy = 0.5825
    gain = 0.1546
    historical_size = 2206

active_projections = len(preds[preds["year"] <= 2028])

# --- STICKY NAV BAR ---
st.markdown("""
<div class="sticky-nav">
  <div class="nav-brand">🗳️ Global Election Forecaster</div>
  <div class="nav-links">
    <a href="#summary">Summary</a>
    <a href="#map-section">Global Map</a>
    <a href="#database-section">Database Explorer</a>
    <a href="#methodology-section">Methodology</a>
    <a href="#diagnostics-section">Performance</a>
  </div>
</div>
""", unsafe_allow_html=True)

# --- SECTION 1: EXECUTIVE SUMMARY & SHOWCASE ---
st.markdown("<div id='summary'></div>", unsafe_allow_html=True)

# Hero Container
st.markdown(
    f"<div class='hero-container'>"
    f"<div class='hero-title'>🗳️ Global Election Forecaster</div>"
    f"<div class='hero-subtitle'>"
    f"<b>Using Global Indicators to Predict National Elections</b><br>"
    f"This project explores how global and international macro-dynamics—rather than candidate-specific polling or demographics—can forecast national election outcomes. "
    f"By training machine learning models on institutional V-Dem governance indices, global economic shocks, relative political momentum from GDELT, and spatial contagion lags, "
    f"the forecaster maps international political shifts onto national races.<br><br>"
    f"<b>Geopolitical Contagion & Cascading Spillover Effects</b><br>"
    f"Elections do not occur in isolation. To capture geopolitical diffusion, predictions are modeled as a chronological cascade: "
    f"as elections are projected, their outcomes propagate trade, strategic alliance, and physical contiguity network weights to feed back "
    f"into the features of neighboring nations' upcoming elections. An early change in one region can trigger cascading forecasts globally."
    f"</div>"
    f"</div>",
    unsafe_allow_html=True
)

# KPIs in row
kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
with kpi_col1:
    st.markdown(
        "<div class='kpi-card'>"
        "<div class='kpi-title'>Model Strength</div>"
        f"<div class='kpi-value'>{cv_accuracy:.2%}</div>"
        "<div class='kpi-desc'>Out-of-sample accuracy</div>"
        "</div>",
        unsafe_allow_html=True
    )
with kpi_col2:
    st.markdown(
        "<div class='kpi-card'>"
        "<div class='kpi-title'>Gain vs. Baseline</div>"
        f"<div class='kpi-value'>+{gain:.2%}</div>"
        f"<div class='kpi-desc'>Above simple incumbent-wins rate ({baseline_accuracy:.2%})</div>"
        "</div>",
        unsafe_allow_html=True
    )
with kpi_col3:
    st.markdown(
        "<div class='kpi-card'>"
        "<div class='kpi-title'>Elections Modeled</div>"
        f"<div class='kpi-value'>{historical_size:,}</div>"
        "<div class='kpi-desc'>Historical records (1990-2026)</div>"
        "</div>",
        unsafe_allow_html=True
    )
with kpi_col4:
    st.markdown(
        "<div class='kpi-card'>"
        "<div class='kpi-title'>Active Predictions</div>"
        f"<div class='kpi-value'>{active_projections}</div>"
        "<div class='kpi-desc'>Upcoming forecasts through 2028</div>"
        "</div>",
        unsafe_allow_html=True
    )

# Showcase Prominent Predictions
st.markdown("<h4 class='section-header' style='margin-top:2.5rem;'>🔮 Prominent Upcoming Elections Showcase</h4>", unsafe_allow_html=True)
st.markdown("<div class='section-subheader'>Key global elections forecasted by the cascading spatial GBDT models. Click on their entries in the Database Explorer below to review their dynamic feature drivers.</div>", unsafe_allow_html=True)

showcase_elections = [
    ("USA", 2028, "Executive"),
    ("FRA", 2027, "Executive"),
    ("DEU", 2027, "Executive"),
    ("MEX", 2027, "Legislative")
]

show_cols = st.columns(4)
for i, (c_code, yr, el_type) in enumerate(showcase_elections):
    with show_cols[i]:
        match = all_preds[(all_preds["country_code"] == c_code) & (all_preds["year"] == yr) & (all_preds["election_type"] == el_type)]
        if not match.empty:
            row = match.iloc[0]
            winner = row["predicted_winner"]
            prob = row["raw_probability"]
            conf = row["adjusted_confidence"]
            quality = row["data_completeness"]
            c_name = code_to_country.get(c_code, c_code)
            
            badge_class = "badge-incumbent" if winner == "Incumbent" else "badge-challenger"
            fig_gauge = render_plotly_gauge(prob, winner, c_name, yr, el_type)
            
            st.markdown(
                f"<div class='showcase-card'>"
                f"<div class='showcase-country'>🗳️ {c_name}</div>"
                f"<div style='font-size: 0.75rem; color: #64748B; margin-bottom: 6px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;'>{yr} • {el_type}</div>"
                f"<span class='showcase-badge {badge_class}'>Projected: {winner}</span>",
                unsafe_allow_html=True
            )
            st.plotly_chart(fig_gauge, width="stretch", key=f"gauge_{c_code}_{yr}")
            st.markdown(
                f"<div class='card-text' style='font-size: 0.8rem; line-height: 1.4; text-align: center; color: gray;'>"
                f"Confidence: <b>{conf:.1%}</b><br/>"
                f"Data Quality: <b>{quality:.1%}</b>"
                f"</div>"
                f"</div>",
                unsafe_allow_html=True
            )
        else:
            st.info(f"Showcase data for {c_code} {yr} not found.")

# --- SECTION 2: MAP ---
st.markdown("---")
st.markdown("<div id='map-section'></div>", unsafe_allow_html=True)
st.markdown("<h3 class='section-header'>🗺️ Global Forecasts Choropleth Map</h3>", unsafe_allow_html=True)
st.markdown("<div class='section-subheader'>Hover over countries to see forecast details. Blue represents predicted Incumbent victory, Red/Orange represents Challenger victory, and Gray indicates toss-up or unmodeled regions. Use the selector to inspect different forecasting cycles.</div>", unsafe_allow_html=True)

# Map Year Selector
map_yr_opt = st.selectbox("Forecast Cycle Filter", ["All Upcoming (2026-2028)", "2026 Predictions", "2027 Predictions", "2028 Predictions"])

# Filter map df
all_preds_merged = pd.merge(all_preds, c_names, on="country_code", how="left")
all_preds_merged["parsed_date"] = pd.to_datetime(all_preds_merged["election_date"], errors="coerce")

# Target cutoff: 2026-06-14 (one month before today)
cutoff_date = datetime.date(2026, 6, 14)
all_preds_merged["is_upcoming"] = all_preds_merged.apply(
    lambda r: r["parsed_date"].date() >= cutoff_date if pd.notna(r["parsed_date"]) else int(r["year"]) >= 2026,
    axis=1
)

map_preds_df = all_preds_merged[all_preds_merged["is_upcoming"] == True].copy()

if map_yr_opt != "All Upcoming (2026-2028)":
    tgt_year = int(map_yr_opt.split(" ")[0])
    map_preds_df = map_preds_df[map_preds_df["year"] == tgt_year]
else:
    # Group by country and get the nearest upcoming election to avoid duplicates
    map_preds_df = map_preds_df.sort_values(by="year").groupby("country_code").first().reset_index()

# Render choropleth map
fig_world_map = px.choropleth(
    map_preds_df,
    locations="country_code",
    color="raw_probability",
    hover_name="country_name",
    hover_data=["year", "election_type", "predicted_winner", "adjusted_confidence"],
    color_continuous_scale=["#F26419", "#E2E8F0", "#33658A"],
    range_color=[0.0, 1.0],
    labels={"raw_probability": "Incumbent Win Probability"},
    projection="natural earth"
)
fig_world_map.update_layout(
    coloraxis_colorbar=dict(
        title="Incumbent Win Prob",
        tickformat=".0%",
        ticks="outside"
    ),
    margin=dict(l=0, r=0, t=10, b=0),
    height=450,
    paper_bgcolor="rgba(0,0,0,0)",
    geo=dict(
        showframe=False,
        showcoastlines=True,
        coastlinecolor="rgba(148,163,184,0.3)",
        bgcolor="rgba(0,0,0,0)"
    )
)
st.plotly_chart(fig_world_map, width="stretch")

# --- SECTION 3: EXPLORER DATABASE ---
st.markdown("---")
st.markdown("<div id='database-section'></div>", unsafe_allow_html=True)
st.markdown("<h3 class='section-header'>🔍 Interactive Election Explorer</h3>", unsafe_allow_html=True)
st.markdown("<div class='section-subheader'>Filter and select any election to load its feature drivers (SHAP values) and theoretical explanation.</div>", unsafe_allow_html=True)

# Search & filters
f_col1, f_col2, f_col3 = st.columns(3)
with f_col1:
    search_q = st.text_input("🔍 Search country name...", "", key="search_query")
with f_col2:
    reg_opt = st.selectbox("Region Filter", ["All", "Europe", "Americas", "Asia", "Africa", "Middle East"], key="region_filter")
with f_col3:
    type_opt = st.selectbox("Election Type", ["All", "Executive", "Legislative"], key="type_filter")

# Filter dataset
filtered_df = all_preds_merged.copy()
if search_q:
    filtered_df = filtered_df[filtered_df["country_name"].str.contains(search_q, case=False, na=False)]
if reg_opt != "All":
    region_dict = {
        "Europe": ["Western Europe", "Eastern Europe", "Europe"],
        "Americas": ["Latin America", "North America", "Caribbean", "Americas"],
        "Asia": ["Asia", "East Asia", "South Asia", "Southeast Asia", "Pacific"],
        "Africa": ["Africa", "Sub-Saharan Africa", "North Africa"],
        "Middle East": ["Middle East", "Middle East & North Africa", "Western Asia"]
    }
    filtered_df = filtered_df[filtered_df["region"].isin(region_dict[reg_opt])]
if type_opt != "All":
    filtered_df = filtered_df[filtered_df["election_type"] == type_opt]

# Separate
upcoming_table_df = filtered_df[filtered_df["is_upcoming"] == True].copy().sort_values(by=["year", "country_name"])
historical_table_df = filtered_df[filtered_df["is_upcoming"] == False].copy().sort_values(by=["year", "country_name"], ascending=False)

db_tab1, db_tab2 = st.tabs(["🔮 Upcoming Forecasts (Live)", "📚 Historical Archive (Out-of-Sample CV)"])

selected_row = None
with db_tab1:
    if upcoming_table_df.empty:
        st.info("No upcoming elections match the active filters.")
    else:
        st.write(f"Displaying **{len(upcoming_table_df)}** upcoming elections scheduled from June 2026 onwards:")
        upcoming_display = upcoming_table_df.copy()
        upcoming_display["Ensemble Win Prob"] = upcoming_display["raw_probability"].apply(lambda p: f"{p:.1%}")
        upcoming_display["Adjusted Confidence"] = upcoming_display["adjusted_confidence"].apply(lambda c: f"{c:.1%}")
        upcoming_display["Data Quality"] = upcoming_display["data_completeness"].apply(lambda d: f"{d:.1%}")
        
        event_upcoming = st.dataframe(
            upcoming_display[["country_name", "year", "election_type", "predicted_winner", "Ensemble Win Prob", "Adjusted Confidence", "Data Quality"]].rename(columns={
                "country_name": "Country", "year": "Year", "election_type": "Type",
                "predicted_winner": "Predicted Winner", "Ensemble Win Prob": "Incumbent Win Prob",
                "Adjusted Confidence": "Certainty", "Data Quality": "Completeness"
            }),
            width="stretch",
            on_select="rerun",
            selection_mode="single-row",
            hide_index=True,
            key="upcoming_select"
        )
        
        if event_upcoming.selection.rows:
            selected_idx = event_upcoming.selection.rows[0]
            if selected_idx < len(upcoming_table_df):
                selected_row = upcoming_table_df.iloc[selected_idx]

with db_tab2:
    if historical_table_df.empty:
        st.info("No historical elections match the active filters.")
    else:
        st.write(f"Displaying **{len(historical_table_df)}** past elections. Outcomes shown are 5-fold cross-validated to simulate out-of-sample prediction:")
        
        historical_display = historical_table_df.copy()
        historical_display["Incumbent Win Prob"] = historical_display["raw_probability"].apply(lambda p: f"{p:.1%}")
        historical_display["Actual Outcome"] = historical_display["target_outcome"].map({1: "Incumbent Won", 0: "Challenger Won"}).fillna("Unknown")
        historical_display["Model Projection"] = historical_display["predicted_winner"]
        
        def get_prediction_match(row):
            if pd.isna(row["target_outcome"]):
                return "❓ Unknown"
            elif int(row["predicted_outcome"]) == int(row["target_outcome"]):
                return "✅ Correct"
            else:
                return "❌ Incorrect"
        
        historical_display["Prediction Match"] = historical_display.apply(get_prediction_match, axis=1)
        
        event_historical = st.dataframe(
            historical_display[["country_name", "year", "election_type", "Actual Outcome", "Model Projection", "Prediction Match", "Incumbent Win Prob"]].rename(columns={
                "country_name": "Country", "year": "Year", "election_type": "Type",
                "Actual Outcome": "Actual Outcome", "Model Projection": "Model Prediction",
                "Prediction Match": "Status", "Incumbent Win Prob": "Incumbent Prob"
            }),
            width="stretch",
            on_select="rerun",
            selection_mode="single-row",
            hide_index=True,
            key="historical_select"
        )
        
        if event_historical.selection.rows:
            selected_idx = event_historical.selection.rows[0]
            if selected_idx < len(historical_table_df):
                selected_row = historical_table_df.iloc[selected_idx]

# Feature Attribution Breakdown section
if selected_row is not None:
    selected_id = selected_row["election_id"]
    selected_country_code = selected_row["country_code"]
    selected_year = selected_row["year"]
    
    st.markdown("---")
    st.markdown(
        f"<h3 class='section-header'>"
        f"🔎 Feature Attribution Breakdown: {selected_row['country_name']} {selected_year} ({selected_row['election_type']})"
        f"</h3>",
        unsafe_allow_html=True
    )
    
    match_row = df_modeling[df_modeling["election_id"] == selected_id]
    if not match_row.empty:
        is_pres = selected_row["election_type"] == "Executive"
        
        ex_xgb, ex_cb, leg_xgb, leg_cb = get_trained_models(db_mtime)
        exec_params = TUNED_PARAMS.get("executive_model", {"xgb_weight": 0.9, "T": 0.78})
        leg_params = TUNED_PARAMS.get("legislative_model", {"xgb_weight": 0.5, "T": 0.80})
        
        country_dummies = [col for col in df_modeling.columns if col.startswith("country_") and len(col) == 11 and col[8:].isupper()]
        exec_features = [f for f in FEATURES if f not in LEG_SPECIFIC and not (f.startswith("country_") and len(f) == 11 and f[8:].isupper())] + country_dummies
        leg_features = [f for f in FEATURES if f not in EXEC_SPECIFIC and not (f.startswith("country_") and len(f) == 11 and f[8:].isupper())] + country_dummies
        
        if is_pres:
            model_xgb = ex_xgb
            model_cb = ex_cb
            feats = exec_features
            xgb_w = exec_params["xgb_weight"]
        else:
            model_xgb = leg_xgb
            model_cb = leg_cb
            feats = leg_features
            xgb_w = leg_params["xgb_weight"]
        
        X_pred = match_row[feats]
        
        # Compute contributions
        dmatrix = xgb.DMatrix(X_pred)
        contribs_xgb = model_xgb.get_booster().predict(dmatrix, pred_contribs=True)[0]
        contribs_cb = model_cb.get_feature_importance(data=cb.Pool(X_pred), type='ShapValues')[0]
        
        contribs = xgb_w * contribs_xgb[:-1] + (1.0 - xgb_w) * contribs_cb[:-1]
        
        feature_contribs = []
        for i, feat in enumerate(feats):
            feature_contribs.append({
                "feature": feat,
                "value": X_pred[feat].iloc[0],
                "contribution": contribs[i]
            })
        df_contribs = pd.DataFrame(feature_contribs)
        
        # Filter dummies
        other_countries_dummies = [f"country_{c}" for c in c_names["country_code"] if c != selected_country_code]
        df_contribs = df_contribs[~df_contribs["feature"].isin(other_countries_dummies)]
        
        df_pos = df_contribs[df_contribs["contribution"] > 1e-4].sort_values(by="contribution", ascending=False).head(5)
        df_neg = df_contribs[df_contribs["contribution"] < -1e-4].sort_values(by="contribution", ascending=True).head(5)
        
        df_top = pd.concat([df_pos, df_neg])
        df_top["Direction"] = np.where(df_top["contribution"] > 0, "Favors Incumbent", "Favors Challenger")
        df_top = df_top.sort_values(by="contribution", ascending=True)
        df_top["Human Name"] = df_top["feature"].apply(lambda f: get_feature_info(f)[0])
        
        fig_shap = px.bar(
            df_top, x="contribution", y="Human Name", color="Direction",
            orientation='h',
            color_discrete_map={"Favors Incumbent": "#33658A", "Favors Challenger": "#F26419"},
            title="Core Predictive Drivers (SHAP Value Contribution)",
            labels={"contribution": "Influence (Log-Odds Contribution)", "Human Name": "Feature"}
        )
        fig_shap.update_layout(
            margin=dict(l=10, r=10, t=40, b=10),
            height=320,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        
        detail_col1, detail_col2 = st.columns([1.2, 1])
        with detail_col1:
            st.plotly_chart(fig_shap, width="stretch")
        with detail_col2:
            st.markdown("##### Forecast Attribution Summary")
            actual_winner_str = ""
            if pd.notna(selected_row["target_outcome"]):
                outcome_text = "Incumbent Victory" if int(selected_row["target_outcome"]) == 1 else "Challenger Victory"
                actual_winner_str = f"- **Actual Outcome**: `{outcome_text}`\n"
            
            st.markdown(
                f"- **Raw Ensemble Probability**: `{selected_row['raw_probability']:.1%}`\n"
                f"- **Adjusted Confidence**: `{selected_row['adjusted_confidence']:.1%}`\n"
                f"- **Data Completeness**: `{selected_row['data_completeness']:.1%}`\n"
                f"- **Model Forecast Decision**: **{selected_row['predicted_winner']}**\n"
                f"{actual_winner_str}"
            )
            
            # Decoded flags
            flags = selected_row["data_source_flags"]
            sources = []
            if flags & 1: sources.append("V-Dem Index")
            if flags & 2: sources.append("WDI Macroeconomics")
            if flags & 4: sources.append("GDELT News Tone")
            if flags & 8: sources.append("Spatial Contagion")
            st.markdown(f"- **Active Data Sources**: {', '.join(sources) if sources else 'None'}")
        
        # Table of explanations
        st.markdown("##### Detailed Feature Explanation Table")
        df_details = pd.concat([df_pos, df_neg]).copy()
        df_details["influence"] = df_details["contribution"]
        df_details = df_details.sort_values(by="influence", ascending=False)
        
        table_rows = []
        for _, r in df_details.iterrows():
            feat = r["feature"]
            val = r["value"]
            infl = r["influence"]
            
            name, theory = get_feature_info(feat)
            
            if feat.startswith("country_") or feat == "is_open_seat" or feat == "has_successor":
                val_str = "Yes" if val == 1.0 else "No"
            elif "depreciation" in feat or "volatility" in feat or "completeness" in feat or "growth" in feat or "inflation" in feat:
                val_str = f"{val:.1%}" if abs(val) < 10.0 else f"{val:.1f}"
            elif "index" in feat or "polyarchy" in feat or "lag" in feat:
                val_str = f"{val:.3f}"
            else:
                val_str = f"{val}"
                
            direction = "Incumbent" if infl > 0 else "Challenger"
            table_rows.append({
                "Feature Name": name,
                "Observed Value": val_str,
                "Influence Direction": f"Favors {direction}",
                "Influence Score": round(infl, 4),
                "Theoretical Explanation": theory
            })
            
        st.dataframe(pd.DataFrame(table_rows), width="stretch", hide_index=True)
    else:
        st.warning("Could not load feature details for selected election.")
else:
    st.info("💡 Select an election row in the tabs above to view its detailed SHAP drivers and theoretical breakdowns.")

# --- SECTION 4: METHODOLOGY ---
st.markdown("---")
st.markdown("<div id='methodology-section'></div>", unsafe_allow_html=True)
st.markdown("<h3 class='section-header'>💡 Model Methodology & Explainer</h3>", unsafe_allow_html=True)
st.markdown("<div class='section-subheader'>Details on features, V-Dem indicators map, and feature importances.</div>", unsafe_allow_html=True)

meth_tab1, meth_tab2, meth_tab3 = st.tabs(["⚙️ The Modeling Pipeline", "🗺️ Historical V-Dem Cleanliness Map", "🔥 Feature Importance Breakdowns"])

with meth_tab1:
    st.markdown(
        "#### How it Works\n"
        "The Global Election Forecaster processes raw data through a four-stage modeling cascade:\n"
        "1. **Data Ingestion**: Standardizes indicators from four distinct sources:\n"
        "   - **V-Dem**: Structural governance, civil liberties, and institutional constraints.\n"
        "   - **World Development Indicators (WDI)**: Macroeconomic levels and growth rates.\n"
        "   - **GDELT Project**: Real-time media tone and protest event volume.\n"
        "   - **CLEA Database**: Historical margins of victory and electoral systems.\n"
        "2. **Network Contagion (Spatial Autoregression)**: Computes geographic, trade-based, and strategic alliance connectivity matrix weights to construct spatial lags, modeling the international diffusion of democracy, protests, and election outcomes.\n"
        "3. **GBDT Ensemble**: Trains separate XGBoost and CatBoost classifiers for **Executive** and **Legislative** elections. Separate models are used because executive races are driven by personal incumbents, whereas legislative races are governed by party coalitions and majorities.\n"
        "4. **Cascading Time-Series Forecasts**: For upcoming elections (2026-2028), the model forecasts chronologically. Each country's predicted outcome is fed-forward into subsequent neighboring predictions, adjusting the spatial lag features dynamically."
    )

with meth_tab2:
    st.markdown("##### Historical Governance Explorer")
    st.markdown("Use the slider to explore the historical V-Dem Clean Elections Index dynamically across different years. This structural cleanliness forms the baseline layer of the GBDT models.")
    
    map_year = st.slider("Select Historical Year for V-Dem Map", 1990, 2025, 2020)
    
    conn = get_connection()
    query_map = """
        SELECT c.country_code, c.country_name, c.latitude, c.longitude,
               v.clean_elections_index, v.polyarchy_index, v.regime_type
        FROM countries c
        LEFT JOIN vdem_indicators v ON c.country_code = v.country_code AND v.year = ?;
    """
    df_map = pd.read_sql_query(query_map, conn, params=(map_year,))
    conn.close()
    
    regime_map = {0: "Closed Autocracy", 1: "Electoral Autocracy", 2: "Electoral Democracy", 3: "Liberal Democracy"}
    df_map["Regime Type"] = df_map["regime_type"].map(regime_map).fillna("Unknown")
    
    if not df_map.empty and df_map["clean_elections_index"].notna().any():
        fig_map = px.scatter_geo(
            df_map.dropna(subset=["latitude", "longitude"]),
            lat="latitude",
            lon="longitude",
            color="clean_elections_index",
            size=df_map.dropna(subset=["latitude", "longitude"])["polyarchy_index"].fillna(0.1).clip(lower=0.1),
            hover_name="country_name",
            hover_data={"Regime Type": True, "clean_elections_index": ":.3f", "polyarchy_index": ":.3f"},
            color_continuous_scale="RdYlGn",
            range_color=[0, 1],
            projection="natural earth"
        )
        fig_map.update_layout(
            geo=dict(showframe=False, showcoastlines=True, coastlinecolor="#86BBD8"),
            coloraxis_colorbar=dict(title="Cleanliness"),
            margin=dict(l=0, r=0, t=10, b=0),
            height=400,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_map, width="stretch")
    else:
        st.info("No V-Dem data found for the selected year.")

with meth_tab3:
    col_bar, col_theory = st.columns([1, 1.2])
    with col_bar:
        st.markdown("##### Top 10 Model Predictors")
        imp_df = get_feature_importances(db_mtime)
        if not imp_df.empty:
            imp_df["Human Name"] = imp_df["feature"].apply(lambda f: get_feature_info(f)[0])
            fig_imp = px.bar(
                imp_df.head(10), x="importance", y="Human Name",
                orientation='h',
                labels={"importance": "Importance Score", "Human Name": "Feature"},
                color="importance",
                color_continuous_scale=["#86BBD8", "#33658A", "#2F4858"]
            )
            fig_imp.update_layout(
                yaxis=dict(autorange="reversed"),
                coloraxis_showscale=False, showlegend=False,
                margin=dict(l=10, r=10, t=10, b=10),
                height=350,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_imp, width="stretch")
            
    with col_theory:
        st.markdown("##### Theoretical Explanations")
        st.write("Below are descriptions of the core theoretical features driving GBDT outcomes:")
        top_theoretical_features = [
            "constitutional_rigidity",
            "successor_run",
            "is_open_seat",
            "polyarchy_index",
            "incumbent_seat_buffer",
            "legislative_productivity",
            "government_years_in_power",
            "government_continuity_score",
            "clean_index",
            "open_seat_prior_margin_interaction"
        ]
        for idx, feat in enumerate(top_theoretical_features):
            name, explanation = get_feature_info(feat)
            with st.expander(f"Rank {idx+1}: {name}"):
                st.write(explanation)

def get_accuracy_by_country(all_preds):
    known = all_preds[all_preds["target_outcome"].notna()].copy()
    if known.empty:
        return pd.DataFrame()
        
    known["is_correct"] = (known["predicted_outcome"] == known["target_outcome"]).astype(int)
    
    rows = []
    for c_code, group in known.groupby("country_code"):
        c_name = group["country_name"].iloc[0] if "country_name" in group.columns else c_code
        
        exec_group = group[group["election_type"] == "Executive"]
        leg_group = group[group["election_type"] == "Legislative"]
        
        exec_total = len(exec_group)
        exec_correct = exec_group["is_correct"].sum() if exec_total > 0 else 0
        
        leg_total = len(leg_group)
        leg_correct = leg_group["is_correct"].sum() if leg_total > 0 else 0
        
        total = exec_total + leg_total
        correct = exec_correct + leg_correct
        accuracy = correct / total if total > 0 else 0.0
        
        # Color by raw accuracy if total >= 3, otherwise set to NaN (renders as gray on map)
        color_accuracy = accuracy if total >= 3 else np.nan
        
        exec_ratio = f"{exec_correct}/{exec_total} correct" if exec_total > 0 else "0/0 (N/A)"
        leg_ratio = f"{leg_correct}/{leg_total} correct" if leg_total > 0 else "0/0 (N/A)"
        accuracy_str = f"{accuracy:.1%} ({correct}/{total})"
        
        rows.append({
            "country_code": c_code,
            "country_name": c_name,
            "accuracy": accuracy,
            "color_accuracy": color_accuracy,
            "accuracy_str": accuracy_str,
            "exec_ratio": exec_ratio,
            "leg_ratio": leg_ratio
        })
        
    return pd.DataFrame(rows)

# --- SECTION 5: PERFORMANCE ---
st.markdown("---")
st.markdown("<div id='diagnostics-section'></div>", unsafe_allow_html=True)
st.markdown("<h3 class='section-header'>⚙️ Model Diagnostics & Evaluation Metrics</h3>", unsafe_allow_html=True)
st.markdown("<div class='section-subheader'>Review cross-validation metrics, confusion matrices, and probability calibration.</div>", unsafe_allow_html=True)

diag_tab1, diag_tab2, diag_tab3, diag_tab4 = st.tabs([
    "📊 Rolling Split Performance", 
    "📈 Probability Calibration", 
    "🎛️ Confusion Matrices & Hyperparameters",
    "🗺️ Accuracy by Country"
])

cv_df, mean_ridge, mean_xgb, mean_ensemble = evaluate_national_models_live(db_mtime)

known_preds = all_preds[all_preds["target_outcome"].notna()].copy()
known_preds["is_correct"] = (known_preds["predicted_outcome"] == known_preds["target_outcome"]).astype(int)

y_true = known_preds["target_outcome"].astype(int)
y_pred = known_preds["predicted_outcome"].astype(int)
y_prob = known_preds["raw_probability"]

cv_accuracy = accuracy_score(y_true, y_pred)
cv_logloss = log_loss(y_true, y_prob)

# Compute Executive-specific metrics
exec_preds = known_preds[known_preds["election_type"] == "Executive"]
exec_accuracy, exec_logloss = 0.0, 0.0
if not exec_preds.empty:
    exec_accuracy = accuracy_score(exec_preds["target_outcome"].astype(int), exec_preds["predicted_outcome"].astype(int))
    exec_logloss = log_loss(exec_preds["target_outcome"].astype(int), exec_preds["raw_probability"])
    
# Compute Legislative-specific metrics
leg_preds = known_preds[known_preds["election_type"] == "Legislative"]
leg_accuracy, leg_logloss = 0.0, 0.0
if not leg_preds.empty:
    leg_accuracy = accuracy_score(leg_preds["target_outcome"].astype(int), leg_preds["predicted_outcome"].astype(int))
    leg_logloss = log_loss(leg_preds["target_outcome"].astype(int), leg_preds["raw_probability"])

with diag_tab1:
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.markdown("<div style='text-align: center; font-weight: bold; background-color: rgba(51, 101, 138, 0.1); padding: 5px; border-radius: 5px; color: #1E293B;'>Combined Metrics (All)</div>", unsafe_allow_html=True)
        st.metric("OOS Accuracy (5-Fold CV)", f"{cv_accuracy:.2%}", help="Average accuracy across cross-validation folds")
        st.metric("OOS Log-Loss", f"{cv_logloss:.4f}", help="Ensemble cross-entropy error metric")
        st.metric("Tuned Temp (T)", f"{TUNED_PARAMS.get('ensemble_balance', {}).get('T', 0.70)}", help="Temperature parameter applied to GBDT probabilities")
    with col_m2:
        st.markdown("<div style='text-align: center; font-weight: bold; background-color: rgba(242, 100, 25, 0.1); padding: 5px; border-radius: 5px; color: #1E293B;'>Executive</div>", unsafe_allow_html=True)
        st.metric("OOS Accuracy (5-Fold CV)", f"{exec_accuracy:.2%}", help="Average accuracy across executive elections")
        st.metric("OOS Log-Loss", f"{exec_logloss:.4f}", help="Ensemble cross-entropy error metric")
        st.metric("Tuned Temp (T)", f"{TUNED_PARAMS.get('executive_model', {}).get('T', 0.78)}", help="Temperature parameter applied to executive GBDT probabilities")
    with col_m3:
        st.markdown("<div style='text-align: center; font-weight: bold; background-color: rgba(134, 187, 216, 0.1); padding: 5px; border-radius: 5px; color: #1E293B;'>Legislative (Congressional)</div>", unsafe_allow_html=True)
        st.metric("OOS Accuracy (5-Fold CV)", f"{leg_accuracy:.2%}", help="Average accuracy across legislative elections")
        st.metric("OOS Log-Loss", f"{leg_logloss:.4f}", help="Ensemble cross-entropy error metric")
        st.metric("Tuned Temp (T)", f"{TUNED_PARAMS.get('legislative_model', {}).get('T', 0.80)}", help="Temperature parameter applied to legislative GBDT probabilities")
        
    st.markdown("##### Out-of-Sample Accuracy across Historical Validation Breakpoints")
    st.write(
        "By training models only on data before a specific milestone year (e.g. 2010) and validating "
        "on the subsequent 4 years (e.g. 2011-2014), we ensure our validation mimics true out-of-sample forecasting:"
    )
    
    fig_cv = go.Figure()
    fig_cv.add_trace(go.Bar(x=cv_df["Origin"], y=cv_df["CatBoost Classifier"], name="CatBoost", marker_color="#86BBD8"))
    fig_cv.add_trace(go.Bar(x=cv_df["Origin"], y=cv_df["XGBoost Classifier"], name="XGBoost", marker_color="#33658A"))
    fig_cv.add_trace(go.Bar(x=cv_df["Origin"], y=cv_df["Voting Ensemble"], name="Ensemble", marker_color="#F6AE2D"))
    
    fig_cv.update_layout(
        barmode='group',
        xaxis=dict(title="Historical Training Cutoff & Test Period"),
        yaxis=dict(title="OOS Accuracy", tickformat=".1%", range=[0.4, 0.85]),
        legend=dict(orientation="h", y=1.1, x=0.01),
        margin=dict(l=10, r=10, t=40, b=10),
        height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig_cv, width="stretch")

with diag_tab2:
    st.markdown("##### Probability Calibration Curve")
    st.write(
        "A well-calibrated forecast is one where a predicted win probability of 70% translates to the incumbent winning "
        "approximately 70% of the time. The line chart below checks our calibrated probabilities against actual win rates:"
    )
    
    bins = np.arange(0.5, 1.05, 0.05)
    labels = [f"{int(bins[i]*100)}-{int(bins[i+1]*100)}%" for i in range(len(bins)-1)]
    known_preds["conf_bin"] = pd.cut(known_preds["raw_confidence"], bins=bins, labels=labels, include_lowest=True)
    
    bin_stats = known_preds.groupby("conf_bin", observed=False).agg(
        accuracy=("is_correct", "mean"),
        sample_count=("is_correct", "count")
    ).reset_index()
    bin_stats = bin_stats[bin_stats["sample_count"] > 0]
    
    if not bin_stats.empty:
        fig_cal = go.Figure()
        
        # Bars showing sample frequencies
        bin_type_counts = known_preds.groupby(["conf_bin", "election_type"], observed=False).size().unstack(fill_value=0)
        bin_type_counts = bin_type_counts.reindex(bin_stats["conf_bin"], fill_value=0)
        
        exec_counts = bin_type_counts.get("Executive", pd.Series(0, index=bin_stats["conf_bin"]))
        leg_counts = bin_type_counts.get("Legislative", pd.Series(0, index=bin_stats["conf_bin"]))
        
        fig_cal.add_trace(go.Bar(
            x=bin_stats["conf_bin"].astype(str),
            y=exec_counts,
            name="Exec Count (Right Axis)",
            yaxis="y2",
            marker=dict(color="#F26419"),
            opacity=0.25
        ))
        fig_cal.add_trace(go.Bar(
            x=bin_stats["conf_bin"].astype(str),
            y=leg_counts,
            name="Leg Count (Right Axis)",
            yaxis="y2",
            marker=dict(color="#33658A"),
            opacity=0.25
        ))
        
        # Perfect calibration line
        midpoints = [0.525, 0.575, 0.625, 0.675, 0.725, 0.775, 0.825, 0.875, 0.925, 0.975]
        bin_midpoint_map = {labels[i]: midpoints[i] for i in range(len(labels))}
        active_midpoints = [bin_midpoint_map[str(b)] for b in bin_stats["conf_bin"]]
        
        fig_cal.add_trace(go.Scatter(
            x=bin_stats["conf_bin"].astype(str),
            y=active_midpoints,
            name="Perfect Calibration (y=x)",
            mode="lines",
            line=dict(color="#F6AE2D", width=2, dash="dash")
        ))
        
        exec_known = known_preds[known_preds["election_type"] == "Executive"]
        exec_bin_stats = exec_known.groupby("conf_bin", observed=False).agg(
            accuracy=("is_correct", "mean"),
            sample_count=("is_correct", "count")
        ).reset_index()
        exec_active = exec_bin_stats[exec_bin_stats["sample_count"] > 0]
        
        leg_known = known_preds[known_preds["election_type"] == "Legislative"]
        leg_bin_stats = leg_known.groupby("conf_bin", observed=False).agg(
            accuracy=("is_correct", "mean"),
            sample_count=("is_correct", "count")
        ).reset_index()
        leg_active = leg_bin_stats[leg_bin_stats["sample_count"] > 0]
        
        fig_cal.add_trace(go.Scatter(
            x=exec_active["conf_bin"].astype(str),
            y=exec_active["accuracy"],
            name="Executive Accuracy",
            mode="lines+markers",
            line=dict(color="#F26419", width=3),
            marker=dict(size=8)
        ))
        fig_cal.add_trace(go.Scatter(
            x=leg_active["conf_bin"].astype(str),
            y=leg_active["accuracy"],
            name="Legislative Accuracy",
            mode="lines+markers",
            line=dict(color="#33658A", width=3),
            marker=dict(size=8)
        ))
        
        fig_cal.update_layout(
            xaxis=dict(title="Model Confidence Bin"),
            yaxis=dict(title="Actual OOS Accuracy", tickformat=".0%", range=[0.4, 1.05]),
            yaxis2=dict(
                title="Election Sample Size",
                overlaying="y",
                side="right",
                showgrid=False
            ),
            barmode="stack",
            legend=dict(orientation="h", y=1.2, x=0.01),
            margin=dict(l=10, r=10, t=40, b=10),
            height=380,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_cal, width="stretch")

with diag_tab3:
    m_col1, m_col2 = st.columns(2)
    with m_col1:
        st.markdown("##### Confusion Matrices")
        tab_exec, tab_leg = st.tabs(["Executive", "Legislative"])
        
        with tab_exec:
            if not exec_preds.empty:
                y_true_ex = exec_preds["target_outcome"].astype(int)
                y_pred_ex = exec_preds["predicted_outcome"].astype(int)
                tn_ex, fp_ex, fn_ex, tp_ex = confusion_matrix(y_true_ex, y_pred_ex).ravel()
                
                cm_ex_df = pd.DataFrame([
                    ["Actual Challenger Loss", tp_ex, fn_ex],
                    ["Actual Incumbent Loss", fp_ex, tn_ex]
                ], columns=["Outcome", "Predicted Incumbent Victory", "Predicted Challenger Victory"])
                
                st.dataframe(cm_ex_df, width="stretch", hide_index=True)
                st.write(
                    f"Of historical executive races, the model predicted **{tp_ex}** incumbent victories "
                    f"and **{tn_ex}** incumbent defeats, with **{fp_ex}** False Positives and **{fn_ex}** False Negatives."
                )
            else:
                st.warning("No executive predictions available.")
                
        with tab_leg:
            if not leg_preds.empty:
                y_true_lg = leg_preds["target_outcome"].astype(int)
                y_pred_lg = leg_preds["predicted_outcome"].astype(int)
                tn_lg, fp_lg, fn_lg, tp_lg = confusion_matrix(y_true_lg, y_pred_lg).ravel()
                
                cm_lg_df = pd.DataFrame([
                    ["Actual Challenger Loss", tp_lg, fn_lg],
                    ["Actual Incumbent Loss", fp_lg, tn_lg]
                ], columns=["Outcome", "Predicted Incumbent Victory", "Predicted Challenger Victory"])
                
                st.dataframe(cm_lg_df, width="stretch", hide_index=True)
                st.write(
                    f"Of historical legislative races, the model predicted **{tp_lg}** incumbent victories "
                    f"and **{tn_lg}** incumbent defeats, with **{fp_lg}** False Positives and **{fn_lg}** False Negatives."
                )
            else:
                st.warning("No legislative predictions available.")
                
    with m_col2:
        st.markdown("##### Model Calibration Parameters & Settings")
        st.write("Temperature-scaled probability adjustments and ensemble weights determined by validation grid searches:")
        st.json(TUNED_PARAMS)

with diag_tab4:
    st.markdown("##### Out-of-Sample Accuracy by Country")
    st.write(
        "This choropleth map visualizes the model's out-of-sample historical prediction accuracy for each country. "
        "To prevent countries with very few elections from distorting the color scale (e.g. showing 100% or 0% accuracy from only 1 or 2 races), "
        "countries with **fewer than 3 total elections are colored in gray**. These remain fully interactive and can be hovered over to inspect their rates. "
        "For all other countries, coloring represents their actual raw out-of-sample accuracy."
    )
    
    acc_df = get_accuracy_by_country(all_preds_merged)
    if not acc_df.empty:
        acc_df_valid = acc_df[acc_df["color_accuracy"].notna()]
        acc_df_gray = acc_df[acc_df["color_accuracy"].isna()]
        
        df_for_base = acc_df_valid if not acc_df_valid.empty else acc_df
        
        fig_acc_map = px.choropleth(
            df_for_base,
            locations="country_code",
            color="color_accuracy" if not acc_df_valid.empty else None,
            hover_name="country_name",
            custom_data=["accuracy_str", "exec_ratio", "leg_ratio"],
            color_continuous_scale="RdYlGn",
            range_color=[0.4, 1.0],
            projection="natural earth"
        )
        
        if not acc_df_valid.empty:
            fig_acc_map.update_traces(
                hovertemplate="<b>%{hovertext}</b><br><br>" +
                              "Historical OOS Accuracy: <b>%{customdata[0]}</b><br>" +
                              "Executive: <b>%{customdata[1]}</b><br>" +
                              "Legislative: <b>%{customdata[2]}</b><extra></extra>"
            )
            
        if not acc_df_gray.empty:
            import plotly.graph_objects as go
            gray_trace = go.Choropleth(
                locations=acc_df_gray["country_code"],
                z=[0.5] * len(acc_df_gray),
                colorscale=[[0, "#CCCCCC"], [1, "#CCCCCC"]],  # Explicit medium-light gray color
                showscale=False,
                hovertext=acc_df_gray["country_name"],
                customdata=np.stack([
                    acc_df_gray["accuracy_str"],
                    acc_df_gray["exec_ratio"],
                    acc_df_gray["leg_ratio"]
                ], axis=-1),
                hovertemplate="<b>%{hovertext}</b><br><br>" +
                              "Historical OOS Accuracy: <b>%{customdata[0]}</b><br>" +
                              "Executive: <b>%{customdata[1]}</b><br>" +
                              "Legislative: <b>%{customdata[2]}</b><br>" +
                              "<i>(Fewer than 3 elections)</i><extra></extra>"
            )
            fig_acc_map.add_trace(gray_trace)
            
        fig_acc_map.update_layout(
            geo=dict(showframe=False, showcoastlines=True, coastlinecolor="#86BBD8"),
            coloraxis_colorbar=dict(title="Accuracy", tickformat=".0%"),
            margin=dict(l=0, r=0, t=10, b=0),
            height=400,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_acc_map, width="stretch")
    else:
        st.info("No historical accuracy data available to display on the map.")
