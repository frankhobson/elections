import os
import sys
import pickle
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Prevent OpenMP conflict issues on macOS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Add parent directory to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from models.correlation_models import (
    FEATURES,
    EXEC_SPECIFIC,
    LEG_SPECIFIC,
    TUNED_PARAMS
)

app = FastAPI(
    title="Global Election Forecaster API",
    description="REST API for global election prediction, historical out-of-sample evaluations, country performance stats, and SHAP feature attributions.",
    version="1.0.0"
)

# Enable CORS for local Vite dev server (port 5173 / 3000 / wildcard)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global in-memory cache
cache_data: Dict[str, Any] = {}

def get_feature_info(feature_name: str):
    info_map = {
        "constitutional_rigidity": ("Constitutional Rigidity Index", "Judicial and legislative institutional constraints (Tsebelis, 2002)."),
        "years_since_last_alternation": ("Governing Longevity (Fatigue)", "Years since executive power alternation (Rose & Mackie, 1983)."),
        "government_continuity_score": ("Government Continuity Score", "Interaction between coalition longevity and legislative constraints."),
        "pedersen_index": ("Pedersen Volatility Index", "Historical electoral volatility and party system instability (Pedersen, 1979)."),
        "electoral_volatility_10yr": ("10-Year Electoral Volatility", "Rolling 10-year variation in party vote shares."),
        "electoral_volatility_change": ("Volatility Acceleration", "Shift in electoral volatility over recent election cycles."),
        "inflation_acceleration": ("Inflation Acceleration", "Change in annual inflation rate relative to prior cycle."),
        "gdp_growth_acceleration": ("GDP Growth Acceleration", "Change in GDP growth velocity relative to prior term."),
        "unemployment_acceleration": ("Unemployment Acceleration", "Shift in labor market unemployment rates."),
        "real_wage_growth": ("Real Income Growth", "Per capita income growth rate."),
        "migration_shock": ("Migration Inflow Shock", "Rapid demographic population shift indicator."),
        "refugee_inflow_shock": ("Refugee Inflow Shock", "Displacement and humanitarian refugee inflow metric."),
        "government_sentiment_score": ("Government Media Tone", "Net tone of media coverage regarding incumbent administration."),
        "government_corruption_events": ("Corruption Event Salience", "Decayed count of public corruption news events."),
        "government_corruption_ratio": ("Corruption Coverage Ratio", "Ratio of corruption news relative to general governance news."),
        "legislative_productivity": ("Legislative Efficiency Proxy", "Institutional legislative throughput indicator."),
        "government_news_salience": ("Executive Media Salience", "Share of total national news focused on governing coalition."),
        "government_media_balance": ("Media Tone Imbalance", "Tone discrepancy between opposition and incumbent news."),
        "latent_global_political_climate": ("Latent Global Political Climate", "PCA aggregate of worldwide political sentiment."),
        "latent_global_economic_anxiety": ("Latent Global Economic Anxiety", "PCA aggregate of international economic anxiety."),
        "latent_global_anti_incumbent_index": ("Latent Global Anti-Incumbent Index", "PCA aggregate of worldwide incumbent win/loss trends."),
        "latent_global_democracy_trend": ("Latent Global Democratic Trend", "PCA aggregate of global democratic health."),
        "spatial_lag_democracy_delta_trade": ("Trade Neighbor Democratic Delta", "Democratic reform/backsliding in key trading partner states."),
        "spatial_lag_clean_delta_trade": ("Trade Neighbor Clean Delta", "Electoral cleanliness shift in trading partner nations."),
        "spatial_lag_democracy_delta_alliance": ("Alliance Neighbor Democratic Delta", "Democratic shifts in formal military/diplomatic allies."),
        "spatial_lag_clean_delta_alliance": ("Alliance Neighbor Clean Delta", "Electoral cleanliness shifts in allied nations."),
        "global_incumbent_punishment_index": ("Global Incumbent Punishment Index", "Worldwide annual rate of challenger victories."),
        "global_inflation_pressure": ("Global Inflation Stress", "Worldwide macroeconomic price level index."),
        "global_energy_pressure": ("Global Energy Shock Index", "Brent oil price volatility and global energy stress."),
        "is_open_seat": ("Open Seat Succession Election", "Elections where the standing chief executive does not run (Jalalzai, 2013)."),
        "successor_run": ("Hand-Picked Successor Running", "Elections where the incumbent party fields a designated successor."),
        "open_seat_prior_margin_interaction": ("Open Seat x Prior Margin", "Interaction between incumbent absence and prior victory margin."),
        "party_fragmentation_legislative_interaction": ("Party System Fragmentation", "Interaction between effective parties (ENP) and proportional representation."),
        "electoral_system_majoritarian": ("Majoritarian Electoral System", "First-past-the-post or single-member district rules."),
        "electoral_system_pr": ("Proportional Representation System", "Party-list proportional representation rules."),
        "incumbent_seat_buffer": ("Incumbent Seat Share Buffer", "Incumbent seat share margin above majority threshold (50%)."),
        "is_majority_fragile": ("Fragile Majority Flag", "Governments holding less than 55% seat majority."),
        "pr_fragmentation_interaction": ("PR x Party Fragmentation", "Proportional representation combined with high party count."),
        "polarization_fragmentation_interaction": ("Polarization x Party Fragmentation", "Ideological polarization combined with party system fragmentation.")
    }
    
    if feature_name in info_map:
        return info_map[feature_name]
    elif feature_name.startswith("country_"):
        ccode = feature_name.replace("country_", "")
        return (f"Country Baseline Factor ({ccode})", "Fixed effect baseline adjustment for country historical tendencies.")
    else:
        clean_name = feature_name.replace("_", " ").title()
        return (clean_name, "Institutional and macroeconomic predictive indicator.")

@app.on_event("startup")
def load_cache():
    global cache_data
    cache_path = os.path.join(BASE_DIR, "dashboard", "cached_dashboard_data.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cache_data = pickle.load(f)
        print("Loaded cached_dashboard_data.pkl into FastAPI memory successfully!")
    else:
        print("WARNING: cached_dashboard_data.pkl not found! Please run precompute_dashboard_cache.py.")

@app.get("/")
def read_root():
    return {
        "status": "online",
        "service": "Global Election Forecaster API",
        "docs_url": "/docs"
    }

@app.get("/api/stats")
def get_stats():
    if not cache_data:
        raise HTTPException(status_code=500, detail="Cache data not loaded")
        
    all_preds = cache_data["all_preds"]
    c_names = cache_data["c_names"]
    
    total_countries = len(c_names)
    historical_count = int(all_preds["target_outcome"].notna().sum())
    upcoming_count = int(all_preds["target_outcome"].isna().sum())
    
    df_known = all_preds[all_preds["target_outcome"].notna()].copy()
    correct_count = int((df_known["target_outcome"].astype(int) == df_known["predicted_outcome"].astype(int)).sum())
    overall_accuracy = round((correct_count / historical_count) * 100.0, 2) if historical_count > 0 else 0.0
    
    clean_df = df_known[df_known["clean_index"] >= 0.65]
    clean_correct = int((clean_df["target_outcome"].astype(int) == clean_df["predicted_outcome"].astype(int)).sum())
    clean_accuracy = round((clean_correct / len(clean_df)) * 100.0, 2) if len(clean_df) > 0 else 0.0
    
    return {
        "total_countries": total_countries,
        "historical_elections": historical_count,
        "upcoming_elections": upcoming_count,
        "overall_accuracy_pct": overall_accuracy,
        "overall_correct": correct_count,
        "clean_accuracy_pct": clean_accuracy,
        "clean_correct": clean_correct,
        "clean_total": len(clean_df),
        "cv_mean_cb": round(float(cache_data.get("mean_cb", 0.67)) * 100.0, 2),
        "cv_mean_xgb": round(float(cache_data.get("mean_xgb", 0.69)) * 100.0, 2),
        "cv_mean_ensemble": round(float(cache_data.get("mean_ensemble", 0.67)) * 100.0, 2)
    }

@app.get("/api/predictions/upcoming")
def get_upcoming_predictions(
    search: Optional[str] = Query(None),
    year: Optional[int] = Query(None),
    election_type: Optional[str] = Query(None),
    region: Optional[str] = Query(None)
):
    if not cache_data:
        raise HTTPException(status_code=500, detail="Cache data not loaded")
        
    all_preds = cache_data["all_preds"].copy()
    c_names = cache_data["c_names"].set_index("country_code")["country_name"].to_dict()
    
    upcoming = all_preds[all_preds["target_outcome"].isna()].copy()
    upcoming["country_name"] = upcoming["country_code"].map(c_names).fillna(upcoming["country_code"])
    
    if search:
        s = search.lower()
        upcoming = upcoming[
            upcoming["country_name"].str.lower().str.contains(s) |
            upcoming["country_code"].str.lower().str.contains(s)
        ]
        
    if year:
        upcoming = upcoming[upcoming["year"] == year]
        
    if election_type:
        upcoming = upcoming[upcoming["election_type"] == election_type]
        
    if region and region != "All":
        upcoming = upcoming[upcoming["region"] == region]
        
    upcoming = upcoming.sort_values(by=["year", "country_name"]).reset_index(drop=True)
    
    records = upcoming.to_dict(orient="records")
    # Clean NaN values for JSON serialization
    for r in records:
        for k, v in r.items():
            if pd.isna(v):
                r[k] = None
    return records

@app.get("/api/predictions/historical")
def get_historical_predictions(
    search: Optional[str] = Query(None),
    region: Optional[str] = Query(None),
    match_status: Optional[str] = Query(None) # Correct, Incorrect, Unknown
):
    if not cache_data:
        raise HTTPException(status_code=500, detail="Cache data not loaded")
        
    all_preds = cache_data["all_preds"].copy()
    c_names = cache_data["c_names"].set_index("country_code")["country_name"].to_dict()
    
    historical = all_preds[all_preds["target_outcome"].notna()].copy()
    historical["country_name"] = historical["country_code"].map(c_names).fillna(historical["country_code"])
    
    historical["target_outcome_int"] = historical["target_outcome"].astype(int)
    historical["predicted_outcome_int"] = historical["predicted_outcome"].astype(int)
    historical["is_correct"] = (historical["target_outcome_int"] == historical["predicted_outcome_int"]).astype(int)
    
    if search:
        s = search.lower()
        historical = historical[
            historical["country_name"].str.lower().str.contains(s) |
            historical["country_code"].str.lower().str.contains(s)
        ]
        
    if region and region != "All":
        historical = historical[historical["region"] == region]
        
    if match_status:
        if match_status == "Correct":
            historical = historical[historical["is_correct"] == 1]
        elif match_status == "Incorrect":
            historical = historical[historical["is_correct"] == 0]
            
    historical = historical.sort_values(by=["year", "country_name"], ascending=[False, True]).reset_index(drop=True)
    
    records = historical.to_dict(orient="records")
    for r in records:
        for k, v in r.items():
            if pd.isna(v):
                r[k] = None
    return records

@app.get("/api/country-accuracy")
def get_country_accuracy():
    if not cache_data:
        raise HTTPException(status_code=500, detail="Cache data not loaded")
        
    all_preds = cache_data["all_preds"].copy()
    c_names_df = cache_data["c_names"]
    c_names = c_names_df.set_index("country_code")["country_name"].to_dict()
    c_regions = c_names_df.set_index("country_code")["region"].to_dict()
    
    hist = all_preds[all_preds["target_outcome"].notna()].copy()
    hist["target_outcome_int"] = hist["target_outcome"].astype(int)
    hist["predicted_outcome_int"] = hist["predicted_outcome"].astype(int)
    hist["is_correct"] = (hist["target_outcome_int"] == hist["predicted_outcome_int"]).astype(int)
    
    hist["is_exec"] = (hist["election_type"] == "Executive").astype(int)
    hist["is_leg"] = (hist["election_type"] == "Legislative").astype(int)
    
    hist["exec_correct"] = hist["is_correct"] * hist["is_exec"]
    hist["leg_correct"] = hist["is_correct"] * hist["is_leg"]
    
    grouped = hist.groupby("country_code").agg(
        total=("target_outcome_int", "count"),
        correct=("is_correct", "sum"),
        exec_total=("is_exec", "sum"),
        exec_correct=("exec_correct", "sum"),
        leg_total=("is_leg", "sum"),
        leg_correct=("leg_correct", "sum")
    ).reset_index()
    
    results = []
    for _, row in grouped.iterrows():
        ccode = row["country_code"]
        tot = int(row["total"])
        corr = int(row["correct"])
        acc = round((corr / tot) * 100.0, 2) if tot > 0 else 0.0
        
        results.append({
            "country_code": ccode,
            "country_name": c_names.get(ccode, ccode),
            "region": c_regions.get(ccode, "Unknown"),
            "total": tot,
            "correct": corr,
            "accuracy_pct": acc,
            "exec_total": int(row["exec_total"]),
            "exec_correct": int(row["exec_correct"]),
            "leg_total": int(row["leg_total"]),
            "leg_correct": int(row["leg_correct"])
        })
        
    results.sort(key=lambda x: (-x["accuracy_pct"], -x["total"], x["country_name"]))
    for rank, r in enumerate(results, 1):
        r["rank"] = rank
        
    return results

@app.get("/api/model-diagnostics")
def get_model_diagnostics():
    if not cache_data:
        raise HTTPException(status_code=500, detail="Cache data not loaded")
        
    cv_df = cache_data["cv_df"].copy()
    cv_records = cv_df.to_dict(orient="records")
    
    # Pre-calculate country accuracy distribution histogram buckets
    country_stats = get_country_accuracy()
    accuracies = [c["accuracy_pct"] for c in country_stats]
    
    bins_hist = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    hist_counts, _ = np.histogram(accuracies, bins=bins_hist)
    
    hist_buckets = []
    for i in range(len(bins_hist) - 1):
        hist_buckets.append({
            "bin": f"{bins_hist[i]}-{bins_hist[i+1]}%",
            "count": int(hist_counts[i])
        })

    from sklearn.metrics import confusion_matrix
    all_preds = cache_data["all_preds"].copy()
    known_preds = all_preds[all_preds["target_outcome"].notna()].copy()
    known_preds["target_outcome_int"] = known_preds["target_outcome"].astype(int)
    known_preds["predicted_outcome_int"] = known_preds["predicted_outcome"].astype(int)
    known_preds["is_correct"] = (known_preds["target_outcome_int"] == known_preds["predicted_outcome_int"]).astype(int)
    
    # 1. Calibration curve statistics
    bins = np.arange(0.5, 1.05, 0.05)
    labels = [f"{int(bins[i]*100)}-{int(bins[i+1]*100)}%" for i in range(len(bins)-1)]
    known_preds["conf_bin"] = pd.cut(known_preds["raw_confidence"], bins=bins, labels=labels, include_lowest=True)
    
    bin_labels = labels
    exec_known = known_preds[known_preds["election_type"] == "Executive"]
    leg_known = known_preds[known_preds["election_type"] == "Legislative"]
    
    calibration_bins = []
    midpoints = [0.525, 0.575, 0.625, 0.675, 0.725, 0.775, 0.825, 0.875, 0.925, 0.975]
    for i, lbl in enumerate(labels):
        sub_ex = exec_known[exec_known["conf_bin"] == lbl]
        sub_lg = leg_known[leg_known["conf_bin"] == lbl]
        
        ex_cnt = len(sub_ex)
        leg_cnt = len(sub_lg)
        
        ex_acc = float(sub_ex["is_correct"].mean()) if ex_cnt > 0 else None
        leg_acc = float(sub_lg["is_correct"].mean()) if leg_cnt > 0 else None
        
        calibration_bins.append({
            "bin": lbl,
            "midpoint": midpoints[i],
            "exec_count": ex_cnt,
            "leg_count": leg_cnt,
            "exec_accuracy": round(ex_acc, 4) if ex_acc is not None else None,
            "leg_accuracy": round(leg_acc, 4) if leg_acc is not None else None,
        })

    # 2. Executive & Legislative Confusion Matrices
    cm_exec = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    if not exec_known.empty:
        tn, fp, fn, tp = confusion_matrix(exec_known["target_outcome_int"], exec_known["predicted_outcome_int"]).ravel()
        cm_exec = {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}
        
    cm_leg = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    if not leg_known.empty:
        tn, fp, fn, tp = confusion_matrix(leg_known["target_outcome_int"], leg_known["predicted_outcome_int"]).ravel()
        cm_leg = {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}

    return {
        "cv_folds": cv_records,
        "mean_catboost": round(float(cache_data.get("mean_cb", 0.67)) * 100.0, 2),
        "mean_xgboost": round(float(cache_data.get("mean_xgb", 0.69)) * 100.0, 2),
        "mean_ensemble": round(float(cache_data.get("mean_ensemble", 0.67)) * 100.0, 2),
        "country_accuracy_distribution": hist_buckets,
        "calibration_bins": calibration_bins,
        "confusion_matrix_exec": cm_exec,
        "confusion_matrix_leg": cm_leg
    }

@app.get("/api/vdem-map/{year}")
def get_vdem_map(year: int):
    if not cache_data or "vdem_map_data" not in cache_data:
        raise HTTPException(status_code=500, detail="V-Dem map data not loaded")
        
    v_data = cache_data["vdem_map_data"].copy()
    df_map = v_data[v_data["year"] == year].copy()
    regime_map = {0: "Closed Autocracy", 1: "Electoral Autocracy", 2: "Electoral Democracy", 3: "Liberal Democracy"}
    df_map["regime_name"] = df_map["regime_type"].map(regime_map).fillna("Unknown")
    
    records = df_map.to_dict(orient="records")
    for r in records:
        for k, v in r.items():
            if pd.isna(v):
                r[k] = None
    return records

@app.get("/api/feature-importances")
def get_feature_importances():
    if not cache_data:
        raise HTTPException(status_code=500, detail="Cache data not loaded")
        
    from models.correlation_models import get_feature_importance_report
    df_modeling = cache_data["df_modeling"]
    imp_df = get_feature_importance_report(df_modeling).head(10)
    
    results = []
    for _, row in imp_df.iterrows():
        feat_name = row["feature"]
        human_name, lit_desc = get_feature_info(feat_name)
        results.append({
            "feature": feat_name,
            "human_name": human_name,
            "importance": round(float(row["importance"]), 4),
            "explanation": lit_desc
        })
    return results

@app.get("/api/shap/{election_id}")
def get_shap_explanation(election_id: str):
    if not cache_data:
        raise HTTPException(status_code=500, detail="Cache data not loaded")
        
    import xgboost as xgb
    import catboost as cb
    
    all_preds = cache_data["all_preds"]
    df_modeling = cache_data["df_modeling"]
    hybrid_model = cache_data["hybrid_model"]
    c_names_df = cache_data["c_names"]
    
    match_row = df_modeling[df_modeling["election_id"] == election_id]
    if match_row.empty:
        raise HTTPException(status_code=404, detail="Election ID not found")
        
    selected_row = match_row.iloc[0]
    selected_country_code = selected_row["country_code"]
    is_pres = selected_row["is_presidential"]
    clean_idx = selected_row["clean_index"]
    is_clean = pd.notna(clean_idx) and clean_idx >= 0.65
    
    country_dummies = [col for col in df_modeling.columns if col.startswith("country_") and len(col) == 11 and col[8:].isupper()]
    exec_features = [f for f in FEATURES if f not in LEG_SPECIFIC and not (f.startswith("country_") and len(f) == 11 and f[8:].isupper())] + country_dummies
    leg_features = [f for f in FEATURES if f not in EXEC_SPECIFIC and not (f.startswith("country_") and len(f) == 11 and f[8:].isupper())] + country_dummies
    
    exec_params = TUNED_PARAMS.get("executive_model", {"xgb_weight": 0.7})
    leg_params = TUNED_PARAMS.get("legislative_model", {"xgb_weight": 0.9})
    
    if is_pres == 1:
        feats = exec_features
        xgb_w = exec_params.get("xgb_weight", 0.7)
        if is_clean and hybrid_model.ex_reg_xgb is not None:
            model_xgb = hybrid_model.ex_reg_xgb
            model_cb = hybrid_model.ex_reg_cb
        else:
            model_xgb = hybrid_model.ex_unclean_xgb
            model_cb = hybrid_model.ex_unclean_cb
    else:
        feats = leg_features
        xgb_w = leg_params.get("xgb_weight", 0.9)
        if is_clean and hybrid_model.leg_reg_xgb is not None:
            model_xgb = hybrid_model.leg_reg_xgb
            model_cb = hybrid_model.leg_reg_cb
        else:
            model_xgb = hybrid_model.leg_unclean_xgb
            model_cb = hybrid_model.leg_unclean_cb
            
    X_pred = match_row[feats]
    
    # Compute contributions
    dmatrix = xgb.DMatrix(X_pred)
    contribs_xgb = model_xgb.get_booster().predict(dmatrix, pred_contribs=True)[0]
    contribs_cb = model_cb.get_feature_importance(data=cb.Pool(X_pred), type='ShapValues')[0]
    
    contribs = xgb_w * contribs_xgb[:-1] + (1.0 - xgb_w) * contribs_cb[:-1]
    
    feature_contribs = []
    for i, feat in enumerate(feats):
        v = X_pred[feat].iloc[0]
        v_clean = float(v) if (pd.notna(v) and not np.isnan(v)) else None
        c_val = float(contribs[i]) if (pd.notna(contribs[i]) and not np.isnan(contribs[i])) else 0.0
        feature_contribs.append({
            "feature": feat,
            "value": v_clean,
            "contribution": c_val
        })
    df_contribs = pd.DataFrame(feature_contribs)
    
    # Filter out other countries' baseline dummies
    other_countries_dummies = [f"country_{c}" for c in c_names_df["country_code"] if c != selected_country_code]
    df_contribs = df_contribs[~df_contribs["feature"].isin(other_countries_dummies)]
    
    df_pos = df_contribs[df_contribs["contribution"] > 1e-4].sort_values(by="contribution", ascending=False).head(5)
    df_neg = df_contribs[df_contribs["contribution"] < -1e-4].sort_values(by="contribution", ascending=True).head(5)
    
    df_top = pd.concat([df_pos, df_neg])
    
    top_features = []
    for _, row_c in df_top.iterrows():
        feat_name = row_c["feature"]
        human_name, lit_desc = get_feature_info(feat_name)
        contrib_val = float(row_c["contribution"])
        val_raw = row_c["value"]
        val_clean = float(val_raw) if (pd.notna(val_raw) and not np.isnan(val_raw)) else None
        top_features.append({
            "feature": feat_name,
            "human_name": human_name,
            "literature_info": lit_desc,
            "value": val_clean,
            "contribution": round(contrib_val, 4),
            "direction": "Favors Incumbent" if contrib_val > 0 else "Favors Challenger"
        })
        
    top_features.sort(key=lambda x: x["contribution"], reverse=True)
    
    # Metadata info
    pred_info = all_preds[all_preds["election_id"] == election_id]
    prob_val = 0.5
    winner_str = "Unknown"
    target_str = None
    if not pred_info.empty:
        p_row = pred_info.iloc[0]
        prob_val = round(float(p_row["raw_probability"]), 4) if pd.notna(p_row["raw_probability"]) else 0.5
        winner_str = str(p_row["predicted_winner"])
        if pd.notna(p_row["target_outcome"]):
            target_str = "Incumbent Victory" if int(p_row["target_outcome"]) == 1 else "Challenger Victory"
            
    return {
        "election_id": election_id,
        "country_code": selected_country_code,
        "year": int(selected_row["year"]),
        "election_type": "Executive" if is_pres == 1 else "Legislative",
        "predicted_winner": winner_str,
        "predicted_probability": prob_val,
        "actual_outcome": target_str,
        "is_clean": bool(is_clean),
        "top_features": top_features
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
