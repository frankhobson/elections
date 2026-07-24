import os
import pickle
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_PATH = os.path.join(BASE_DIR, "dashboard", "cached_dashboard_data.pkl")
OUTPUT_DIR = os.path.join(BASE_DIR, "frontend", "public", "data")

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Loading cached model data from {CACHE_PATH}...")
with open(CACHE_PATH, "rb") as f:
    cache_data = pickle.load(f)

# Helper to sanitize NaN / Inf values for JSON safety
def sanitize(val):
    if isinstance(val, (float, np.floating)):
        if np.isnan(val) or np.isinf(val):
            return None
        return float(val)
    elif isinstance(val, (int, np.integer)):
        return int(val)
    elif isinstance(val, (bool, np.bool_)):
        return bool(val)
    elif pd.isna(val):
        return None
    return val

def sanitize_dict(d):
    if isinstance(d, dict):
        return {k: sanitize_dict(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [sanitize_dict(v) for v in d]
    else:
        return sanitize(d)

all_preds = cache_data["all_preds"].copy()
c_names = cache_data["c_names"].copy()

# Add country names & regions to all_preds
merged_preds = all_preds.merge(c_names[["country_code", "country_name"]], on="country_code", how="left")
merged_preds["country_name"] = merged_preds["country_name"].fillna(merged_preds["country_code"])
merged_preds["region"] = merged_preds["region"].fillna("Other")

# 1. Stats JSON
known_preds = merged_preds[merged_preds["target_outcome"].notna()].copy()
known_preds["is_correct"] = (known_preds["predicted_outcome"] == known_preds["target_outcome"]).astype(int)
overall_acc = known_preds["is_correct"].mean()

clean_preds = known_preds[known_preds["clean_index"] >= 0.65]
clean_acc = clean_preds["is_correct"].mean() if not clean_preds.empty else overall_acc

stats = {
    "total_countries": int(merged_preds["country_code"].nunique()),
    "historical_elections": int(len(known_preds)),
    "upcoming_elections": int(len(merged_preds[merged_preds["target_outcome"].isna()])),
    "overall_accuracy_pct": round(float(overall_acc) * 100.0, 2),
    "overall_correct": int(known_preds["is_correct"].sum()),
    "clean_accuracy_pct": round(float(clean_acc) * 100.0, 2),
    "clean_correct": int(clean_preds["is_correct"].sum()) if not clean_preds.empty else 0,
    "clean_total": int(len(clean_preds)),
    "cv_mean_cb": round(float(cache_data.get("mean_cb", 0.67)), 4),
    "cv_mean_xgb": round(float(cache_data.get("mean_xgb", 0.69)), 4),
    "cv_mean_ensemble": round(float(cache_data.get("mean_ensemble", 0.67)), 4)
}

with open(os.path.join(OUTPUT_DIR, "stats.json"), "w") as f:
    json.dump(sanitize_dict(stats), f, indent=2)

# 2. Upcoming Elections JSON
upcoming_df = merged_preds[merged_preds["target_outcome"].isna()].sort_values(["year", "country_name"])
upcoming_records = []
for _, r in upcoming_df.iterrows():
    upcoming_records.append({
        "election_id": str(r["election_id"]),
        "source": str(r.get("source", "NELDA")),
        "country_code": str(r["country_code"]),
        "country_name": str(r["country_name"]),
        "region": str(r["region"]),
        "year": int(r["year"]),
        "election_type": str(r["election_type"]),
        "is_scheduled": int(r.get("is_scheduled", 1)),
        "clean_index": sanitize(r.get("clean_index")),
        "gdp_growth": sanitize(r.get("gdp_growth")),
        "raw_probability": round(float(r["raw_probability"]), 4),
        "predicted_outcome": int(r["predicted_outcome"]),
        "predicted_winner": "Incumbent" if r["predicted_outcome"] == 1 else "Challenger",
        "raw_confidence": round(float(r["raw_confidence"]), 4),
        "adjusted_confidence": round(float(r["adjusted_confidence"]), 4),
        "data_completeness": round(float(r.get("data_completeness", 1.0)), 4),
        "data_source_flags": int(r.get("data_source_flags", 4)),
        "target_outcome": None,
        "election_date": str(r["election_date"]) if pd.notna(r.get("election_date")) else None
    })

with open(os.path.join(OUTPUT_DIR, "upcoming.json"), "w") as f:
    json.dump(sanitize_dict(upcoming_records), f, indent=2)

# 3. Historical Elections JSON
historical_df = merged_preds[merged_preds["target_outcome"].notna()].sort_values(["year", "country_name"], ascending=[False, True])
historical_records = []
for _, r in historical_df.iterrows():
    target_int = int(r["target_outcome"])
    pred_int = int(r["predicted_outcome"])
    historical_records.append({
        "election_id": str(r["election_id"]),
        "source": str(r.get("source", "NELDA")),
        "country_code": str(r["country_code"]),
        "country_name": str(r["country_name"]),
        "region": str(r["region"]),
        "year": int(r["year"]),
        "election_type": str(r["election_type"]),
        "is_scheduled": int(r.get("is_scheduled", 1)),
        "clean_index": sanitize(r.get("clean_index")),
        "raw_probability": round(float(r["raw_probability"]), 4),
        "predicted_outcome": pred_int,
        "predicted_winner": "Incumbent" if pred_int == 1 else "Challenger",
        "raw_confidence": round(float(r["raw_confidence"]), 4),
        "adjusted_confidence": round(float(r["adjusted_confidence"]), 4),
        "data_completeness": round(float(r.get("data_completeness", 1.0)), 4),
        "data_source_flags": int(r.get("data_source_flags", 4)),
        "target_outcome": target_int,
        "target_outcome_int": target_int,
        "predicted_outcome_int": pred_int,
        "is_correct": 1 if target_int == pred_int else 0,
        "election_date": str(r["election_date"]) if pd.notna(r.get("election_date")) else None
    })

with open(os.path.join(OUTPUT_DIR, "historical.json"), "w") as f:
    json.dump(sanitize_dict(historical_records), f, indent=2)

# 4. Country Accuracy JSON
country_rows = []
for c_code, group in known_preds.groupby("country_code"):
    c_name = group["country_name"].iloc[0]
    region = group["region"].iloc[0]
    total = len(group)
    correct = int(group["is_correct"].sum())
    acc_pct = (correct / total) * 100.0 if total > 0 else 0.0

    exec_group = group[group["election_type"] == "Executive"]
    leg_group = group[group["election_type"] == "Legislative"]

    country_rows.append({
        "country_code": c_code,
        "country_name": c_name,
        "region": region,
        "total": total,
        "correct": correct,
        "accuracy_pct": round(acc_pct, 2),
        "exec_total": len(exec_group),
        "exec_correct": int(exec_group["is_correct"].sum()) if not exec_group.empty else 0,
        "leg_total": len(leg_group),
        "leg_correct": int(leg_group["is_correct"].sum()) if not leg_group.empty else 0
    })

country_rows.sort(key=lambda x: (-x["accuracy_pct"], -x["total"], x["country_name"]))
for rank, r in enumerate(country_rows, 1):
    r["rank"] = rank

with open(os.path.join(OUTPUT_DIR, "country_accuracy.json"), "w") as f:
    json.dump(sanitize_dict(country_rows), f, indent=2)

# 5. Model Diagnostics JSON
cv_df = cache_data["cv_df"].copy()
cv_records = cv_df.to_dict(orient="records")

accuracies = [c["accuracy_pct"] for c in country_rows]
bins_hist = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
hist_counts, _ = np.histogram(accuracies, bins=bins_hist)
hist_buckets = [{"bin": f"{bins_hist[i]}-{bins_hist[i+1]}%", "count": int(hist_counts[i])} for i in range(len(bins_hist)-1)]

# Calibration bins
bins = np.arange(0.5, 1.05, 0.05)
labels = [f"{int(bins[i]*100)}-{int(bins[i+1]*100)}%" for i in range(len(bins)-1)]
known_preds["conf_bin"] = pd.cut(known_preds["raw_confidence"], bins=bins, labels=labels, include_lowest=True)

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

# Confusion Matrices
cm_exec = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
if not exec_known.empty:
    tn, fp, fn, tp = confusion_matrix(exec_known["target_outcome"].astype(int), exec_known["predicted_outcome"].astype(int)).ravel()
    cm_exec = {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}

cm_leg = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
if not leg_known.empty:
    tn, fp, fn, tp = confusion_matrix(leg_known["target_outcome"].astype(int), leg_known["predicted_outcome"].astype(int)).ravel()
    cm_leg = {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}

diagnostics = {
    "cv_folds": cv_records,
    "mean_catboost": round(float(cache_data.get("mean_cb", 0.67)) * 100.0, 2),
    "mean_xgboost": round(float(cache_data.get("mean_xgb", 0.69)) * 100.0, 2),
    "mean_ensemble": round(float(cache_data.get("mean_ensemble", 0.67)) * 100.0, 2),
    "country_accuracy_distribution": hist_buckets,
    "calibration_bins": calibration_bins,
    "confusion_matrix_exec": cm_exec,
    "confusion_matrix_leg": cm_leg
}

with open(os.path.join(OUTPUT_DIR, "model_diagnostics.json"), "w") as f:
    json.dump(sanitize_dict(diagnostics), f, indent=2)

# 6. V-Dem Map JSON (grouped by year)
v_data = cache_data["vdem_map_data"].copy()
regime_map = {0: "Closed Autocracy", 1: "Electoral Autocracy", 2: "Electoral Democracy", 3: "Liberal Democracy"}
v_data["regime_name"] = v_data["regime_type"].map(regime_map).fillna("Unknown")

vdem_by_year = {}
for yr, grp in v_data.groupby("year"):
    records = grp.to_dict(orient="records")
    clean_records = []
    for r in records:
        clean_records.append({
            "country_code": str(r["country_code"]),
            "country_name": str(r["country_name"]),
            "latitude": sanitize(r.get("latitude")),
            "longitude": sanitize(r.get("longitude")),
            "clean_elections_index": sanitize(r.get("clean_elections_index")),
            "polyarchy_index": sanitize(r.get("polyarchy_index")),
            "regime_type": sanitize(r.get("regime_type")),
            "regime_name": str(r["regime_name"])
        })
    vdem_by_year[int(yr)] = clean_records

with open(os.path.join(OUTPUT_DIR, "vdem_map.json"), "w") as f:
    json.dump(sanitize_dict(vdem_by_year), f)

def get_feature_info(feature_name):
    info_map = {
        "constitutional_rigidity": ("Constitutional Rigidity Index", "Judicial and legislative institutional constraints (Tsebelis, 2002)."),
        "polyarchy_index": ("Polyarchy Electoral Democracy Index", "Electoral competitiveness and civil liberties benchmark."),
        "clean_elections_index": ("Clean Elections Index", "Absence of electoral fraud and intimidation (V-Dem)."),
        "liberal_component_index": ("Liberal Component Index", "Rule of law and executive oversight constraints."),
        "judicial_independence": ("Judicial Independence Score", "Autonomy of high courts from political intervention."),
        "gdp_per_capita": ("GDP Per Capita (USD)", "Level of economic development and living standards."),
        "gdp_growth": ("Annual Real GDP Growth (%)", "Short-term macroeconomic momentum (Economic Voting Theory)."),
        "inflation_rate": ("Consumer Price Inflation (%)", "Erosion of purchasing power and price instability."),
        "trade_pct_gdp": ("Trade Openness (% GDP)", "Exposure to global economic and supply chain shocks."),
        "gdelt_event_count": ("Protest & Conflict Media Salience", "Decayed 12-month GDELT news volume on domestic unrest."),
        "gdelt_avg_goldstein": ("Media Tone Goldstein Score", "Net positive or negative tone in global news coverage."),
        "spatial_lag_incumbent_win": ("Geographic Neighbor Incumbent Win Rate", "Incumbent victory rate in contiguous border states."),
        "spatial_lag_democracy_delta": ("Regional Democratic Shift (Contagion)", "Net democratisation or backsliding in neighboring states."),
        "coalition_size": ("Governing Coalition Member Count", "Number of formal political parties in cabinet/government."),
        "is_coalition": ("Coalition Government Flag", "Multi-party governance indicator (1 = Coalition, 0 = Single Party)."),
        "cabinet_ideology_dispersion": ("Cabinet Ideological Polarisation", "Standard deviation of left-right positions among cabinet parties."),
        "effective_parties_clea": ("Effective Number of Parliamentary Parties (ENP)", "Party system fragmentation index (Laakso & Taagepera, 1979)."),
        "polarization_vdem": ("Societal Political Polarisation", "V-Dem assessment of partisan antagonism in civil society."),
        "prior_incumbent_seat_share": ("Prior Incumbent Seat Share (%)", "Share of legislative seats held prior to election."),
        "prior_margin_victory": ("Prior Election Margin of Victory (%)", "Difference between top two parties in prior election."),
        "years_in_power": ("Incumbent Tenure in Power (Years)", "Executive fatigue and incumbency advantage duration."),
        "term_limit_nearing": ("Term Limit Expiration Impending", "Chief executive facing statutory term limits."),
        "executive_decrees_per_year": ("Executive Decree Frequency", "Unilateral executive rule frequency index."),
        "state_media_bias": ("State Media Bias Score", "Favoritism toward governing party in state-owned broadcasting."),
        "opposition_repress_index": ("Opposition Repression Index", "Government harassment or disqualification of opponents."),
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
        "polarization_fragmentation_interaction": ("Polarization x Party Fragmentation", "Ideological polarization combined with high party count.")
    }
    if feature_name in info_map:
        return info_map[feature_name]
    elif feature_name.startswith("country_"):
        ccode = feature_name.replace("country_", "")
        return (f"Country Baseline Factor ({ccode})", "Fixed effect baseline adjustment for country historical tendencies.")
    else:
        clean_name = feature_name.replace("_", " ").title()
        return (clean_name, "Institutional and macroeconomic predictive indicator.")

# 7. Feature Importances JSON
from models.correlation_models import get_feature_importance_report
df_modeling = cache_data["df_modeling"]
imp_df = get_feature_importance_report(df_modeling).head(10)

importances = []
for _, row in imp_df.iterrows():
    feat_name = row["feature"]
    human_name, lit_desc = get_feature_info(feat_name)
    importances.append({
        "feature": str(feat_name),
        "human_name": str(human_name),
        "importance": round(float(row["importance"]), 4),
        "explanation": str(lit_desc)
    })

with open(os.path.join(OUTPUT_DIR, "feature_importances.json"), "w") as f:
    json.dump(sanitize_dict(importances), f, indent=2)

# 8. SHAP Explanations JSON (precompute for all elections!)
print("Precomputing SHAP explanations for all elections...")
import xgboost as xgb
import catboost as cb
from models.correlation_models import FEATURES, EXEC_SPECIFIC, LEG_SPECIFIC, TUNED_PARAMS

country_dummies = [col for col in df_modeling.columns if col.startswith("country_") and len(col) == 11 and col[8:].isupper()]
exec_features = [f for f in FEATURES if f not in LEG_SPECIFIC and not (f.startswith("country_") and len(f) == 11 and f[8:].isupper())] + country_dummies
leg_features = [f for f in FEATURES if f not in EXEC_SPECIFIC and not (f.startswith("country_") and len(f) == 11 and f[8:].isupper())] + country_dummies

exec_params = TUNED_PARAMS.get("executive_model", {"xgb_weight": 0.7})
leg_params = TUNED_PARAMS.get("legislative_model", {"xgb_weight": 0.9})

shap_dict = {}
for _, row in df_modeling.iterrows():
    eid = str(row["election_id"])
    c_code = str(row["country_code"])
    yr = int(row["year"])
    is_pres = row["is_presidential"]
    clean_idx = row["clean_index"]
    is_clean = pd.notna(clean_idx) and clean_idx >= 0.65

    etype = "Executive" if is_pres == 1 else "Legislative"
    if "hybrid_model" in cache_data:
        hybrid_model = cache_data["hybrid_model"]
        if is_pres == 1:
            cur_features = exec_features
            cur_w_xgb = exec_params.get("xgb_weight", 0.7)
            if is_clean and hybrid_model.ex_reg_xgb is not None:
                model_xgb = hybrid_model.ex_reg_xgb
                model_cb = hybrid_model.ex_reg_cb
            else:
                model_xgb = hybrid_model.ex_unclean_xgb
                model_cb = hybrid_model.ex_unclean_cb
        else:
            cur_features = leg_features
            cur_w_xgb = leg_params.get("xgb_weight", 0.9)
            if is_clean and hybrid_model.leg_reg_xgb is not None:
                model_xgb = hybrid_model.leg_reg_xgb
                model_cb = hybrid_model.leg_reg_cb
            else:
                model_xgb = hybrid_model.leg_unclean_xgb
                model_cb = hybrid_model.leg_unclean_cb
    else:
        if is_pres == 1:
            cur_features = exec_features
            cur_w_xgb = exec_params.get("xgb_weight", 0.7)
            model_xgb = cache_data["ex_xgb"]
            model_cb = cache_data["ex_cb"]
        else:
            cur_features = leg_features
            cur_w_xgb = leg_params.get("xgb_weight", 0.9)
            model_xgb = cache_data["leg_xgb"]
            model_cb = cache_data["leg_cb"]

    x_vec = row[cur_features].to_frame().T.astype(float)
    x_imputed = x_vec.fillna(0.0)

    try:
        contrib_xgb = model_xgb.get_booster().predict(xgb.DMatrix(x_imputed), pred_contribs=True)[0][:-1]
        contrib_cb = model_cb.get_feature_importance(cb.Pool(x_imputed), type="ShapValues")[0][:-1]
        combo_contribs = (cur_w_xgb * contrib_xgb) + ((1.0 - cur_w_xgb) * contrib_cb)
    except Exception:
        combo_contribs = np.zeros(len(cur_features))

    feat_tuples = []
    for f_idx, f_name in enumerate(cur_features):
        c_val = combo_contribs[f_idx]
        raw_val = row[f_name]
        feat_tuples.append((f_name, c_val, raw_val))

    feat_tuples.sort(key=lambda x: abs(x[1]), reverse=True)

    top_feats = []
    for f_name, c_val, raw_val in feat_tuples[:10]:
        h_name, lit_info = get_feature_info(f_name)
        direction = "Favors Incumbent" if c_val > 0 else "Favors Challenger"
        top_feats.append({
            "feature": str(f_name),
            "human_name": str(h_name),
            "literature_info": str(lit_info),
            "value": sanitize(raw_val),
            "contribution": round(float(c_val), 4),
            "direction": direction
        })

    pred_match = merged_preds[merged_preds["election_id"] == eid]
    pred_prob = float(pred_match["raw_probability"].iloc[0]) if not pred_match.empty else 0.5
    pred_winner = "Incumbent" if pred_prob >= 0.5 else "Challenger"

    actual_outcome_str = None
    if not pred_match.empty and pd.notna(pred_match["target_outcome"].iloc[0]):
        actual_outcome_str = "Incumbent Victory" if pred_match["target_outcome"].iloc[0] == 1 else "Challenger Victory"

    shap_dict[eid] = {
        "election_id": eid,
        "country_code": c_code,
        "year": yr,
        "election_type": etype,
        "predicted_winner": pred_winner,
        "predicted_probability": round(pred_prob, 4),
        "actual_outcome": actual_outcome_str,
        "is_clean": is_clean,
        "top_features": top_feats
    }

with open(os.path.join(OUTPUT_DIR, "shap_explanations.json"), "w") as f:
    json.dump(sanitize_dict(shap_dict), f)

print(f"Successfully exported all precomputed JSON files into {OUTPUT_DIR}!")
