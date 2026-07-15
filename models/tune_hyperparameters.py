import os
import sqlite3
import pandas as pd
import numpy as np
import json
import itertools
import math
from datetime import datetime
import xgboost as xgb
import catboost as cb
from sklearn.metrics import accuracy_score, log_loss

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "elections.db")
RAW_DIR = os.path.join(BASE_DIR, "raw_data")
CONTIGUITY_PATH = os.path.join(RAW_DIR, "contdird.csv")
MAP_PATH = os.path.join(BASE_DIR, "data_ingest", "cow_to_iso.json")

import sys
sys.path.insert(0, BASE_DIR)
from models.correlation_models import (
    build_modeling_dataset,
    FEATURES,
    BASE_FEATURES,
    calibrate_probability,
    EXEC_SPECIFIC,
    LEG_SPECIFIC
)

# We will group features for completeness calculations
VDEM_FEATURES = {
    "clean_index", "polyarchy_index", "clean_index_delta_1yr", "clean_index_delta_3yr",
    "polyarchy_delta_1yr", "polyarchy_delta_3yr"
}

PRIOR_FEATURES = {
    "prior_incumbent_seat_share", "prior_margin_of_victory", "country_incumbent_win_rate",
    "incumbent_term_count", "prior_margin_presidential_interaction", "party_fragmentation_legislative_interaction"
}

GDELT_CORE_FEATURES = {
    "average_news_tone", "total_news_volume", "protest_pressure", "coop_score",
    "conflict_level", "coercion_ratio", "aid_ratio", "protest_velocity", "conflict_velocity"
}

MACRO_FEATURES = {
    "oil_price_shock", "oil_exporter_shock", "fed_rate_change", "us_president_ideology"
}

def compute_data_completeness_custom(df, features, category_multipliers):
    """Compute data completeness vectorised over the entire DataFrame."""
    # Pre-extract values to avoid Series lookup overhead
    vdem_mask = [f in VDEM_FEATURES for f in features]
    prior_mask = [f in PRIOR_FEATURES for f in features]
    gdelt_mask = [f in GDELT_CORE_FEATURES for f in features]
    macro_mask = [f in MACRO_FEATURES for f in features]
    spatial_mask = ["spatial_lag_" in f for f in features]
    
    # Baseline weights
    base_weights = {
        "clean_index": 2.0, "polyarchy_index": 2.0,
        "prior_incumbent_seat_share": 2.0, "prior_margin_of_victory": 1.5, "country_incumbent_win_rate": 1.5,
        "incumbent_term_count": 1.0, "average_news_tone": 1.0, "total_news_volume": 1.0,
        "protest_pressure": 1.0, "coop_score": 1.0, "conflict_level": 1.0, "coercion_ratio": 1.0,
        "aid_ratio": 1.0, "protest_velocity": 1.0, "conflict_velocity": 1.0,
        "clean_index_delta_1yr": 1.5, "clean_index_delta_3yr": 1.5,
        "polyarchy_delta_1yr": 1.5, "polyarchy_delta_3yr": 1.5,
        "prior_margin_presidential_interaction": 1.5, "party_fragmentation_legislative_interaction": 1.5,
        "oil_price_shock": 1.0, "oil_exporter_shock": 1.0, "fed_rate_change": 1.0, "us_president_ideology": 1.0
    }
    
    weights = np.array([base_weights.get(f, 0.1) for f in features])
    
    # Apply category multipliers
    for idx, f in enumerate(features):
        if vdem_mask[idx]:
            weights[idx] *= category_multipliers.get("m_vdem", 1.0)
        elif prior_mask[idx]:
            weights[idx] *= category_multipliers.get("m_prior", 1.0)
        elif gdelt_mask[idx]:
            weights[idx] *= category_multipliers.get("m_gdelt", 1.0)
        elif macro_mask[idx]:
            weights[idx] *= category_multipliers.get("m_macro", 1.0)
        elif spatial_mask[idx]:
            weights[idx] *= category_multipliers.get("m_spatial", 1.0)
            
    total_weight = weights.sum()
    if total_weight == 0:
        return np.zeros(len(df))
        
    # Get not-null matrix
    not_null = df[features].notna().values.astype(float)
    
    # Apply gdelt coverage factor
    gdelt_indices = [idx for idx, f in enumerate(features) if gdelt_mask[idx]]
    if gdelt_indices:
        gdelt_coverage = df["gdelt_coverage"].fillna(0.0).values[:, np.newaxis]
        not_null[:, gdelt_indices] *= gdelt_coverage
        
    weighted_sums = not_null.dot(weights)
    return weighted_sums / total_weight


def tune_pipeline():
    conn = sqlite3.connect(DB_PATH)
    print("Loading base dataset (V-Dem, GDELT, and global features)...")
    df_base = build_modeling_dataset(conn, include_upcoming=True)
    
    # Load COW-to-ISO mapping
    with open(MAP_PATH) as f:
        cow_to_iso = {int(k): v for k, v in json.load(f).items()}
    manual_overrides = {
        31: "BHS", 54: "DMA", 55: "GRD", 56: "LCA", 57: "VCT", 58: "ATG", 60: "KNA",
        80: "BLZ", 221: "MCO", 223: "LIE", 232: "AND", 325: "SMR", 340: "SRB", 345: "SRB",
        940: "NRU", 946: "KIR", 947: "TUV", 955: "TON", 983: "MHL", 986: "PLW",
        987: "FSM", 990: "WSM", 835: "BRN"
    }
    for k, v in manual_overrides.items():
        cow_to_iso[k] = v
        
    cursor = conn.cursor()
    cursor.execute("SELECT country_code FROM countries;")
    valid_codes = {row[0] for row in cursor.fetchall()}
    
    # Load trade, alliance, unga_voting weights from spatial_weights for cascade unreliability check
    cursor.execute("SELECT country_code_source, country_code_target, weight_type, year, weight_value FROM spatial_weights WHERE weight_type IN ('trade', 'alliance');")
    other_weights_cache = {}
    for src, tgt, wt_type, yr, val in cursor.fetchall():
        if wt_type not in other_weights_cache:
            other_weights_cache[wt_type] = {}
        if yr not in other_weights_cache[wt_type]:
            other_weights_cache[wt_type][yr] = {}
        if src not in other_weights_cache[wt_type][yr]:
            other_weights_cache[wt_type][yr][src] = {}
        other_weights_cache[wt_type][yr][src][tgt] = val
        
    # Load contiguity raw data
    print("Loading raw contiguity data...")
    df_cont = pd.read_csv(CONTIGUITY_PATH)
    df_cont = df_cont[(df_cont["year"] >= 1990) & (df_cont["year"] <= 2028)]
    max_year = df_cont["year"].max()
    if max_year < 2028:
        df_latest = df_cont[df_cont["year"] == max_year]
        extra_dfs = []
        for yr in range(max_year + 1, 2029):
            df_yr = df_latest.copy()
            df_yr["year"] = yr
            extra_dfs.append(df_yr)
        df_cont = pd.concat([df_cont] + extra_dfs, ignore_index=True)

    # Pre-parse contiguity raw data as list of tuples once outside the loop for maximum speed
    print("Pre-parsing contiguity raw data...")
    contiguity_raw_list = []
    for _, row in df_cont.iterrows():
        yr = int(row["year"])
        src_cow = int(row["state1no"])
        tgt_cow = int(row["state2no"])
        src_iso = cow_to_iso.get(src_cow)
        tgt_iso = cow_to_iso.get(tgt_cow)
        
        if src_iso and tgt_iso and src_iso in valid_codes and tgt_iso in valid_codes:
            if src_iso == tgt_iso:
                continue
            contiguity_raw_list.append((yr, src_iso, tgt_iso, int(row["conttype"])))

    # Pre-index target outcomes for extremely fast spatial lag calculations
    # Store elections by year in a list to preserve multiple elections per year per country
    elections_by_year = {}
    for _, row in df_base.iterrows():
        y = int(row["year"])
        if y not in elections_by_year:
            elections_by_year[y] = []
        elections_by_year[y].append({
            "country_code": row["country_code"],
            "clean_index": row.get("clean_index"),
            "target_outcome": row.get("target_outcome"),
            "protest_pressure": row.get("protest_pressure"),
            "conflict_level": row.get("conflict_level")
        })

    def compute_contiguity_lags_in_memory(df_target, cont_weights):
        cont_lookup = {}
        for yr, src_iso, tgt_iso, conttype in contiguity_raw_list:
            w = cont_weights.get(conttype, 0.0)
            if w > 0.0:
                if yr not in cont_lookup:
                    cont_lookup[yr] = {}
                if src_iso not in cont_lookup[yr]:
                    cont_lookup[yr][src_iso] = {}
                cont_lookup[yr][src_iso][tgt_iso] = w

        lag_clean, lag_outcome, lag_protest, lag_conflict = [], [], [], []
        
        for _, row in df_target.iterrows():
            source_code = row["country_code"]
            yr = int(row["year"])
            
            src_weights = cont_lookup.get(yr, {}).get(source_code, {})
            if not src_weights:
                lag_clean.append(np.nan)
                lag_outcome.append(np.nan)
                lag_protest.append(np.nan)
                lag_conflict.append(np.nan)
                continue
                
            total_w_c = 0.0
            sum_w_c = 0.0
            total_w_o = 0.0
            sum_w_o = 0.0
            total_w_p = 0.0
            sum_w_p = 0.0
            total_w_f = 0.0
            sum_w_f = 0.0
            
            # Find all target elections in [yr-4, yr]
            for past_yr in range(yr - 4, yr + 1):
                for other in elections_by_year.get(past_yr, []):
                    tgt = other["country_code"]
                    if tgt in src_weights and tgt != source_code:
                        w = src_weights[tgt]
                        
                        c_val = other["clean_index"]
                        if c_val is not None and pd.notna(c_val):
                            sum_w_c += w * c_val
                            total_w_c += w
                            
                        o_val = other["target_outcome"]
                        if o_val is not None and pd.notna(o_val):
                            sum_w_o += w * o_val
                            total_w_o += w
                            
                        p_val = other["protest_pressure"]
                        if p_val is not None and pd.notna(p_val):
                            sum_w_p += w * p_val
                            total_w_p += w
                            
                        f_val = other["conflict_level"]
                        if f_val is not None and pd.notna(f_val):
                            sum_w_f += w * f_val
                            total_w_f += w
                            
            lag_clean.append(sum_w_c / total_w_c if total_w_c > 0 else np.nan)
            lag_outcome.append(sum_w_o / total_w_o if total_w_o > 0 else np.nan)
            lag_protest.append(sum_w_p / total_w_p if total_w_p > 0 else np.nan)
            lag_conflict.append(sum_w_f / total_w_f if total_w_f > 0 else np.nan)
            
        return lag_clean, lag_outcome, lag_protest, lag_conflict

    # Search Space definitions
    grid_xgb_depth = [2, 3, 4]
    grid_xgb_lr = [0.03, 0.06, 0.10]
    grid_cb_depth = [3, 4, 5]
    grid_cb_lr = [0.03, 0.05, 0.10]
    
    grid_w2_w3_w4_w5 = [
        # (w2, w3, w4, w5)
        (1.0, 0.5, 0.3, 0.1), # Default
        (0.8, 0.4, 0.2, 0.0), # Stricter water contiguity decay
        (1.0, 0.6, 0.4, 0.2), # Wider water contiguity influence
        (0.5, 0.2, 0.1, 0.0), # Land-dominated (water borders strongly penalized)
        (0.3, 0.1, 0.0, 0.0), # Very land-dominated
        (0.2, 0.0, 0.0, 0.0)  # Extremely land-dominated (only immediate land + very narrow water)
    ]
    
    grid_completeness = [
        # (m_vdem, m_prior, m_gdelt, m_spatial, m_macro)
        (1.0, 1.0, 1.0, 1.0, 1.0), # Default
        (2.0, 1.0, 1.0, 0.5, 1.0), # Prioritize institutional (V-Dem)
        (1.0, 2.0, 0.5, 0.5, 1.0), # Prioritize election outcomes
        (1.0, 1.0, 1.5, 0.5, 1.0)  # Prioritize contemporary news
    ]
    
    grid_cascade = [
        # (cascade_coeff, ff_self, ff_spatial, ff_region)
        (0.3, 0.15, 0.15, 0.10), # Default
        (0.2, 0.10, 0.10, 0.05), # Mild unreliability penalty
        (0.4, 0.20, 0.20, 0.15)  # Heavy unreliability penalty (conservative)
    ]
    
    grid_ensemble = [
        # (xgb_weight, T)
        (0.4, 1.10),
        (0.4, 1.20),
        (0.5, 1.15),
        (0.5, 1.20),
        (0.6, 1.10)
    ]
    
    origins = [2010, 2014, 2018, 2022]
    
    best_acc = 0.0
    best_loss = 999.0
    best_config = None
    
    print("\nStarting Parameter Tuning Search Loop...")
    
    # We will search over a diverse set of combinations
    combinations = list(itertools.product(
        grid_w2_w3_w4_w5,
        grid_completeness,
        grid_cascade,
        grid_ensemble,
        grid_xgb_depth,
        grid_xgb_lr,
        grid_cb_depth,
        grid_cb_lr
    ))
    
    # Let's sample a randomized subset to keep runtime reasonable (e.g. 80 iterations)
    np.random.seed(42)
    sample_indices = np.random.choice(len(combinations), size=min(120, len(combinations)), replace=False)
    
    # Make sure default configuration is tested
    default_config = (
        (1.0, 0.5, 0.3, 0.1),
        (1.0, 1.0, 1.0, 1.0, 1.0),
        (0.3, 0.15, 0.15, 0.10),
        (0.5, 0.6),
        3, 0.06, 4, 0.05
    )
    
    # Initialize run list
    configs_to_run = [default_config]
    for idx in sample_indices:
        config = combinations[idx]
        if config != default_config:
            configs_to_run.append(config)
            
    print(f"Total configurations to evaluate: {len(configs_to_run)}")
    
    for run_idx, (cont_w, comp_m, casc_p, ens_b, x_depth, x_lr, c_depth, c_lr) in enumerate(configs_to_run):
        # 1. Update contiguity weights in memory
        cont_weights = {1: 1.0, 2: cont_w[0], 3: cont_w[1], 4: cont_w[2], 5: cont_w[3]}
        c, o, p, f = compute_contiguity_lags_in_memory(df_base, cont_weights)
        
        df_iter = df_base.copy()
        df_iter["spatial_lag_clean_contiguity"] = c
        df_iter["spatial_lag_outcome_contiguity"] = o
        df_iter["spatial_lag_protest_contiguity"] = p
        df_iter["spatial_lag_conflict_contiguity"] = f
        
        # 2. Recompute data completeness in memory
        category_multipliers = {
            "m_vdem": comp_m[0], "m_prior": comp_m[1], "m_gdelt": comp_m[2],
            "m_spatial": comp_m[3], "m_macro": comp_m[4]
        }
        df_iter["data_completeness"] = compute_data_completeness_custom(df_iter, FEATURES, category_multipliers)
        
        # Only train/test on historical records
        df_known = df_iter[df_iter["target_outcome"].notna()].copy().reset_index(drop=True)
        
        cv_accuracies = []
        cv_loglosses = []
        
        # 3. Rolling Origin Cross Validation
        # Extract country dummies dynamically to prevent duplication issues
        country_dummies = [col for col in df_known.columns if col.startswith("country_") and len(col) == 11 and col[8:].isupper()]
        exec_features = [f for f in FEATURES if f not in LEG_SPECIFIC and not (f.startswith("country_") and len(f) == 11 and f[8:].isupper())] + country_dummies
        leg_features = [f for f in FEATURES if f not in EXEC_SPECIFIC and not (f.startswith("country_") and len(f) == 11 and f[8:].isupper())] + country_dummies

        for y in origins:
            train_df = df_known[df_known["year"] <= y]
            test_df = df_known[(df_known["year"] > y) & (df_known["year"] <= y + 4)].copy()
            
            if train_df.empty or test_df.empty:
                continue
                
            train_exec = train_df[train_df["is_presidential"] == 1]
            train_leg = train_df[train_df["is_presidential"] == 0]
            
            test_exec = test_df[test_df["is_presidential"] == 1].copy()
            test_leg = test_df[test_df["is_presidential"] == 0].copy()
            
            # Predict for Executive
            if not test_exec.empty and not train_exec.empty:
                model_ex_xgb = xgb.XGBClassifier(
                    max_depth=x_depth, learning_rate=x_lr, n_estimators=150, eval_metric='logloss', random_state=42
                ).fit(train_exec[exec_features], train_exec["target_outcome"])
                
                model_ex_cb = cb.CatBoostClassifier(
                    iterations=200, depth=c_depth, learning_rate=c_lr, verbose=0, random_seed=42
                ).fit(train_exec[exec_features], train_exec["target_outcome"])
                
                p_ex_xgb = model_ex_xgb.predict_proba(test_exec[exec_features])[:, 1]
                p_ex_cb = model_ex_cb.predict_proba(test_exec[exec_features])[:, 1]
                
                xgb_w, T_val = ens_b
                p_ens = xgb_w * p_ex_xgb + (1 - xgb_w) * p_ex_cb
                p_cal = np.array([calibrate_probability(p, T=T_val) for p in p_ens])
                test_exec["raw_prob"] = p_cal
            else:
                test_exec["raw_prob"] = []
                
            # Predict for Legislative
            if not test_leg.empty and not train_leg.empty:
                model_leg_xgb = xgb.XGBClassifier(
                    max_depth=x_depth, learning_rate=x_lr, n_estimators=150, eval_metric='logloss', random_state=42
                ).fit(train_leg[leg_features], train_leg["target_outcome"])
                
                model_leg_cb = cb.CatBoostClassifier(
                    iterations=200, depth=c_depth, learning_rate=c_lr, verbose=0, random_seed=42
                ).fit(train_leg[leg_features], train_leg["target_outcome"])
                
                p_leg_xgb = model_leg_xgb.predict_proba(test_leg[leg_features])[:, 1]
                p_leg_cb = model_leg_cb.predict_proba(test_leg[leg_features])[:, 1]
                
                xgb_w, T_val = ens_b
                p_ens = xgb_w * p_leg_xgb + (1 - xgb_w) * p_leg_cb
                p_cal = np.array([calibrate_probability(p, T=T_val) for p in p_ens])
                test_leg["raw_prob"] = p_cal
            else:
                test_leg["raw_prob"] = []
                
            # Reassemble test_df with probabilities
            test_combined = pd.concat([test_exec, test_leg]).sort_index()
            
            # Apply unreliability/cascade penalties to test probabilities
            adjusted_probs = []
            cascade_coeff, ff_self, ff_spatial, ff_region = casc_p
            
            for test_idx, row in test_combined.iterrows():
                source_code = row["country_code"]
                yr = int(row["year"])
                raw_prob = row["raw_prob"]
                completeness = row["data_completeness"]
                
                # Check for spatial lag cascade unreliability
                total_weight_sum = 0.0
                predicted_weight_sum = 0.0
                
                for wt in ["trade", "alliance"]:
                    src_weights = other_weights_cache.get(wt, {}).get(yr, {}).get(source_code, {})
                    if src_weights:
                        matching_others = test_combined[test_combined["country_code"].isin(src_weights)]
                        for _, other in matching_others.iterrows():
                            tgt = other["country_code"]
                            w = src_weights[tgt]
                            total_weight_sum += w
                            prob_tgt = other["raw_prob"]
                            conf = max(prob_tgt, 1 - prob_tgt)
                            predicted_weight_sum += w * (1.0 - conf)
                            
                if total_weight_sum > 0:
                    avg_unreliability = predicted_weight_sum / total_weight_sum
                    cascade_penalty = 1.0 - cascade_coeff * avg_unreliability
                else:
                    cascade_penalty = 1.0
                    
                K = completeness * cascade_penalty
                prob_adj = 0.5 + (raw_prob - 0.5) * K
                prob_adj = np.clip(prob_adj, 1e-15, 1 - 1e-15)
                adjusted_probs.append(prob_adj)
                
            acc = accuracy_score(test_combined["target_outcome"], (test_combined["raw_prob"] >= 0.50).astype(int))
            loss = log_loss(test_combined["target_outcome"], adjusted_probs)
            
            cv_accuracies.append(acc)
            cv_loglosses.append(loss)
            
        avg_acc = np.mean(cv_accuracies)
        avg_loss = np.mean(cv_loglosses)
        
        is_default = (cont_w, comp_m, casc_p, ens_b, x_depth, x_lr, c_depth, c_lr) == default_config
        label = " [DEFAULT]" if is_default else ""
        
        # Print progress every 10 runs
        if run_idx % 10 == 0 or is_default:
            print(f"Config {run_idx:3d}{label}: Acc={avg_acc:.4%} | Log-Loss={avg_loss:.4f}")
            
        # We want to maximize accuracy, and break ties with log-loss
        if avg_acc > best_acc or (np.isclose(avg_acc, best_acc) and avg_loss < best_loss):
            best_acc = avg_acc
            best_loss = avg_loss
            best_config = {
                "contiguity_weights": cont_weights,
                "category_multipliers": category_multipliers,
                "cascade_parameters": {
                    "cascade_coeff": cascade_coeff,
                    "ff_self": ff_self,
                    "ff_spatial": ff_spatial,
                    "ff_region": ff_region
                },
                "ensemble_balance": {
                    "xgb_weight": xgb_w,
                    "T": T_val
                },
                "model_parameters": {
                    "xgb_max_depth": x_depth,
                    "xgb_lr": x_lr,
                    "cb_depth": c_depth,
                    "cb_lr": c_lr
                }
            }
            
    print("\n--- Tuning Optimization Completed ---")
    print(f"Optimal Out-of-Sample Accuracy (Default Threshold): {best_acc:.2%}")
    print(f"Optimal Out-of-Sample Log-Loss: {best_loss:.4f}")
    
    print("\n--- Optimizing Classification Decision Threshold ---")
    # Retrieve best parameters
    best_c_weights = best_config["contiguity_weights"]
    best_c_weights = {int(k): float(v) for k, v in best_c_weights.items()}
    best_category_multipliers = best_config["category_multipliers"]
    best_model_params = best_config["model_parameters"]
    best_ensemble_balance = best_config["ensemble_balance"]
    
    # Re-build dataset with best weights in memory
    df_best = df_base.copy()
    c, o, p, f = compute_contiguity_lags_in_memory(df_best, best_c_weights)
    df_best["spatial_lag_clean_contiguity"] = c
    df_best["spatial_lag_outcome_contiguity"] = o
    df_best["spatial_lag_protest_contiguity"] = p
    df_best["spatial_lag_conflict_contiguity"] = f
    
    df_best["data_completeness"] = compute_data_completeness_custom(df_best, FEATURES, best_category_multipliers)
    
    df_known_best = df_best[df_best["target_outcome"].notna()].copy().reset_index(drop=True)
    X_known = df_known_best[FEATURES]
    y_known = df_known_best["target_outcome"]
    
    # 5-fold CV to collect out-of-sample probabilities
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    oos_probs = np.zeros(len(df_known_best))
    
    country_dummies = [col for col in df_known_best.columns if col.startswith("country_") and len(col) == 11 and col[8:].isupper()]
    exec_features = [f for f in FEATURES if f not in LEG_SPECIFIC and not (f.startswith("country_") and len(f) == 11 and f[8:].isupper())] + country_dummies
    leg_features = [f for f in FEATURES if f not in EXEC_SPECIFIC and not (f.startswith("country_") and len(f) == 11 and f[8:].isupper())] + country_dummies
    
    # We calibrate probability locally
    def local_cal(pr, T_val):
        pr = np.clip(pr, 1e-5, 1.0 - 1e-5)
        logit = np.log(pr / (1.0 - pr))
        return 1.0 / (1.0 + np.exp(-logit / T_val))
        
    for train_idx, val_idx in kf.split(df_known_best):
        train_df_cv = df_known_best.iloc[train_idx]
        val_df_cv = df_known_best.iloc[val_idx]
        
        train_exec = train_df_cv[train_df_cv["is_presidential"] == 1]
        train_leg = train_df_cv[train_df_cv["is_presidential"] == 0]
        
        val_exec = val_df_cv[val_df_cv["is_presidential"] == 1]
        val_leg = val_df_cv[val_df_cv["is_presidential"] == 0]
        
        # Fit Executive model
        if not val_exec.empty and not train_exec.empty:
            m_exec_cb = cb.CatBoostClassifier(
                iterations=200, depth=best_model_params["cb_depth"], learning_rate=best_model_params["cb_lr"], verbose=0, random_seed=42
            ).fit(train_exec[exec_features], train_exec["target_outcome"])
            
            m_exec_xgb = xgb.XGBClassifier(
                max_depth=best_model_params["xgb_max_depth"], learning_rate=best_model_params["xgb_lr"], n_estimators=150, eval_metric='logloss', random_state=42
            ).fit(train_exec[exec_features], train_exec["target_outcome"])
            
            p_ex_xgb = m_exec_xgb.predict_proba(val_exec[exec_features])[:, 1]
            p_ex_cb = m_exec_cb.predict_proba(val_exec[exec_features])[:, 1]
            
            xgb_w = best_ensemble_balance["xgb_weight"]
            T_val = best_ensemble_balance["T"]
            p_ens = xgb_w * p_ex_xgb + (1.0 - xgb_w) * p_ex_cb
            p_cal = np.array([local_cal(pr, T_val) for pr in p_ens])
            
            for idx_in_val, val_idx_global in enumerate(val_exec.index):
                oos_probs[val_idx_global] = p_cal[idx_in_val]
                
        # Fit Legislative model
        if not val_leg.empty and not train_leg.empty:
            m_leg_cb = cb.CatBoostClassifier(
                iterations=200, depth=best_model_params["cb_depth"], learning_rate=best_model_params["cb_lr"], verbose=0, random_seed=42
            ).fit(train_leg[leg_features], train_leg["target_outcome"])
            
            m_leg_xgb = xgb.XGBClassifier(
                max_depth=best_model_params["xgb_max_depth"], learning_rate=best_model_params["xgb_lr"], n_estimators=150, eval_metric='logloss', random_state=42
            ).fit(train_leg[leg_features], train_leg["target_outcome"])
            
            p_leg_xgb = m_leg_xgb.predict_proba(val_leg[leg_features])[:, 1]
            p_leg_cb = m_leg_cb.predict_proba(val_leg[leg_features])[:, 1]
            
            xgb_w_lg = best_ensemble_balance["xgb_weight"]
            T_val_lg = best_ensemble_balance["T"]
            p_ens = xgb_w_lg * p_leg_xgb + (1.0 - xgb_w_lg) * p_leg_cb
            p_cal = np.array([local_cal(pr, T_val_lg) for pr in p_ens])
            
            for idx_in_val, val_idx_global in enumerate(val_leg.index):
                oos_probs[val_idx_global] = p_cal[idx_in_val]
                
    # Grid search for threshold
    thresholds = np.arange(0.40, 0.60, 0.005)
    best_thresh = 0.50
    best_thresh_acc = 0.0
    
    y_true_vals = y_known.values.astype(int)
    for t in thresholds:
        preds_t = (oos_probs >= t).astype(int)
        acc_t = accuracy_score(y_true_vals, preds_t)
        if acc_t > best_thresh_acc:
            best_thresh_acc = acc_t
            best_thresh = t
            
    print(f"Optimal Decision Threshold: {best_thresh:.3f} (OOS Accuracy: {best_thresh_acc:.2%})")
    
    # Save the parameters in split format so they align with correlation_models.py and the dashboard!
    xgb_w = best_ensemble_balance["xgb_weight"]
    T_val = best_ensemble_balance["T"]
    
    best_config["executive_model"] = {
        "xgb_weight": float(xgb_w),
        "T": float(T_val),
        "classification_threshold": float(round(best_thresh, 3))
    }
    best_config["legislative_model"] = {
        "xgb_weight": float(best_ensemble_balance["xgb_weight"]),
        "T": float(best_ensemble_balance["T"]),
        "classification_threshold": float(round(best_thresh, 3))
    }
    best_config["classification_threshold"] = float(round(best_thresh, 3))
    
    print("\nBest Parameter Configuration:")
    print(json.dumps(best_config, indent=4))
    
    # Save the configuration
    output_path = os.path.join(BASE_DIR, "models", "tuned_parameters.json")
    with open(output_path, "w") as f:
        json.dump(best_config, f, indent=4)
    print(f"Saved optimal parameters to: {output_path}")
    conn.close()

if __name__ == "__main__":
    tune_pipeline()
