import os
# Prevent OpenMP conflict issues on macOS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
import sqlite3
import math
import pandas as pd
import numpy as np
from datetime import datetime
from esda.moran import Moran, Moran_Local
import xgboost as xgb
import catboost as cb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import warnings

# Add workspace to system path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Ignore warnings for clean output
warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "elections.db")

# Load tuned parameters from models/tuned_parameters.json dynamically
def load_tuned_parameters():
    """Load tuned parameters from tuned_parameters.json, fallback to defaults."""
    default_params = {
        "ensemble_balance": {
            "xgb_weight": 0.40,
            "T": 0.70
        },
        "model_parameters": {
            "xgb_max_depth": 3,
            "xgb_lr": 0.03,
            "cb_depth": 5,
            "cb_lr": 0.03
        },
        "executive_model": {
            "xgb_weight": 0.9,
            "T": 0.78,
            "classification_threshold": 0.50
        },
        "legislative_model": {
            "xgb_weight": 0.5,
            "T": 0.80,
            "classification_threshold": 0.50
        }
    }
    
    this_dir = os.path.dirname(os.path.abspath(__file__))
    tuned_path = os.path.join(this_dir, "tuned_parameters.json")
    if os.path.exists(tuned_path):
        import json
        try:
            with open(tuned_path, "r") as f:
                params = json.load(f)
                for k, v in default_params.items():
                    if k not in params:
                        params[k] = v
                    elif isinstance(v, dict):
                        for sub_k, sub_v in v.items():
                            if sub_k not in params[k]:
                                params[k][sub_k] = sub_v
                return params
        except Exception as e:
            pass
    return default_params

TUNED_PARAMS = load_tuned_parameters()

def load_weights_cache(cursor):
    # Pre-cache non-contiguity spatial weights
    cursor.execute("SELECT country_code_source, country_code_target, weight_type, year, weight_value FROM spatial_weights WHERE weight_type != 'contiguity';")
    weights_cache = {}
    for src, tgt, wt_type, yr, val in cursor.fetchall():
        if wt_type not in weights_cache:
            weights_cache[wt_type] = {}
        if yr not in weights_cache[wt_type]:
            weights_cache[wt_type][yr] = {}
        if src not in weights_cache[wt_type][yr]:
            weights_cache[wt_type][yr][src] = {}
        weights_cache[wt_type][yr][src][tgt] = val
        
    # Dynamically build contiguity weights based on TUNED_PARAMS["contiguity_weights"]
    weights_cache["contiguity"] = {}
    cont_weights = TUNED_PARAMS.get("contiguity_weights", {1: 1.0, 2: 1.0, 3: 0.6, 4: 0.4, 5: 0.2})
    # Map key strings to ints
    cont_weights = {int(k): float(v) for k, v in cont_weights.items()}
    
    # Load raw contiguity from data/contdird.csv
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    contiguity_path = os.path.join(base_dir, "data", "contdird.csv")
    if os.path.exists(contiguity_path):
        df_cont = pd.read_csv(contiguity_path)
        df_cont = df_cont[(df_cont["year"] >= 1990) & (df_cont["year"] <= 2028)]
        # Forward fill to 2028 if max_year < 2028
        max_year = df_cont["year"].max()
        if max_year < 2028:
            df_latest = df_cont[df_cont["year"] == max_year]
            extra_dfs = []
            for yr in range(max_year + 1, 2029):
                df_yr = df_latest.copy()
                df_yr["year"] = yr
                extra_dfs.append(df_yr)
            df_cont = pd.concat([df_cont] + extra_dfs, ignore_index=True)
            
        # Get COW to ISO mappings
        cursor.execute("SELECT cow_code, country_code FROM countries WHERE cow_code IS NOT NULL;")
        cow_to_iso = dict(cursor.fetchall())
        cursor.execute("SELECT country_code FROM countries;")
        valid_codes = set(r[0] for r in cursor.fetchall())
        
        for _, row in df_cont.iterrows():
            yr = int(row["year"])
            src_cow = int(row["state1no"])
            tgt_cow = int(row["state2no"])
            src_iso = cow_to_iso.get(src_cow)
            tgt_iso = cow_to_iso.get(tgt_cow)
            
            if src_iso and tgt_iso and src_iso in valid_codes and tgt_iso in valid_codes:
                if src_iso == tgt_iso:
                    continue
                conttype = int(row["conttype"])
                w = cont_weights.get(conttype, 0.0)
                if w > 0.0:
                    if yr not in weights_cache["contiguity"]:
                        weights_cache["contiguity"][yr] = {}
                    if src_iso not in weights_cache["contiguity"][yr]:
                        weights_cache["contiguity"][yr][src_iso] = {}
                    weights_cache["contiguity"][yr][src_iso][tgt_iso] = w
                    
    return weights_cache

# Canonical feature list used for all modeling (base set before country OHE expansion)
BASE_FEATURES = [
    "clean_index", "polyarchy_index", "is_scheduled", "is_presidential",
    "gdp_shock", "inflation_shock", "average_news_tone",
    # NELDA pre-election indicators
    "opposition_boycott", "monitors_present",
    # Spatial Lags (V-Dem clean indices)
    "spatial_lag_clean_trade", "spatial_lag_clean_unga_voting",
    "spatial_lag_clean_alliance", "spatial_lag_clean_contiguity",
    # Spatial Lags (Election Outcomes)
    "spatial_lag_outcome_trade", "spatial_lag_outcome_unga_voting",
    "spatial_lag_outcome_alliance", "spatial_lag_outcome_contiguity",
    # Spatial Lags (GDELT protest pressure)
    "spatial_lag_protest_trade", "spatial_lag_protest_unga_voting",
    "spatial_lag_protest_alliance", "spatial_lag_protest_contiguity",
    # Spatial Lags (GDELT conflict level)
    "spatial_lag_conflict_trade", "spatial_lag_conflict_unga_voting",
    "spatial_lag_conflict_alliance", "spatial_lag_conflict_contiguity",
    # GDELT Event Indicators
    "protest_pressure", "coop_score", "conflict_level",
    "coercion_ratio", "aid_ratio",
    "protest_velocity", "conflict_velocity",
    "raw_protest_score", "raw_conflict_score", "raw_coop_score",
    "raw_coercion_score", "raw_aid_score", "total_news_volume",
    "gov_news_volume", "opp_news_volume", "gov_avg_tone", "opp_avg_tone",
    "opp_gov_volume_ratio", "opp_gov_tone_diff",
    # Tides
    "regional_challenger_tide", "global_challenger_tide", "global_economic_shock",
    "regime_challenger_tide", "regional_tide_36mo", "regional_tide_24mo",
    # Term count and historical win rate
    "incumbent_term_count", "country_incumbent_win_rate", "is_open_seat", "successor_run",
    # Prior election features (CLEA)
    "prior_turnout", "prior_margin_of_victory", "prior_effective_parties", "prior_incumbent_seat_share",
    # Institutional interactions
    "prior_margin_presidential_interaction", "party_fragmentation_legislative_interaction",
    # V-Dem shocks
    "clean_index_delta_1yr", "clean_index_delta_3yr", "polyarchy_delta_1yr", "polyarchy_delta_3yr",
    # Global anchors
    "oil_price_shock", "oil_exporter_shock", "fed_rate_change", "us_president_ideology",
    # Exchange rate volatility and shocks (Hypothesis 2)
    "currency_depreciation_6mo", "currency_depreciation_12mo", "currency_volatility_6mo",
    # Interaction terms
    "economy_regime_interaction", "protest_regime_interaction",
    "open_seat_prior_margin_interaction", "term_fatigue_economy_interaction",
    "opposition_momentum_margin_interaction",
    # Cross-Pollination Features
    "last_alt_incumbent_won", "last_alt_margin", "months_since_last_alt",
    # Legislative Improvement features
    "electoral_system_majoritarian", "electoral_system_pr", "political_polarization",
    "incumbent_seat_buffer", "is_majority_fragile",
    "pr_fragmentation_interaction", "polarization_fragmentation_interaction",
    # Expanded Structural Features: Fatigue
    "government_years_in_power", "ideology_years_in_power", "years_since_last_alternation", 
    "historical_alternation_rate", "government_continuity_score",
    # Expanded Structural Features: Institutional Stability & Volatility
    "cabinet_reshuffles_current_term", "government_type_majority", "government_type_minority", 
    "government_type_coalition", "government_type_caretaker", "government_type_technocratic", 
    "coalition_size", "coalition_fragmentation", "democratic_age", "constitutional_rigidity", 
    "electoral_volatility_10yr", "electoral_volatility_20yr", "electoral_volatility_change", "pedersen_index",
    # Expanded Structural Features: Change-based Economy & Shocks
    "inflation_acceleration", "gdp_growth_acceleration", "unemployment_acceleration", 
    "real_wage_growth", "migration_shock", "refugee_inflow_shock",
    # Expanded Structural Features: Performance Proxies (Non-Polling)
    "government_sentiment_score", "government_corruption_events", "government_corruption_ratio", 
    "legislative_productivity", "government_news_salience", "government_media_balance",
    # Expanded Structural Features: Network-Based Diffusion
    "spatial_lag_clean_proximity", "spatial_lag_outcome_proximity", 
    "spatial_lag_protest_proximity", "spatial_lag_conflict_proximity",
    "spatial_lag_democracy_delta_trade", "spatial_lag_democracy_delta_unga_voting", 
    "spatial_lag_democracy_delta_alliance", "spatial_lag_democracy_delta_contiguity", 
    "spatial_lag_democracy_delta_proximity", "spatial_lag_clean_delta_trade", 
    "spatial_lag_clean_delta_unga_voting", "spatial_lag_clean_delta_alliance", 
    "spatial_lag_clean_delta_contiguity", "spatial_lag_clean_delta_proximity",
    "global_inflation_pressure", "global_energy_pressure", "global_migration_pressure", 
    "global_democratic_backsliding", "global_incumbent_punishment_index",
    # Expanded Structural Features: Latent Global Climate Index
    "latent_global_political_climate", "latent_global_economic_anxiety", 
    "latent_global_anti_incumbent_index", "latent_global_democracy_trend"
]

# Feature classifications for model separation
EXEC_SPECIFIC = ["is_open_seat", "successor_run", "open_seat_prior_margin_interaction", "is_presidential"]
LEG_SPECIFIC = [
    "party_fragmentation_legislative_interaction", 
    "electoral_system_majoritarian", "electoral_system_pr",
    "incumbent_seat_buffer", "is_majority_fragile",
    "pr_fragmentation_interaction", "polarization_fragmentation_interaction",
    "is_presidential"
]

FEATURES = list(BASE_FEATURES)


def months_between(date_str1, date_str2):
    """
    Calculate months between date_str1 (earlier) and date_str2 (later).
    Handles YYYY-MM-DD or YYYY-MM formats.
    """
    if not date_str1 or not date_str2:
        return np.nan
    try:
        y1, m1 = int(date_str1[:4]), int(date_str1[5:7])
        y2, m2 = int(date_str2[:4]), int(date_str2[5:7])
        return (y2 - y1) * 12 + (m2 - m1)
    except Exception:
        return np.nan


def compute_cross_pollination_features(df):
    """
    Computes cross-pollination features for the entire DataFrame in place.
    """
    df["last_alt_incumbent_won"] = np.nan
    df["last_alt_margin"] = np.nan
    df["months_since_last_alt"] = np.nan
    
    # Sort by election_date to make lookup efficient
    df_sorted = df.sort_values(by="election_date")
    
    # Iterate over df to find last alt
    for idx, row in df.iterrows():
        code = row["country_code"]
        curr_date = row["election_date"]
        is_pres = row["is_presidential"]
        
        # Filter for prior alternate elections
        past_alt = df_sorted[
            (df_sorted["country_code"] == code) & 
            (df_sorted["election_date"] < curr_date) & 
            (df_sorted["is_presidential"] != is_pres)
        ]
        
        if not past_alt.empty:
            last_el = past_alt.iloc[-1]
            df.loc[idx, "last_alt_incumbent_won"] = last_el["target_outcome"]
            df.loc[idx, "last_alt_margin"] = last_el["prior_margin_of_victory"]
            df.loc[idx, "months_since_last_alt"] = months_between(last_el["election_date"], curr_date)


def get_connection():
    conn = sqlite3.connect(DB_PATH, timeout=60.0)
    conn.execute("PRAGMA foreign_keys = ON;")
    try:
        conn.execute("PRAGMA journal_mode = WAL;")
    except sqlite3.OperationalError:
        pass
    return conn

# 1. Exchange Rate Feature Calculation (NEW)
def get_exchange_rate_features(cursor, country_code, election_date_str):
    """
    Calculate currency depreciation and volatility features for an election.
    Returns:
        currency_depreciation_6mo, currency_depreciation_12mo, currency_volatility_6mo
    """
    try:
        parts = election_date_str.split('-')
        year = int(parts[0])
        month = int(parts[1])
    except (AttributeError, IndexError, ValueError):
        if isinstance(election_date_str, str) and len(election_date_str) >= 4:
            try:
                year = int(election_date_str[:4])
            except ValueError:
                return np.nan, np.nan, np.nan
        else:
            return np.nan, np.nan, np.nan
        month = 6
        
    election_serial = year * 12 + month
    
    # Query rates from year-2 to year for the country
    cursor.execute("""
        SELECT year, month, exchange_rate 
        FROM exchange_rates 
        WHERE country_code = ? AND year >= ? AND year <= ?
        ORDER BY year ASC, month ASC
    """, (country_code, year - 2, year))
    rows = cursor.fetchall()
    
    if not rows:
        return np.nan, np.nan, np.nan
        
    # Map to serial month -> rate
    rates_dict = { (r[0] * 12 + r[1]): r[2] for r in rows }
    
    # Find t0 (closest available month <= election_serial)
    available_serials = sorted([s for s in rates_dict.keys() if s <= election_serial])
    if not available_serials:
        return np.nan, np.nan, np.nan
        
    t0_serial = available_serials[-1]
    rate_t0 = rates_dict[t0_serial]
    
    # We want to find the rate 6 months ago (around t0_serial - 6)
    # Find the closest available month in the range [t0_serial - 8, t0_serial - 4]
    serials_6mo = [s for s in available_serials if t0_serial - 8 <= s <= t0_serial - 4]
    rate_6mo = np.nan
    if serials_6mo:
        closest_6 = min(serials_6mo, key=lambda s: abs(s - (t0_serial - 6)))
        rate_6mo = rates_dict[closest_6]
        
    # We want to find the rate 12 months ago (around t0_serial - 12)
    # Find the closest available month in the range [t0_serial - 14, t0_serial - 10]
    serials_12mo = [s for s in available_serials if t0_serial - 14 <= s <= t0_serial - 10]
    rate_12mo = np.nan
    if serials_12mo:
        closest_12 = min(serials_12mo, key=lambda s: abs(s - (t0_serial - 12)))
        rate_12mo = rates_dict[closest_12]
        
    # Calculate depreciation (positive = local currency depreciated / LCU per USD rose)
    depr_6mo = np.nan
    if pd.notna(rate_6mo) and rate_6mo > 0:
        depr_6mo = (rate_t0 - rate_6mo) / rate_6mo
        
    depr_12mo = np.nan
    if pd.notna(rate_12mo) and rate_12mo > 0:
        depr_12mo = (rate_t0 - rate_12mo) / rate_12mo
        
    # Volatility in the last 6 months: standard deviation of month-on-month percent changes
    # Rates in the range [t0_serial - 6, t0_serial]
    rates_last_6mo = [rates_dict[s] for s in available_serials if t0_serial - 6 <= s <= t0_serial]
    vol_6mo = np.nan
    if len(rates_last_6mo) >= 3:
        pct_changes = []
        for i in range(1, len(rates_last_6mo)):
            prev = rates_last_6mo[i-1]
            curr = rates_last_6mo[i]
            if prev > 0:
                pct_changes.append((curr - prev) / prev)
        if len(pct_changes) >= 2:
            vol_6mo = np.std(pct_changes, ddof=1)
            
    return depr_6mo, depr_12mo, vol_6mo


# 2. GDELT Decay Event Aggregation with Relative Normalization
def get_decayed_events_for_election(cursor, country_code, election_date_str, half_life_days=30):
    """
    Retrieves events for the country in the 180 days prior to the election,
    calculates decay-weighted scores, and normalizes them by total pre-election event volume.
    Also computes granular temporal GDELT coverage (fraction of active days out of 180).
    """
    cursor.execute("""
        SELECT event_date, cameo_code, goldstein_scale, avg_tone, news_volume, actor1_code, actor2_code 
        FROM gdelt_events 
        WHERE country_code = ? AND event_date BETWEEN date(?, '-180 days') AND ?;
    """, (country_code, election_date_str, election_date_str))
    
    events = cursor.fetchall()
    total_events = len(events)
    
    protest_score = 0.0
    coop_score = 0.0
    conflict_score = 0.0
    coercion_score = 0.0
    aid_score = 0.0
    
    protest_score_30d = 0.0
    protest_score_150d = 0.0
    conflict_score_30d = 0.0
    conflict_score_150d = 0.0
    
    total_volume = 0
    weighted_tone_sum = 0.0
    weighted_tone_weight = 0.0
    
    gov_news_volume = 0
    opp_news_volume = 0
    gov_tone_sum = 0.0
    gov_tone_count = 0
    opp_tone_sum = 0.0
    opp_tone_count = 0
    corruption_events = 0
    
    elect_date = datetime.strptime(election_date_str, "%Y-%m-%d")
    lambda_decay = np.log(2.0) / half_life_days
    
    date_cache = {}
    active_dates = set()
    
    for ev_date_str, cameo, goldstein, tone, volume, actor1, actor2 in events:
        active_dates.add(ev_date_str)
        ev_date = date_cache.get(ev_date_str)
        if ev_date is None:
            ev_date = datetime.strptime(ev_date_str, "%Y-%m-%d")
            date_cache[ev_date_str] = ev_date
        delta_days = (elect_date - ev_date).days
        
        # Exponential decay weight
        weight = math.exp(-lambda_decay * delta_days) * (volume / 100.0)
        
        if cameo.startswith("14"):
            protest_score += weight * abs(goldstein)
            if delta_days <= 30:
                protest_score_30d += weight * abs(goldstein)
            else:
                protest_score_150d += weight * abs(goldstein)
        elif cameo.startswith("19"):
            conflict_score += weight * abs(goldstein)
            if delta_days <= 30:
                conflict_score_30d += weight * abs(goldstein)
            else:
                conflict_score_150d += weight * abs(goldstein)
        elif cameo.startswith("03") or cameo.startswith("04"):
            coop_score += weight * goldstein
        elif cameo.startswith("17"):
            coercion_score += weight * abs(goldstein)
        elif cameo.startswith("05"):
            aid_score += weight * goldstein
            
        if cameo.startswith("112"):
            corruption_events += 1
            
        total_volume += volume
        weighted_tone_sum += tone * weight
        weighted_tone_weight += weight
        
        # Calculate relative political momentum:
        is_gov = False
        is_opp = False
        if actor1:
            if "GOV" in actor1:
                is_gov = True
            elif "OPP" in actor1 or "REB" in actor1:
                is_opp = True
        if actor2:
            if "GOV" in actor2:
                is_gov = True
            elif "OPP" in actor2 or "REB" in actor2:
                is_opp = True
                
        if is_gov:
            gov_news_volume += volume
            gov_tone_sum += tone
            gov_tone_count += 1
        if is_opp:
            opp_news_volume += volume
            opp_tone_sum += tone
            opp_tone_count += 1
            
    # Normalize by total pre-election news volume to get intensity ratios
    if total_events > 0:
        protest_ratio = protest_score / total_events
        conflict_ratio = conflict_score / total_events
        coop_ratio = coop_score / total_events
        coercion_ratio = coercion_score / total_events
        aid_ratio = aid_score / total_events
        corruption_ratio = float(corruption_events) / total_events
        
        # Velocity metric (ratio of final 30d intensity to historical preceding 150d intensity)
        # Bounded by 5-fold normalization and small smoothing coefficient to prevent div by zero
        protest_velocity = protest_score_30d / (protest_score_150d / 5.0 + 1.0)
        conflict_velocity = conflict_score_30d / (conflict_score_150d / 5.0 + 1.0)
        gdelt_coverage = min(len(active_dates) / 180.0, 1.0)
        
        gov_avg_tone = gov_tone_sum / gov_tone_count if gov_tone_count > 0 else np.nan
        opp_avg_tone = opp_tone_sum / opp_tone_count if opp_tone_count > 0 else np.nan
        opp_gov_volume_ratio = opp_news_volume / (gov_news_volume + 1.0)
        opp_gov_tone_diff = (opp_avg_tone - gov_avg_tone) if (pd.notna(opp_avg_tone) and pd.notna(gov_avg_tone)) else 0.0
    else:
        protest_ratio = np.nan
        conflict_ratio = np.nan
        coop_ratio = np.nan
        coercion_ratio = np.nan
        aid_ratio = np.nan
        corruption_ratio = np.nan
        protest_velocity = np.nan
        conflict_velocity = np.nan
        gdelt_coverage = 0.0
        
        gov_news_volume = np.nan
        opp_news_volume = np.nan
        gov_avg_tone = np.nan
        opp_avg_tone = np.nan
        opp_gov_volume_ratio = np.nan
        opp_gov_tone_diff = np.nan
        
    avg_news_tone = weighted_tone_sum / weighted_tone_weight if weighted_tone_weight > 0 else np.nan
    
    return (protest_ratio, coop_ratio, conflict_ratio, coercion_ratio, aid_ratio,
            protest_velocity, conflict_velocity,
            protest_score, coop_score, conflict_score, coercion_score, aid_score,
            avg_news_tone, total_volume if total_events > 0 else np.nan, gdelt_coverage,
            gov_news_volume, opp_news_volume, gov_avg_tone, opp_avg_tone,
            opp_gov_volume_ratio, opp_gov_tone_diff, corruption_events, corruption_ratio)


def compute_data_completeness(row, features):
    """Compute weighted fraction of non-null features for a single row, incorporating granular GDELT coverage and tuned category multipliers."""
    # Critical feature weights (tuned category multipliers applied: vdem=1.0, prior=2.0, gdelt=0.5, macro=1.0, spatial=0.5)
    core_weights = {
        "clean_index": 2.0,
        "polyarchy_index": 2.0,
        "prior_incumbent_seat_share": 4.0,
        "prior_margin_of_victory": 3.0,
        "country_incumbent_win_rate": 3.0,
        "incumbent_term_count": 2.0,
        "average_news_tone": 0.5,
        "total_news_volume": 0.5,
        "protest_pressure": 0.5,
        "coop_score": 0.5,
        "conflict_level": 0.5,
        "coercion_ratio": 0.5,
        "aid_ratio": 0.5,
        "protest_velocity": 0.5,
        "conflict_velocity": 0.5,
        "gov_news_volume": 0.5,
        "opp_news_volume": 0.5,
        "gov_avg_tone": 0.5,
        "opp_avg_tone": 0.5,
        "opp_gov_volume_ratio": 0.5,
        "opp_gov_tone_diff": 0.5,
        "clean_index_delta_1yr": 1.5,
        "clean_index_delta_3yr": 1.5,
        "polyarchy_delta_1yr": 1.5,
        "polyarchy_delta_3yr": 1.5,
        "prior_margin_presidential_interaction": 3.0,
        "party_fragmentation_legislative_interaction": 3.0,
        "oil_price_shock": 1.0,
        "oil_exporter_shock": 1.0,
        "fed_rate_change": 1.0,
        "us_president_ideology": 1.0,
        "currency_depreciation_6mo": 1.0,
        "currency_depreciation_12mo": 1.0,
        "currency_volatility_6mo": 1.0,
        "economy_regime_interaction": 1.0,
        "protest_regime_interaction": 1.0,
        "open_seat_prior_margin_interaction": 1.0,
        "term_fatigue_economy_interaction": 1.0,
        "opposition_momentum_margin_interaction": 1.0
    }
    
    gdelt_features = {
        "average_news_tone", "total_news_volume", "protest_pressure", "coop_score", 
        "conflict_level", "coercion_ratio", "aid_ratio", "protest_velocity", "conflict_velocity",
        "gov_news_volume", "opp_news_volume", "gov_avg_tone", "opp_avg_tone",
        "opp_gov_volume_ratio", "opp_gov_tone_diff"
    }
    
    gdelt_coverage = row.get("gdelt_coverage", 0.0)
    if pd.isna(gdelt_coverage):
        gdelt_coverage = 0.0
        
    total_weight = 0.0
    non_null_weight = 0.0
    
    for f in features:
        w = core_weights.get(f, 0.05)  # Tuned default weight for spatial lags / minor features (0.1 * 0.5 = 0.05)
        total_weight += w
        if pd.notna(row.get(f)):
            if f in gdelt_features:
                non_null_weight += w * gdelt_coverage
            else:
                non_null_weight += w
            
    return non_null_weight / total_weight if total_weight > 0 else 0.0


def compute_data_source_flags(row):
    """Compute bitmask of which data sources contributed: V-Dem=1, Macro=2, GDELT=4, Networks=8."""
    flags = 0
    if pd.notna(row.get("clean_index")) and pd.notna(row.get("polyarchy_index")):
        flags |= 1  # V-Dem
    if pd.notna(row.get("gdp_shock")) and pd.notna(row.get("inflation_shock")):
        flags |= 2  # Macro
    if pd.notna(row.get("protest_pressure")) and pd.notna(row.get("average_news_tone")):
        flags |= 4  # GDELT
    if pd.notna(row.get("spatial_lag_clean_trade")):
        flags |= 8  # Networks
    return flags


# 2. Build Dataset with Advanced Feature Engineering
def build_modeling_dataset(conn, include_upcoming=False):
    """
    Build the full feature matrix from the database.
    
    Key change from prior version: elections with missing V-Dem or macro data
    are NO LONGER skipped — they get NaN values, which XGBoost handles natively.
    """
    cursor = conn.cursor()
    
    # Get all NELDA election records
    cursor.execute("""
        SELECT e.election_id, e.country_code, e.election_year, e.election_date, 
               e.election_type, e.is_scheduled, e.is_fraudulent, e.incumbent_ran, e.has_successor,
               n.nelda11_fraud_allegations, n.nelda15_opposition_boycott, 
               n.nelda23_incumbent_won, n.nelda51_monitors_present
        FROM elections e
        JOIN nelda_indicators n ON e.election_id = n.election_id;
    """)
    elections = cursor.fetchall()
    
    # Optionally include upcoming elections
    upcoming_elections = []
    if include_upcoming:
        cursor.execute("""
            SELECT upcoming_id, country_code, election_year, election_date,
                   election_type, is_scheduled, actual_outcome, incumbent_ran, has_successor
            FROM upcoming_elections;
        """)
        upcoming_elections = cursor.fetchall()
       # Pre-cache V-Dem and Macro indicators by (country, year) for speed
    cursor.execute("""
        SELECT country_code, year, clean_elections_index, polyarchy_index, regime_type,
               electoral_system_majoritarian, electoral_system_pr, political_polarization,
               v2cafres, v2casurv, v2lgbicam, v2exhoshog, v2xlg_legcon, v2x_jucon, v2x_execorr, v2x_pubcorr
        FROM vdem_indicators;
    """)
    vdem_list = cursor.fetchall()
    vdem_cache = {}
    vdem_years_by_country = {}
    
    country_maj_fallback = {}
    country_pr_fallback = {}
    country_polar_fallback = {}
    democracy_start_year = {}
    
    for code, yr, clean, poly, regime, maj, pr, polar, v2cafres, v2casurv, v2lgbicam, v2exhoshog, v2xlg_legcon, v2x_jucon, v2x_execorr, v2x_pubcorr in vdem_list:
        vdem_cache[(code, yr)] = (clean, poly, regime, maj, pr, polar, v2cafres, v2casurv, v2lgbicam, v2exhoshog, v2xlg_legcon, v2x_jucon, v2x_execorr, v2x_pubcorr)
        if code not in vdem_years_by_country:
            vdem_years_by_country[code] = []
        vdem_years_by_country[code].append(yr)
        
        # Populate country defaults (taking latest available year's value)
        if pd.notna(maj):
            country_maj_fallback[code] = maj
        if pd.notna(pr):
            country_pr_fallback[code] = pr
        if pd.notna(polar):
            country_polar_fallback[code] = polar
            
        # Pre-calculate democracy start year (first year with regime_type >= 2)
        if regime is not None and regime >= 2:
            if code not in democracy_start_year or yr < democracy_start_year[code]:
                democracy_start_year[code] = yr
                
    for code in vdem_years_by_country:
        vdem_years_by_country[code].sort()
    
    cursor.execute("""
        SELECT country_code, year, gdp_growth, inflation, gdp_growth_deviation, inflation_deviation,
               unemployment, gdp_per_capita_growth, net_migration, refugee_asylum, refugee_origin
        FROM macro_indicators;
    """)
    macro_cache = {(r[0], r[1]): (r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9], r[10]) for r in cursor.fetchall()}

    # Pre-cache CLEA indicators for prior election linking
    cursor.execute("""
        SELECT country_code, year, turnout, margin_of_victory, effective_parties, incumbent_seat_share, pedersen_index
        FROM clea_indicators;
    """)
    clea_list = cursor.fetchall()
    clea_cache = {}
    clea_years_by_country = {}
    for code, yr, to, mov, ep, iss, ped in clea_list:
        clea_cache[(code, yr)] = (to, mov, ep, iss, ped)
        if code not in clea_years_by_country:
            clea_years_by_country[code] = []
        clea_years_by_country[code].append(yr)
    for code in clea_years_by_country:
        clea_years_by_country[code].sort()
        
    # Pre-cache global indicators
    cursor.execute("SELECT year, brent_oil_price, fed_funds_rate, us_president_ideology FROM global_indicators;")
    global_cache = {r[0]: (r[1], r[2], r[3]) for r in cursor.fetchall()}
    
    # Pre-cache net oil exporter status of countries
    cursor.execute("SELECT country_code, is_oil_exporter FROM countries;")
    oil_exporter_cache = dict(cursor.fetchall())
        
    cursor.execute("SELECT country_code, region FROM countries;")
    region_cache = dict(cursor.fetchall())
    
    # Pre-cache all spatial weights for faster lookups
    weights_cache = load_weights_cache(cursor)
    
    # Pre-compute fatigue features chronologically using all elections
    # We load outcome indicators to compute consecutive years in power
    cursor.execute("""
        SELECT e.election_id, e.country_code, e.election_year, n.nelda23_incumbent_won
        FROM elections e
        JOIN nelda_indicators n ON e.election_id = n.election_id;
    """)
    hist_elects = cursor.fetchall()
    
    cursor.execute("""
        SELECT upcoming_id, country_code, election_year, actual_outcome
        FROM upcoming_elections;
    """)
    up_elects = cursor.fetchall()
    
    all_elects_list = []
    for eid, code, yr, n23 in hist_elects:
        outcome = 'incumbent' if n23 == 'Yes' else ('challenger' if n23 == 'No' else 'unknown')
        all_elects_list.append({
            "id": eid,
            "code": code,
            "year": yr,
            "outcome": outcome
        })
    for uid, code, yr, actual in up_elects:
        outcome = actual if actual in ['incumbent', 'challenger'] else 'unknown'
        all_elects_list.append({
            "id": f"upcoming_{uid}",
            "code": code,
            "year": yr,
            "outcome": outcome
        })
        
    by_country = {}
    for el in all_elects_list:
        if el["code"] not in by_country:
            by_country[el["code"]] = []
        by_country[el["code"]].append(el)
        
    fatigue_cache = {}
    for code, el_list in by_country.items():
        el_list = sorted(el_list, key=lambda x: x["year"])
        last_alternation_year = None
        challenger_win_count = 0
        total_elections_count = 0
        
        for el in el_list:
            if last_alternation_year is None:
                last_alternation_year = el["year"]
            gov_years = el["year"] - last_alternation_year
            years_since_alt = el["year"] - last_alternation_year
            alt_rate = float(challenger_win_count) / total_elections_count if total_elections_count > 0 else 0.0
            
            fatigue_cache[el["id"]] = {
                "government_years_in_power": gov_years,
                "ideology_years_in_power": gov_years,
                "years_since_last_alternation": years_since_alt,
                "historical_alternation_rate": alt_rate
            }
            
            if el["outcome"] == 'challenger':
                last_alternation_year = el["year"]
                challenger_win_count += 1
            total_elections_count += 1
        
    total_nelda = len(elections)
    total_upcoming = len(upcoming_elections)
    print(f"Starting modeling dataset assembly: {total_nelda} historical records and {total_upcoming} upcoming records.")
    
    data = []
    
    # Process NELDA elections
    for idx, row in enumerate(elections):
        if (idx + 1) % 500 == 0 or idx == 0 or idx == total_nelda - 1:
            print(f"  [Progress] Processed {idx + 1}/{total_nelda} historical records...")
        elect_id, code, year, date_str, el_type, scheduled, is_fraud, incumbent_ran, has_successor, n11, n15, n23, n51 = row
        
        # Fetch V-Dem scores — NaN if missing, forward-filled if missing for later years
        vdem_res = vdem_cache.get((code, year))
        if not vdem_res and code in vdem_years_by_country:
            avail_years = [y for y in vdem_years_by_country[code] if y <= year]
            if avail_years:
                vdem_res = vdem_cache.get((code, avail_years[-1]))
                
        if vdem_res:
            (clean_idx, poly_idx, regime, maj_sys, pr_sys, polarization, 
             v2cafres, v2casurv, v2lgbicam, v2exhoshog, v2xlg_legcon, v2x_jucon, v2x_execorr, v2x_pubcorr) = vdem_res
        else:
            (clean_idx, poly_idx, regime, maj_sys, pr_sys, polarization, 
             v2cafres, v2casurv, v2lgbicam, v2exhoshog, v2xlg_legcon, v2x_jucon, v2x_execorr, v2x_pubcorr) = (
                 np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
             )
            
        # Apply structural defaults for country-year cells that lack them
        if pd.isna(maj_sys):
            maj_sys = country_maj_fallback.get(code, np.nan)
        if pd.isna(pr_sys):
            pr_sys = country_pr_fallback.get(code, np.nan)
        if pd.isna(polarization):
            polarization = country_polar_fallback.get(code, np.nan)
            
        # Calculate V-Dem shocks (1-year and 3-year deltas)
        clean_idx_prev_1yr, poly_idx_prev_1yr = np.nan, np.nan
        vdem_res_prev_1yr = vdem_cache.get((code, year - 1))
        if not vdem_res_prev_1yr and code in vdem_years_by_country:
            avail_years = [y for y in vdem_years_by_country[code] if y <= year - 1]
            if avail_years:
                vdem_res_prev_1yr = vdem_cache.get((code, avail_years[-1]))
        if vdem_res_prev_1yr:
            clean_idx_prev_1yr, poly_idx_prev_1yr, _, _, _, _, _, _, _, _, _, _, _, _ = vdem_res_prev_1yr
            
        clean_idx_prev_3yr, poly_idx_prev_3yr = np.nan, np.nan
        vdem_res_prev_3yr = vdem_cache.get((code, year - 3))
        if not vdem_res_prev_3yr and code in vdem_years_by_country:
            avail_years = [y for y in vdem_years_by_country[code] if y <= year - 3]
            if avail_years:
                vdem_res_prev_3yr = vdem_cache.get((code, avail_years[-1]))
        if vdem_res_prev_3yr:
            clean_idx_prev_3yr, poly_idx_prev_3yr, _, _, _, _, _, _, _, _, _, _, _, _ = vdem_res_prev_3yr
            
        clean_delta_1yr = clean_idx - clean_idx_prev_1yr if (pd.notna(clean_idx) and pd.notna(clean_idx_prev_1yr)) else 0.0
        clean_delta_3yr = clean_idx - clean_idx_prev_3yr if (pd.notna(clean_idx) and pd.notna(clean_idx_prev_3yr)) else 0.0
        poly_delta_1yr = poly_idx - poly_idx_prev_1yr if (pd.notna(poly_idx) and pd.notna(poly_idx_prev_1yr)) else 0.0
        poly_delta_3yr = poly_idx - poly_idx_prev_3yr if (pd.notna(poly_idx) and pd.notna(poly_idx_prev_3yr)) else 0.0
        
        # Fetch Macro Economic Shock scores — NaN if missing (no synthetic fallback)
        macro_res = macro_cache.get((code, year))
        if macro_res:
            gdp, inf, gdp_dev, inf_dev, unemployment, gdp_per_capita_growth, net_migration, refugee_asylum, refugee_origin = macro_res
        else:
            gdp, inf, gdp_dev, inf_dev, unemployment, gdp_per_capita_growth, net_migration, refugee_asylum, refugee_origin = (
                np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
            )
            
        # Calculate economic accelerations
        macro_prev = macro_cache.get((code, year - 1))
        if macro_prev:
            gdp_prev, inf_prev, _, _, unemp_prev, _, net_mig_prev, ref_asyl_prev, _ = macro_prev
        else:
            gdp_prev, inf_prev, unemp_prev, net_mig_prev, ref_asyl_prev = np.nan, np.nan, np.nan, np.nan, np.nan
            
        inflation_acceleration = inf - inf_prev if (pd.notna(inf) and pd.notna(inf_prev)) else 0.0
        gdp_growth_acceleration = gdp - gdp_prev if (pd.notna(gdp) and pd.notna(gdp_prev)) else 0.0
        unemployment_acceleration = unemployment - unemp_prev if (pd.notna(unemployment) and pd.notna(unemp_prev)) else 0.0
        migration_shock = net_migration - net_mig_prev if (pd.notna(net_migration) and pd.notna(net_mig_prev)) else 0.0
        refugee_inflow_shock = refugee_asylum - ref_asyl_prev if (pd.notna(refugee_asylum) and pd.notna(ref_asyl_prev)) else 0.0
        
        # Fetch GDELT normalized decay features (returns NaN if no events)
        (protest, coop, conflict, coercion, aid,
         protest_vel, conflict_vel,
         raw_protest, raw_coop, raw_conflict, raw_coercion, raw_aid,
         avg_tone, vol, gdelt_coverage,
         gov_news_volume, opp_news_volume, gov_avg_tone, opp_avg_tone,
         opp_gov_volume_ratio, opp_gov_tone_diff, corruption_events, corruption_ratio) = get_decayed_events_for_election(cursor, code, date_str)
         
        (currency_depreciation_6mo, currency_depreciation_12mo,
         currency_volatility_6mo) = get_exchange_rate_features(cursor, code, date_str)
         

        # Fetch prior CLEA election statistics
        prior_to, prior_mov, prior_ep, prior_iss, prior_ped = np.nan, np.nan, np.nan, np.nan, np.nan
        if code in clea_years_by_country:
            past_years = [y for y in clea_years_by_country[code] if y < year]
            if past_years:
                prior_yr = past_years[-1]
                prior_to, prior_mov, prior_ep, prior_iss, prior_ped = clea_cache[(code, prior_yr)]
                
        # Calculate historical volatility features
        past_volatilities = []
        if code in clea_years_by_country:
            for past_yr in clea_years_by_country[code]:
                if past_yr < year:
                    ped_val = clea_cache[(code, past_yr)][4]
                    if pd.notna(ped_val):
                        past_volatilities.append((past_yr, ped_val))
                        
        vol_10yr = [v for y, v in past_volatilities if year - y <= 10]
        vol_20yr = [v for y, v in past_volatilities if year - y <= 20]
        electoral_volatility_10yr = np.mean(vol_10yr) if vol_10yr else np.nan
        electoral_volatility_20yr = np.mean(vol_20yr) if vol_20yr else np.nan
        electoral_volatility_change = prior_ped - electoral_volatility_10yr if (pd.notna(prior_ped) and pd.notna(electoral_volatility_10yr)) else 0.0
                
        # Fetch global indicators
        glob_res = global_cache.get(year)
        if not glob_res:
            avail_years = [y for y in global_cache.keys() if y <= year]
            if avail_years:
                glob_res = global_cache.get(max(avail_years))
        if glob_res:
            oil_price, fed_rate, us_pres = glob_res
        else:
            oil_price, fed_rate, us_pres = np.nan, np.nan, np.nan
            
        glob_res_prev = global_cache.get(year - 1)
        if not glob_res_prev and glob_res:
            glob_res_prev = glob_res
        if glob_res_prev:
            oil_price_prev, fed_rate_prev, _ = glob_res_prev
        else:
            oil_price_prev, fed_rate_prev = np.nan, np.nan
            
        if pd.notna(oil_price) and pd.notna(oil_price_prev) and oil_price_prev > 0:
            oil_price_shock = (oil_price - oil_price_prev) / oil_price_prev
        else:
            oil_price_shock = 0.0
            
        is_oil_exp = oil_exporter_cache.get(code, 0)
        oil_exporter_shock = oil_price_shock * is_oil_exp
        
        if pd.notna(fed_rate) and pd.notna(fed_rate_prev):
            fed_rate_change = fed_rate - fed_rate_prev
        else:
            fed_rate_change = 0.0
            
        is_presidential = 1 if el_type in ["Presidential", "Executive"] else 0
        prior_margin_presidential_interaction = prior_mov * is_presidential if (pd.notna(prior_mov)) else 0.0
        party_fragmentation_legislative_interaction = prior_ep * (1 - is_presidential) if (pd.notna(prior_ep)) else 0.0
        
        # New legislative improvement features
        incumbent_seat_buffer = prior_iss - 0.50 if pd.notna(prior_iss) else 0.0
        is_majority_fragile = 1.0 if (pd.notna(prior_iss) and 0.50 <= prior_iss < 0.55) else 0.0
        pr_fragmentation_interaction = pr_sys * prior_ep if (pd.notna(pr_sys) and pd.notna(prior_ep)) else 0.0
        polarization_fragmentation_interaction = polarization * prior_ep if (pd.notna(polarization) and pd.notna(prior_ep)) else 0.0
        
        # Convert opposition boycott and monitors present to numeric indicators
        opposition_boycott = 1.0 if n15 == "Yes" else (0.0 if n15 == "No" else np.nan)
        monitors_present = 1.0 if n51 == "Yes" else (0.0 if n51 == "No" else np.nan)

        # Fatigue calculations
        fatigue = fatigue_cache.get(elect_id, {})
        government_years_in_power = fatigue.get("government_years_in_power", np.nan)
        ideology_years_in_power = fatigue.get("ideology_years_in_power", np.nan)
        years_since_last_alternation = fatigue.get("years_since_last_alternation", np.nan)
        historical_alternation_rate = fatigue.get("historical_alternation_rate", np.nan)
        government_continuity_score = government_years_in_power * v2xlg_legcon if (pd.notna(government_years_in_power) and pd.notna(v2xlg_legcon)) else np.nan
        
        # Institutional calculations
        government_type_majority = 1.0 if (pd.notna(prior_iss) and prior_iss >= 0.50) else (0.0 if pd.notna(prior_iss) else np.nan)
        government_type_minority = 1.0 if (pd.notna(prior_iss) and prior_iss < 0.50) else (0.0 if pd.notna(prior_iss) else np.nan)
        government_type_coalition = 1.0 if (pd.notna(prior_ep) and prior_ep > 2.5) else (0.0 if pd.notna(prior_ep) else np.nan)
        government_type_caretaker = 0.0
        government_type_technocratic = 0.0
        coalition_size = prior_ep
        coalition_fragmentation = polarization
        democratic_age = year - democracy_start_year[code] if (code in democracy_start_year and year >= democracy_start_year[code]) else 0
        constitutional_rigidity = (v2xlg_legcon + v2x_jucon) / 2.0 if (pd.notna(v2xlg_legcon) and pd.notna(v2x_jucon)) else np.nan
        
        # Media & Sentiment calculations
        government_sentiment_score = gov_avg_tone
        government_news_salience = (gov_news_volume + opp_news_volume) / (vol + 1.0) if pd.notna(vol) else np.nan
        government_media_balance = opp_gov_tone_diff
        legislative_productivity = v2xlg_legcon # legislative constraints proxy
        
        # Targets:
        # Target A: Incumbent Won (1 = Yes, 0 = No)
        incumbent_won = 1 if n23 == "Yes" else 0
        
        # Target B: Integrity level
        if is_fraud == 1:
            integrity_level = 2
        elif pd.notna(clean_idx) and clean_idx < 0.65:
            integrity_level = 1
        else:
            integrity_level = 0
            
        data.append({
            "election_id": elect_id,
            "source": "nelda",
            "country_code": code,
            "region": region_cache.get(code, "Unknown"),
            "year": year,
            "election_date": date_str,
            "is_scheduled": scheduled,
            "is_presidential": is_presidential,
            "opposition_boycott": opposition_boycott,
            "monitors_present": monitors_present,
            "clean_index": clean_idx,
            "polyarchy_index": poly_idx,
            "regime_type": regime,
            "protest_pressure": protest,
            "coop_score": coop,
            "conflict_level": conflict,
            "coercion_ratio": coercion,
            "aid_ratio": aid,
            "protest_velocity": protest_vel,
            "conflict_velocity": conflict_vel,
            "raw_protest_score": raw_protest,
            "raw_conflict_score": raw_conflict,
            "raw_coop_score": raw_coop,
            "raw_coercion_score": raw_coercion,
            "raw_aid_score": raw_aid,
            "total_news_volume": vol,
            "average_news_tone": avg_tone,
            "gov_news_volume": gov_news_volume,
            "opp_news_volume": opp_news_volume,
            "gov_avg_tone": gov_avg_tone,
            "opp_avg_tone": opp_avg_tone,
            "opp_gov_volume_ratio": opp_gov_volume_ratio,
            "opp_gov_tone_diff": opp_gov_tone_diff,
            "gdp_growth": gdp,
            "inflation": inf,
            "gdp_shock": gdp_dev,
            "inflation_shock": inf_dev,
            "prior_turnout": prior_to,
            "prior_margin_of_victory": prior_mov,
            "prior_effective_parties": prior_ep,
            "prior_incumbent_seat_share": prior_iss,
            "prior_margin_presidential_interaction": prior_margin_presidential_interaction,
            "party_fragmentation_legislative_interaction": party_fragmentation_legislative_interaction,
            "clean_index_delta_1yr": clean_delta_1yr,
            "clean_index_delta_3yr": clean_delta_3yr,
            "polyarchy_delta_1yr": poly_delta_1yr,
            "polyarchy_delta_3yr": poly_delta_3yr,
            "oil_price_shock": oil_price_shock,
            "oil_exporter_shock": oil_exporter_shock,
            "fed_rate_change": fed_rate_change,
            "us_president_ideology": us_pres,
            "incumbent_ran": incumbent_ran,
            "has_successor": has_successor,
            "currency_depreciation_6mo": currency_depreciation_6mo,
            "currency_depreciation_12mo": currency_depreciation_12mo,
            "currency_volatility_6mo": currency_volatility_6mo,
            "electoral_system_majoritarian": maj_sys,
            "electoral_system_pr": pr_sys,
            "political_polarization": polarization,
            "incumbent_seat_buffer": incumbent_seat_buffer,
            "is_majority_fragile": is_majority_fragile,
            "pr_fragmentation_interaction": pr_fragmentation_interaction,
            "polarization_fragmentation_interaction": polarization_fragmentation_interaction,
            # Expanded Structural Features
            "government_years_in_power": government_years_in_power,
            "ideology_years_in_power": ideology_years_in_power,
            "years_since_last_alternation": years_since_last_alternation,
            "historical_alternation_rate": historical_alternation_rate,
            "government_continuity_score": government_continuity_score,
            "cabinet_reshuffles_current_term": v2cafres,
            "government_type_majority": government_type_majority,
            "government_type_minority": government_type_minority,
            "government_type_coalition": government_type_coalition,
            "government_type_caretaker": government_type_caretaker,
            "government_type_technocratic": government_type_technocratic,
            "coalition_size": coalition_size,
            "coalition_fragmentation": coalition_fragmentation,
            "democratic_age": democratic_age,
            "constitutional_rigidity": constitutional_rigidity,
            "pedersen_index": prior_ped,
            "electoral_volatility_10yr": electoral_volatility_10yr,
            "electoral_volatility_20yr": electoral_volatility_20yr,
            "electoral_volatility_change": electoral_volatility_change,
            "inflation_acceleration": inflation_acceleration,
            "gdp_growth_acceleration": gdp_growth_acceleration,
            "unemployment_acceleration": unemployment_acceleration,
            "real_wage_growth": gdp_per_capita_growth,
            "migration_shock": migration_shock,
            "refugee_inflow_shock": refugee_inflow_shock,
            "government_sentiment_score": government_sentiment_score,
            "government_corruption_events": corruption_events,
            "government_corruption_ratio": corruption_ratio,
            "legislative_productivity": legislative_productivity,
            "government_news_salience": government_news_salience,
            "government_media_balance": government_media_balance,
            "target_outcome": incumbent_won,
            "target_integrity": integrity_level,
            "gdelt_coverage": gdelt_coverage
        })
    
    # Process upcoming elections
    for idx, row in enumerate(upcoming_elections):
        if (idx + 1) % 100 == 0 or idx == 0 or idx == total_upcoming - 1:
            print(f"  [Progress] Processed {idx + 1}/{total_upcoming} upcoming records...")
        up_id, code, year, date_str, el_type, scheduled, actual_outcome, incumbent_ran, has_successor = row
        
        # Fetch V-Dem scores — NaN if missing, forward-filled if missing for later years
        vdem_res = vdem_cache.get((code, year))
        if not vdem_res and code in vdem_years_by_country:
            avail_years = [y for y in vdem_years_by_country[code] if y <= year]
            if avail_years:
                vdem_res = vdem_cache.get((code, avail_years[-1]))
                
        if vdem_res:
            (clean_idx, poly_idx, regime, maj_sys, pr_sys, polarization, 
             v2cafres, v2casurv, v2lgbicam, v2exhoshog, v2xlg_legcon, v2x_jucon, v2x_execorr, v2x_pubcorr) = vdem_res
        else:
            (clean_idx, poly_idx, regime, maj_sys, pr_sys, polarization, 
             v2cafres, v2casurv, v2lgbicam, v2exhoshog, v2xlg_legcon, v2x_jucon, v2x_execorr, v2x_pubcorr) = (
                 np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
             )
            
        # Apply structural defaults for country-year cells that lack them
        if pd.isna(maj_sys):
            maj_sys = country_maj_fallback.get(code, np.nan)
        if pd.isna(pr_sys):
            pr_sys = country_pr_fallback.get(code, np.nan)
        if pd.isna(polarization):
            polarization = country_polar_fallback.get(code, np.nan)
            
        # Calculate V-Dem shocks (1-year and 3-year deltas)
        clean_idx_prev_1yr, poly_idx_prev_1yr = np.nan, np.nan
        vdem_res_prev_1yr = vdem_cache.get((code, year - 1))
        if not vdem_res_prev_1yr and code in vdem_years_by_country:
            avail_years = [y for y in vdem_years_by_country[code] if y <= year - 1]
            if avail_years:
                vdem_res_prev_1yr = vdem_cache.get((code, avail_years[-1]))
        if vdem_res_prev_1yr:
            clean_idx_prev_1yr, poly_idx_prev_1yr, _, _, _, _, _, _, _, _, _, _, _, _ = vdem_res_prev_1yr
            
        clean_idx_prev_3yr, poly_idx_prev_3yr = np.nan, np.nan
        vdem_res_prev_3yr = vdem_cache.get((code, year - 3))
        if not vdem_res_prev_3yr and code in vdem_years_by_country:
            avail_years = [y for y in vdem_years_by_country[code] if y <= year - 3]
            if avail_years:
                vdem_res_prev_3yr = vdem_cache.get((code, avail_years[-1]))
        if vdem_res_prev_3yr:
            clean_idx_prev_3yr, poly_idx_prev_3yr, _, _, _, _, _, _, _, _, _, _, _, _ = vdem_res_prev_3yr
            
        clean_delta_1yr = clean_idx - clean_idx_prev_1yr if (pd.notna(clean_idx) and pd.notna(clean_idx_prev_1yr)) else 0.0
        clean_delta_3yr = clean_idx - clean_idx_prev_3yr if (pd.notna(clean_idx) and pd.notna(clean_idx_prev_3yr)) else 0.0
        poly_delta_1yr = poly_idx - poly_idx_prev_1yr if (pd.notna(poly_idx) and pd.notna(poly_idx_prev_1yr)) else 0.0
        poly_delta_3yr = poly_idx - poly_idx_prev_3yr if (pd.notna(poly_idx) and pd.notna(poly_idx_prev_3yr)) else 0.0
        
        # Fetch macro indicators
        macro_res = macro_cache.get((code, year))
        if macro_res:
            gdp, inf, gdp_dev, inf_dev, unemployment, gdp_per_capita_growth, net_migration, refugee_asylum, refugee_origin = macro_res
        else:
            gdp, inf, gdp_dev, inf_dev, unemployment, gdp_per_capita_growth, net_migration, refugee_asylum, refugee_origin = (
                np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
            )
            
        # Calculate economic accelerations
        macro_prev = macro_cache.get((code, year - 1))
        if macro_prev:
            gdp_prev, inf_prev, _, _, unemp_prev, _, net_mig_prev, ref_asyl_prev, _ = macro_prev
        else:
            gdp_prev, inf_prev, unemp_prev, net_mig_prev, ref_asyl_prev = np.nan, np.nan, np.nan, np.nan, np.nan
            
        inflation_acceleration = inf - inf_prev if (pd.notna(inf) and pd.notna(inf_prev)) else 0.0
        gdp_growth_acceleration = gdp - gdp_prev if (pd.notna(gdp) and pd.notna(gdp_prev)) else 0.0
        unemployment_acceleration = unemployment - unemp_prev if (pd.notna(unemployment) and pd.notna(unemp_prev)) else 0.0
        migration_shock = net_migration - net_mig_prev if (pd.notna(net_migration) and pd.notna(net_mig_prev)) else 0.0
        refugee_inflow_shock = refugee_asylum - ref_asyl_prev if (pd.notna(refugee_asylum) and pd.notna(ref_asyl_prev)) else 0.0
        
        # GDELT — try to fetch if we have a date
        if date_str:
            (protest, coop, conflict, coercion, aid,
             protest_vel, conflict_vel,
             raw_protest, raw_coop, raw_conflict, raw_coercion, raw_aid,
             avg_tone, vol, gdelt_coverage,
             gov_news_volume, opp_news_volume, gov_avg_tone, opp_avg_tone,
             opp_gov_volume_ratio, opp_gov_tone_diff, corruption_events, corruption_ratio) = get_decayed_events_for_election(cursor, code, date_str)
            (currency_depreciation_6mo, currency_depreciation_12mo,
             currency_volatility_6mo) = get_exchange_rate_features(cursor, code, date_str)
        else:
            protest, coop, conflict, coercion, aid = np.nan, np.nan, np.nan, np.nan, np.nan
            protest_vel, conflict_vel = np.nan, np.nan
            raw_protest, raw_coop, raw_conflict, raw_coercion, raw_aid = np.nan, np.nan, np.nan, np.nan, np.nan
            avg_tone, vol = np.nan, np.nan
            gdelt_coverage = 0.0
            gov_news_volume, opp_news_volume = np.nan, np.nan
            gov_avg_tone, opp_avg_tone = np.nan, np.nan
            opp_gov_volume_ratio, opp_gov_tone_diff = np.nan, np.nan
            corruption_events, corruption_ratio = np.nan, np.nan
            currency_depreciation_6mo, currency_depreciation_12mo, currency_volatility_6mo = np.nan, np.nan, np.nan
            
        # Fetch prior CLEA election statistics
        prior_to, prior_mov, prior_ep, prior_iss, prior_ped = np.nan, np.nan, np.nan, np.nan, np.nan
        if code in clea_years_by_country:
            past_years = [y for y in clea_years_by_country[code] if y < year]
            if past_years:
                prior_yr = past_years[-1]
                prior_to, prior_mov, prior_ep, prior_iss, prior_ped = clea_cache[(code, prior_yr)]
                
        # Calculate historical volatility features
        past_volatilities = []
        if code in clea_years_by_country:
            for past_yr in clea_years_by_country[code]:
                if past_yr < year:
                    ped_val = clea_cache[(code, past_yr)][4]
                    if pd.notna(ped_val):
                        past_volatilities.append((past_yr, ped_val))
                        
        vol_10yr = [v for y, v in past_volatilities if year - y <= 10]
        vol_20yr = [v for y, v in past_volatilities if year - y <= 20]
        electoral_volatility_10yr = np.mean(vol_10yr) if vol_10yr else np.nan
        electoral_volatility_20yr = np.mean(vol_20yr) if vol_20yr else np.nan
        electoral_volatility_change = prior_ped - electoral_volatility_10yr if (pd.notna(prior_ped) and pd.notna(electoral_volatility_10yr)) else 0.0
                
        # Fetch global indicators
        glob_res = global_cache.get(year)
        if not glob_res:
            avail_years = [y for y in global_cache.keys() if y <= year]
            if avail_years:
                glob_res = global_cache.get(max(avail_years))
        if glob_res:
            oil_price, fed_rate, us_pres = glob_res
        else:
            oil_price, fed_rate, us_pres = np.nan, np.nan, np.nan
            
        glob_res_prev = global_cache.get(year - 1)
        if not glob_res_prev and glob_res:
            glob_res_prev = glob_res
        if glob_res_prev:
            oil_price_prev, fed_rate_prev, _ = glob_res_prev
        else:
            oil_price_prev, fed_rate_prev = np.nan, np.nan
            
        if pd.notna(oil_price) and pd.notna(oil_price_prev) and oil_price_prev > 0:
            oil_price_shock = (oil_price - oil_price_prev) / oil_price_prev
        else:
            oil_price_shock = 0.0
            
        is_oil_exp = oil_exporter_cache.get(code, 0)
        oil_exporter_shock = oil_price_shock * is_oil_exp
        
        if pd.notna(fed_rate) and pd.notna(fed_rate_prev):
            fed_rate_change = fed_rate - fed_rate_prev
        else:
            fed_rate_change = 0.0
            
        is_presidential = 1 if el_type in ["Presidential", "Executive"] else 0
        prior_margin_presidential_interaction = prior_mov * is_presidential if (pd.notna(prior_mov)) else 0.0
        party_fragmentation_legislative_interaction = prior_ep * (1 - is_presidential) if (pd.notna(prior_ep)) else 0.0
        
        # New legislative improvement features
        incumbent_seat_buffer = prior_iss - 0.50 if pd.notna(prior_iss) else 0.0
        is_majority_fragile = 1.0 if (pd.notna(prior_iss) and 0.50 <= prior_iss < 0.55) else 0.0
        pr_fragmentation_interaction = pr_sys * prior_ep if (pd.notna(pr_sys) and pd.notna(prior_ep)) else 0.0
        polarization_fragmentation_interaction = polarization * prior_ep if (pd.notna(polarization) and pd.notna(prior_ep)) else 0.0
        
        # Target: use actual outcome if known, NaN if unknown
        if actual_outcome == "incumbent":
            target = 1
        elif actual_outcome == "challenger":
            target = 0
        else:
            target = np.nan
            
        # Fatigue calculations
        fatigue = fatigue_cache.get(f"upcoming_{up_id}", {})
        government_years_in_power = fatigue.get("government_years_in_power", np.nan)
        ideology_years_in_power = fatigue.get("ideology_years_in_power", np.nan)
        years_since_last_alternation = fatigue.get("years_since_last_alternation", np.nan)
        historical_alternation_rate = fatigue.get("historical_alternation_rate", np.nan)
        government_continuity_score = government_years_in_power * v2xlg_legcon if (pd.notna(government_years_in_power) and pd.notna(v2xlg_legcon)) else np.nan
        
        # Institutional calculations
        government_type_majority = 1.0 if (pd.notna(prior_iss) and prior_iss >= 0.50) else (0.0 if pd.notna(prior_iss) else np.nan)
        government_type_minority = 1.0 if (pd.notna(prior_iss) and prior_iss < 0.50) else (0.0 if pd.notna(prior_iss) else np.nan)
        government_type_coalition = 1.0 if (pd.notna(prior_ep) and prior_ep > 2.5) else (0.0 if pd.notna(prior_ep) else np.nan)
        government_type_caretaker = 0.0
        government_type_technocratic = 0.0
        coalition_size = prior_ep
        coalition_fragmentation = polarization
        democratic_age = year - democracy_start_year[code] if (code in democracy_start_year and year >= democracy_start_year[code]) else 0
        constitutional_rigidity = (v2xlg_legcon + v2x_jucon) / 2.0 if (pd.notna(v2xlg_legcon) and pd.notna(v2x_jucon)) else np.nan
        
        # Media & Sentiment calculations
        government_sentiment_score = gov_avg_tone
        government_news_salience = (gov_news_volume + opp_news_volume) / (vol + 1.0) if pd.notna(vol) else np.nan
        government_media_balance = opp_gov_tone_diff
        legislative_productivity = v2xlg_legcon # legislative constraints proxy
            
        data.append({
            "election_id": f"upcoming_{up_id}",
            "source": "upcoming",
            "country_code": code,
            "region": region_cache.get(code, "Unknown"),
            "year": year,
            "election_date": date_str,
            "is_scheduled": scheduled,
            "is_presidential": is_presidential,
            "opposition_boycott": np.nan,
            "monitors_present": np.nan,
            "clean_index": clean_idx,
            "polyarchy_index": poly_idx,
            "regime_type": regime,
            "protest_pressure": protest,
            "coop_score": coop,
            "conflict_level": conflict,
            "coercion_ratio": coercion,
            "aid_ratio": aid,
            "protest_velocity": protest_vel,
            "conflict_velocity": conflict_vel,
            "raw_protest_score": raw_protest,
            "raw_conflict_score": raw_conflict,
            "raw_coop_score": raw_coop,
            "raw_coercion_score": raw_coercion,
            "raw_aid_score": raw_aid,
            "average_news_tone": avg_tone,
            "total_news_volume": vol,
            "gov_news_volume": gov_news_volume,
            "opp_news_volume": opp_news_volume,
            "gov_avg_tone": gov_avg_tone,
            "opp_avg_tone": opp_avg_tone,
            "opp_gov_volume_ratio": opp_gov_volume_ratio,
            "opp_gov_tone_diff": opp_gov_tone_diff,
            "gdp_growth": gdp,
            "inflation": inf,
            "gdp_shock": gdp_dev,
            "inflation_shock": inf_dev,
            "prior_turnout": prior_to,
            "prior_margin_of_victory": prior_mov,
            "prior_effective_parties": prior_ep,
            "prior_incumbent_seat_share": prior_iss,
            "prior_margin_presidential_interaction": prior_margin_presidential_interaction,
            "party_fragmentation_legislative_interaction": party_fragmentation_legislative_interaction,
            "clean_index_delta_1yr": clean_delta_1yr,
            "clean_index_delta_3yr": clean_delta_3yr,
            "polyarchy_delta_1yr": poly_delta_1yr,
            "polyarchy_delta_3yr": poly_delta_3yr,
            "oil_price_shock": oil_price_shock,
            "oil_exporter_shock": oil_exporter_shock,
            "fed_rate_change": fed_rate_change,
            "us_president_ideology": us_pres,
            "incumbent_ran": incumbent_ran,
            "has_successor": has_successor,
            "currency_depreciation_6mo": currency_depreciation_6mo,
            "currency_depreciation_12mo": currency_depreciation_12mo,
            "currency_volatility_6mo": currency_volatility_6mo,
            "electoral_system_majoritarian": maj_sys,
            "electoral_system_pr": pr_sys,
            "political_polarization": polarization,
            "incumbent_seat_buffer": incumbent_seat_buffer,
            "is_majority_fragile": is_majority_fragile,
            "pr_fragmentation_interaction": pr_fragmentation_interaction,
            "polarization_fragmentation_interaction": polarization_fragmentation_interaction,
            # Expanded Structural Features
            "government_years_in_power": government_years_in_power,
            "ideology_years_in_power": ideology_years_in_power,
            "years_since_last_alternation": years_since_last_alternation,
            "historical_alternation_rate": historical_alternation_rate,
            "government_continuity_score": government_continuity_score,
            "cabinet_reshuffles_current_term": v2cafres,
            "government_type_majority": government_type_majority,
            "government_type_minority": government_type_minority,
            "government_type_coalition": government_type_coalition,
            "government_type_caretaker": government_type_caretaker,
            "government_type_technocratic": government_type_technocratic,
            "coalition_size": coalition_size,
            "coalition_fragmentation": coalition_fragmentation,
            "democratic_age": democratic_age,
            "constitutional_rigidity": constitutional_rigidity,
            "pedersen_index": prior_ped,
            "electoral_volatility_10yr": electoral_volatility_10yr,
            "electoral_volatility_20yr": electoral_volatility_20yr,
            "electoral_volatility_change": electoral_volatility_change,
            "inflation_acceleration": inflation_acceleration,
            "gdp_growth_acceleration": gdp_growth_acceleration,
            "unemployment_acceleration": unemployment_acceleration,
            "real_wage_growth": gdp_per_capita_growth,
            "migration_shock": migration_shock,
            "refugee_inflow_shock": refugee_inflow_shock,
            "government_sentiment_score": government_sentiment_score,
            "government_corruption_events": corruption_events,
            "government_corruption_ratio": corruption_ratio,
            "legislative_productivity": legislative_productivity,
            "government_news_salience": government_news_salience,
            "government_media_balance": government_media_balance,
            "target_outcome": target,
            "target_integrity": np.nan,
            "gdelt_coverage": gdelt_coverage
        })
        
    df = pd.DataFrame(data)
    
    if df.empty:
        return df
    
    # Sort by country and year to compute consecutive incumbent wins (term fatigue)
    df = df.sort_values(by=["country_code", "year"]).reset_index(drop=True)
    
    # 1-year economic diffs (NaN-safe)
    df["gdp_growth_diff_1yr"] = df.groupby("country_code")["gdp_growth"].diff(1)
    df["inflation_diff_1yr"] = df.groupby("country_code")["inflation"].diff(1)
    
    incumbent_term_count = []
    for idx, row in df.iterrows():
        code = row["country_code"]
        yr = row["year"]
        
        # Get past elections for this country (only those with known outcomes)
        past = df[(df["country_code"] == code) & (df["year"] < yr) & df["target_outcome"].notna()].sort_values(by="year", ascending=False)
        
        consecutive_wins = 0
        for _, p_row in past.iterrows():
            if p_row["target_outcome"] == 1: # Incumbent won
                consecutive_wins += 1
            else:
                break
        incumbent_term_count.append(consecutive_wins)
        
    df["incumbent_term_count"] = incumbent_term_count
    
    # Parse dates safely
    dates = []
    for idx, row in df.iterrows():
        dt = row.get("election_date")
        if pd.notna(dt) and dt != "" and dt != "None":
            try:
                dates.append(pd.to_datetime(dt))
            except:
                dates.append(pd.to_datetime(f"{int(row['year'])}-06-01"))
        else:
            dates.append(pd.to_datetime(f"{int(row['year'])}-06-01"))
    df["date_parsed"] = dates

    # Calculate 36mo and 24mo regional challenger tides (rolling date based)
    region_lookup = {}
    for idx, row in df.iterrows():
        reg = row["region"]
        if reg not in region_lookup:
            region_lookup[reg] = []
        if pd.notna(row["target_outcome"]):
            region_lookup[reg].append((row["date_parsed"], row["target_outcome"], row["election_id"]))

    regional_tide_36mo = []
    regional_tide_24mo = []
    
    for idx, row in df.iterrows():
        curr_date = row["date_parsed"]
        curr_reg = row["region"]
        curr_id = row["election_id"]
        
        candidates = region_lookup.get(curr_reg, [])
        
        # 36 months candidates
        outcomes_36 = [
            out for dt, out, eid in candidates 
            if curr_date - pd.Timedelta(days=1095) <= dt < curr_date and eid != curr_id
        ]
        
        # 24 months candidates
        outcomes_24 = [
            out for dt, out, eid in candidates 
            if curr_date - pd.Timedelta(days=730) <= dt < curr_date and eid != curr_id
        ]
        
        t_36 = np.mean([1 - o for o in outcomes_36]) if outcomes_36 else 0.40
        t_24 = np.mean([1 - o for o in outcomes_24]) if outcomes_24 else 0.40
        
        regional_tide_36mo.append(t_36)
        regional_tide_24mo.append(t_24)
        
    df["regional_tide_36mo"] = regional_tide_36mo
    df["regional_tide_24mo"] = regional_tide_24mo
    
    # Calculate Regional Challenger Tide (anti-incumbent wave)
    regional_challenger_tide = []
    for idx, row in df.iterrows():
        reg = row["region"]
        yr = row["year"]
        
        regional_elections = df[(df["region"] == reg) & (df["year"] >= yr - 3) & (df["year"] < yr) & df["target_outcome"].notna()]
        
        if regional_elections.empty:
            regional_challenger_tide.append(0.5)
        else:
            challenger_wins = len(regional_elections[regional_elections["target_outcome"] == 0])
            tide = challenger_wins / len(regional_elections)
            regional_challenger_tide.append(tide)
            
    df["regional_challenger_tide"] = regional_challenger_tide

    # Calculate Global Challenger Tide
    global_challenger_tide = []
    for idx, row in df.iterrows():
        yr = row["year"]
        global_elections = df[(df["year"] >= yr - 3) & (df["year"] < yr) & df["target_outcome"].notna()]
        
        if global_elections.empty:
            global_challenger_tide.append(0.5)
        else:
            challenger_wins = len(global_elections[global_elections["target_outcome"] == 0])
            tide = challenger_wins / len(global_elections)
            global_challenger_tide.append(tide)
            
    df["global_challenger_tide"] = global_challenger_tide

    # Calculate Regime-Type Challenger Tide
    regime_challenger_tide = []
    for idx, row in df.iterrows():
        reg_t = row["regime_type"]
        yr = row["year"]
        if pd.isna(reg_t):
            regime_challenger_tide.append(0.5)
            continue
        same_regime_elections = df[(df["regime_type"] == reg_t) & (df["year"] >= yr - 3) & (df["year"] < yr) & df["target_outcome"].notna()]
        if same_regime_elections.empty:
            regime_challenger_tide.append(0.5)
        else:
            challenger_wins = len(same_regime_elections[same_regime_elections["target_outcome"] == 0])
            tide = challenger_wins / len(same_regime_elections)
            regime_challenger_tide.append(tide)
            
    df["regime_challenger_tide"] = regime_challenger_tide

    # Calculate Country Historical win rates (smoothed via Bayesian prior)
    incumbent_rates = []
    global_incumbent_rate = 0.593
    m_prior = 3
    for idx, row in df.iterrows():
        code = row["country_code"]
        yr = row["year"]
        past_c = df[(df["country_code"] == code) & (df["year"] < yr) & df["target_outcome"].notna()]
        if past_c.empty:
            rate = global_incumbent_rate
        else:
            wins = past_c["target_outcome"].sum()
            total = len(past_c)
            rate = (wins + m_prior * global_incumbent_rate) / (total + m_prior)
        incumbent_rates.append(rate)
    df["country_incumbent_win_rate"] = incumbent_rates

    # Calculate Global Economic Shock (average gdp_shock across all countries in that year)
    df["global_economic_shock"] = df.groupby("year")["gdp_shock"].transform("mean")

    # Calculate Dynamic Spatial Lags for all 5 network layers
    weight_types = ["trade", "unga_voting", "alliance", "contiguity", "proximity"]
    
    for wt in weight_types:
        lag_clean_list = []
        lag_outcome_list = []
        lag_protest_list = []
        lag_conflict_list = []
        lag_dem_delta_list = []
        lag_cln_delta_list = []
        
        for idx, row in df.iterrows():
            yr = row["year"]
            source_code = row["country_code"]
            
            src_weights = weights_cache.get(wt, {}).get(yr, {}).get(source_code, {})
            
            if not src_weights:
                lag_clean_list.append(np.nan)
                lag_outcome_list.append(np.nan)
                lag_protest_list.append(np.nan)
                lag_conflict_list.append(np.nan)
                lag_dem_delta_list.append(np.nan)
                lag_cln_delta_list.append(np.nan)
                continue
                
            other_elections = df[(df["year"] <= yr) & (df["year"] >= yr - 4) & (df["country_code"] != source_code)]
            
            if other_elections.empty:
                lag_clean_list.append(np.nan)
                lag_outcome_list.append(np.nan)
                lag_protest_list.append(np.nan)
                lag_conflict_list.append(np.nan)
                lag_dem_delta_list.append(np.nan)
                lag_cln_delta_list.append(np.nan)
                continue
                
            total_w_c = 0.0
            sum_w_c = 0.0
            total_w_o = 0.0
            sum_w_o = 0.0
            total_w_p = 0.0
            sum_w_p = 0.0
            total_w_f = 0.0
            sum_w_f = 0.0
            total_w_dd = 0.0
            sum_w_dd = 0.0
            total_w_cd = 0.0
            sum_w_cd = 0.0
            
            for _, other in other_elections.iterrows():
                tgt = other["country_code"]
                if tgt in src_weights:
                    w = src_weights[tgt]
                    if pd.notna(other["clean_index"]):
                        sum_w_c += w * other["clean_index"]
                        total_w_c += w
                    if pd.notna(other["target_outcome"]):
                        sum_w_o += w * other["target_outcome"]
                        total_w_o += w
                    if pd.notna(other["protest_pressure"]):
                        sum_w_p += w * other["protest_pressure"]
                        total_w_p += w
                    if pd.notna(other["conflict_level"]):
                        sum_w_f += w * other["conflict_level"]
                        total_w_f += w
                    if pd.notna(other["polyarchy_delta_1yr"]):
                        sum_w_dd += w * other["polyarchy_delta_1yr"]
                        total_w_dd += w
                    if pd.notna(other["clean_index_delta_1yr"]):
                        sum_w_cd += w * other["clean_index_delta_1yr"]
                        total_w_cd += w
                    
            lag_clean_list.append(sum_w_c / total_w_c if total_w_c > 0 else np.nan)
            lag_outcome_list.append(sum_w_o / total_w_o if total_w_o > 0 else np.nan)
            lag_protest_list.append(sum_w_p / total_w_p if total_w_p > 0 else np.nan)
            lag_conflict_list.append(sum_w_f / total_w_f if total_w_f > 0 else np.nan)
            lag_dem_delta_list.append(sum_w_dd / total_w_dd if total_w_dd > 0 else np.nan)
            lag_cln_delta_list.append(sum_w_cd / total_w_cd if total_w_cd > 0 else np.nan)
            
        df[f"spatial_lag_clean_{wt}"] = lag_clean_list
        df[f"spatial_lag_outcome_{wt}"] = lag_outcome_list
        df[f"spatial_lag_protest_{wt}"] = lag_protest_list
        df[f"spatial_lag_conflict_{wt}"] = lag_conflict_list
        df[f"spatial_lag_democracy_delta_{wt}"] = lag_dem_delta_list
        df[f"spatial_lag_clean_delta_{wt}"] = lag_cln_delta_list
    
    # Calculate Executive Candidacy / Term Limit Features
    # Fill missing incumbent_ran with 1 by default (assume incumbent ran/running if not specified or legislative)
    # Fill missing has_successor with 0
    df["incumbent_ran_filled"] = df["incumbent_ran"].fillna(1)
    df["has_successor_filled"] = df["has_successor"].fillna(0)
    
    # is_open_seat is 1 if is_presidential is 1 and incumbent_ran is 0
    # is_open_seat is 0 if is_presidential is 1 and incumbent_ran is 1
    # is_open_seat is NaN if is_presidential is 0
    df["is_open_seat"] = np.where(
        df["is_presidential"] == 1,
        (df["incumbent_ran_filled"] == 0).astype(float),
        np.nan
    )
    
    # successor_run is 1 if is_presidential is 1, incumbent did not run, and has successor
    # successor_run is 0 if is_presidential is 1 and successor did not run
    # successor_run is NaN if is_presidential is 0
    df["successor_run"] = np.where(
        df["is_presidential"] == 1,
        ((df["incumbent_ran_filled"] == 0) & (df["has_successor_filled"] == 1)).astype(float),
        np.nan
    )

    # Calculate new interaction terms (Hypothesis 2, 3, and 1 interactions)
    df["economy_regime_interaction"] = df["currency_depreciation_6mo"] * df["polyarchy_index"]
    df["protest_regime_interaction"] = df["protest_pressure"] * df["polyarchy_index"]
    df["open_seat_prior_margin_interaction"] = df["is_open_seat"] * df["prior_margin_of_victory"]
    df["term_fatigue_economy_interaction"] = df["incumbent_term_count"] * df["currency_depreciation_6mo"]
    df["opposition_momentum_margin_interaction"] = df["opp_gov_volume_ratio"] * df["prior_margin_of_victory"]

    # Calculate cross-pollination features
    compute_cross_pollination_features(df)

    # Calculate global pressures
    df["global_inflation_pressure"] = df.groupby("year")["inflation"].transform("mean").fillna(0.0)
    df["global_energy_pressure"] = df.groupby("year")["oil_price_shock"].transform("mean").fillna(0.0)
    df["global_migration_pressure"] = df.groupby("year")["migration_shock"].transform("mean").fillna(0.0)
    df["global_democratic_backsliding"] = df.groupby("year")["polyarchy_delta_1yr"].transform("mean").fillna(0.0)
    df["global_incumbent_punishment_index"] = df.groupby("year")["target_outcome"].transform(lambda x: 1.0 - x.dropna().mean() if not x.dropna().empty else 0.5).fillna(0.5)

    # Calculate Latent Global Political Climate Indices using PCA
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    
    # Group by year and calculate global annual indicators
    annual_indicators = df.groupby("year").agg({
        "target_outcome": lambda x: 1.0 - x.dropna().mean() if not x.dropna().empty else np.nan,
        "polyarchy_delta_1yr": "mean",
        "inflation": "mean",
        "gdp_growth": "mean",
        "protest_pressure": "mean"
    }).reset_index()
    
    # Fill missing values in annual indicators with global means
    for col in ["target_outcome", "polyarchy_delta_1yr", "inflation", "gdp_growth", "protest_pressure"]:
        if col in annual_indicators.columns:
            annual_indicators[col] = annual_indicators[col].fillna(annual_indicators[col].mean()).fillna(0.0)
        else:
            annual_indicators[col] = 0.0
        
    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(annual_indicators[["target_outcome", "polyarchy_delta_1yr", "inflation", "gdp_growth", "protest_pressure"]])
    
    # Fit PCA for latent global political climate
    pca_climate = PCA(n_components=1)
    annual_indicators["latent_global_political_climate"] = pca_climate.fit_transform(X)[:, 0]
    
    # Fit PCA for economic anxiety
    X_econ = StandardScaler().fit_transform(annual_indicators[["inflation", "gdp_growth"]])
    pca_econ = PCA(n_components=1)
    annual_indicators["latent_global_economic_anxiety"] = pca_econ.fit_transform(X_econ)[:, 0]
    
    # Other latent indices
    annual_indicators["latent_global_anti_incumbent_index"] = annual_indicators["target_outcome"]
    annual_indicators["latent_global_democracy_trend"] = annual_indicators["polyarchy_delta_1yr"]
    
    # Map back to df
    climate_map = dict(zip(annual_indicators["year"], annual_indicators["latent_global_political_climate"]))
    econ_map = dict(zip(annual_indicators["year"], annual_indicators["latent_global_economic_anxiety"]))
    anti_inc_map = dict(zip(annual_indicators["year"], annual_indicators["latent_global_anti_incumbent_index"]))
    dem_trend_map = dict(zip(annual_indicators["year"], annual_indicators["latent_global_democracy_trend"]))
    
    df["latent_global_political_climate"] = df["year"].map(climate_map)
    df["latent_global_economic_anxiety"] = df["year"].map(econ_map)
    df["latent_global_anti_incumbent_index"] = df["year"].map(anti_inc_map)
    df["latent_global_democracy_trend"] = df["year"].map(dem_trend_map)

    # Add country dummies
    cursor.execute("SELECT country_code FROM countries ORDER BY country_code;")
    all_countries = [r[0] for r in cursor.fetchall()]
    for c_code in all_countries:
        df[f"country_{c_code}"] = (df["country_code"] == c_code).astype(int)
        
    # Update FEATURES in place
    FEATURES.clear()
    FEATURES.extend(BASE_FEATURES)
    FEATURES.extend([f"country_{c_code}" for c_code in all_countries])
    
    # Compute data completeness and source flags (using BASE_FEATURES so OHE dummies don't inflate completeness)
    df["data_completeness"] = df.apply(lambda r: compute_data_completeness(r, BASE_FEATURES), axis=1)
    df["data_source_flags"] = df.apply(compute_data_source_flags, axis=1)
        
    return df

# 3. Moran's I
def compute_moran_metrics(df, weight_type="trade", year=2020):
    from spatial.calc_weights import get_row_normalized_weights
    
    try:
        w_obj, countries = get_row_normalized_weights(weight_type, year)
    except Exception as e:
        print(f"Skipping Moran's I for {weight_type} in {year}: {e}")
        return None, [], []
        
    conn = get_connection()
    df_vdem = pd.read_sql_query(
        "SELECT country_code, clean_elections_index FROM vdem_indicators WHERE year = ?;",
        conn,
        params=(year,)
    )
    conn.close()
    
    if df_vdem.empty:
        return None, [], []
        
    df_aligned = df_vdem[df_vdem["country_code"].isin(countries)].set_index("country_code")
    df_aligned = df_aligned.reindex(countries)
    df_aligned["clean_elections_index"] = df_aligned["clean_elections_index"].fillna(0.5)
    
    y = df_aligned["clean_elections_index"].values
    
    mi = Moran(y, w_obj)
    lm = Moran_Local(y, w_obj)
    
    print(f"\n--- Dynamic Spatial Autocorrelation (Moran's I) in {year} using {weight_type} matrix ---")
    print(f"Global Moran's I score: {mi.I:.4f} (p-value: {mi.p_sim:.4f})")
    
    return mi.I, [], []

# 3.5 Probability Calibration
def calibrate_probability(prob, T=None):
    """Calibrate probabilities using temperature scaling."""
    if T is None:
        T = TUNED_PARAMS["ensemble_balance"]["T"]
    prob = np.clip(prob, 1e-5, 1.0 - 1e-5)
    logit = np.log(prob / (1.0 - prob))
    cal_prob = 1.0 / (1.0 + np.exp(-logit / T))
    return cal_prob


# 4. Rolling Origin Validation
def evaluate_predictive_models(df):
    """Evaluate models using rolling origin cross-validation with separate Exec/Leg models."""
    print("\n--- Running Separate Executive & Legislative Ensemble Benchmark ---")
    origins = [2010, 2014, 2018, 2022]
    
    exec_features = [f for f in FEATURES if f not in LEG_SPECIFIC]
    leg_features = [f for f in FEATURES if f not in EXEC_SPECIFIC]
    
    # Load tuned parameters
    exec_params = TUNED_PARAMS.get("executive_model", {
        "xgb_weight": 0.9,
        "T": 0.78,
        "classification_threshold": 0.50
    })
    leg_params = TUNED_PARAMS.get("legislative_model", {
        "xgb_weight": 0.5,
        "T": 0.80,
        "classification_threshold": 0.50
    })
    
    cb_accs, xgb_accs, ensemble_accs = [], [], []
    exec_accs, leg_accs = [], []
    
    # Only train/test on rows with known outcomes
    df_known = df[df["target_outcome"].notna()].copy()
    
    for y in origins:
        train_df = df_known[df_known["year"] <= y]
        test_df = df_known[(df_known["year"] > y) & (df_known["year"] <= y + 4)]
        
        if train_df.empty or test_df.empty:
            continue
            
        train_exec = train_df[train_df["is_presidential"] == 1]
        train_leg = train_df[train_df["is_presidential"] == 0]
        test_exec = test_df[test_df["is_presidential"] == 1]
        test_leg = test_df[test_df["is_presidential"] == 0]
        
        # Predictions array for combined evaluation
        probs_cb = np.zeros(len(test_df))
        probs_xgb = np.zeros(len(test_df))
        probs_ens = np.zeros(len(test_df))
        test_indices = test_df.index.tolist()
        
        # --- Executive Model Group ---
        if not train_exec.empty and not test_exec.empty:
            X_tr_ex, y_tr_ex = train_exec[exec_features], train_exec["target_outcome"]
            X_te_ex, y_te_ex = test_exec[exec_features], test_exec["target_outcome"]
            
            ex_cb = cb.CatBoostClassifier(
                iterations=200,
                depth=TUNED_PARAMS["model_parameters"]["cb_depth"],
                learning_rate=TUNED_PARAMS["model_parameters"]["cb_lr"],
                verbose=0,
                random_seed=42
            ).fit(X_tr_ex, y_tr_ex)
            
            ex_xgb = xgb.XGBClassifier(
                max_depth=TUNED_PARAMS["model_parameters"]["xgb_max_depth"],
                learning_rate=TUNED_PARAMS["model_parameters"]["xgb_lr"],
                n_estimators=150,
                eval_metric='logloss',
                random_state=42
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
            
            # Save predictions
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
                iterations=200,
                depth=TUNED_PARAMS["model_parameters"]["cb_depth"],
                learning_rate=TUNED_PARAMS["model_parameters"]["cb_lr"],
                verbose=0,
                random_seed=42
            ).fit(X_tr_lg, y_tr_lg)
            
            lg_xgb = xgb.XGBClassifier(
                max_depth=TUNED_PARAMS["model_parameters"]["xgb_max_depth"],
                learning_rate=TUNED_PARAMS["model_parameters"]["xgb_lr"],
                n_estimators=150,
                eval_metric='logloss',
                random_state=42
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
            
            # Save predictions
            for idx_in_test, original_idx in enumerate(test_leg.index):
                pos = test_indices.index(original_idx)
                probs_cb[pos] = p_cb_lg[idx_in_test]
                probs_xgb[pos] = p_xgb_lg[idx_in_test]
                probs_ens[pos] = p_ens_lg_cal[idx_in_test]

        # Evaluate combined outcomes using respective thresholds
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
        acc_ensemble = accuracy_score(y_test, preds_ens)
        
        cb_accs.append(acc_cb)
        xgb_accs.append(acc_xgb)
        ensemble_accs.append(acc_ensemble)
        
        # Track separate metrics
        acc_ex = 0.0
        acc_lg = 0.0
        if not test_exec.empty:
            indices_ex = [test_indices.index(idx) for idx in test_exec.index]
            acc_ex = accuracy_score(test_exec["target_outcome"], preds_ens[indices_ex])
            exec_accs.append(acc_ex)
        if not test_leg.empty:
            indices_lg = [test_indices.index(idx) for idx in test_leg.index]
            acc_lg = accuracy_score(test_leg["target_outcome"], preds_ens[indices_lg])
            leg_accs.append(acc_lg)
            
        print(f" Origin Year <= {y} -> Tested on {y+1} to {y+4} ({len(test_df)} elections):")
        print(f"   Accuracy: CatBoost: {acc_cb:.4f} | XGBoost: {acc_xgb:.4f} | Ensemble: {acc_ensemble:.4f}")
        if not test_exec.empty and not test_leg.empty:
            print(f"     Exec Accuracy: {acc_ex:.4f} | Leg Accuracy: {acc_lg:.4f}")
            
    avg_cb = np.mean(cb_accs) if cb_accs else 0.0
    avg_xgb = np.mean(xgb_accs) if xgb_accs else 0.0
    avg_ensemble = np.mean(ensemble_accs) if ensemble_accs else 0.0
    avg_exec = np.mean(exec_accs) if exec_accs else 0.0
    avg_leg = np.mean(leg_accs) if leg_accs else 0.0
    
    print("\n--- Summary Benchmark Results ---")
    print(f"Model A (Outcomes) Average Accuracies:")
    print(f"  - CatBoost Classifier:               {avg_cb:.4f}")
    print(f"  - XGBoost Classifier:                {avg_xgb:.4f}")
    print(f"  - Voting Ensemble Combined:          {avg_ensemble:.4f}")
    print(f"  - Executive Model Ensemble:          {avg_exec:.4f}")
    print(f"  - Legislative Model Ensemble:        {avg_leg:.4f}")
    
    return avg_ensemble


# 5. Cascading Prediction for Upcoming Elections
def update_dynamic_features_for_year(df, target_year, weights_cache, region_cache):
    """
    Recalculate target_outcome-dependent features for all elections in target_year,
    using outcomes of elections in previous years (including soft predicted outcomes).
    """
    # 1. incumbent_term_count
    for idx, row in df[df["year"] == target_year].iterrows():
        code = row["country_code"]
        past = df[(df["country_code"] == code) & (df["year"] < target_year) & df["target_outcome"].notna()].sort_values(by="year", ascending=False)
        consecutive_wins = 0
        for _, p_row in past.iterrows():
            val = round(p_row["target_outcome"])
            if val == 1:
                consecutive_wins += 1
            else:
                break
        df.loc[idx, "incumbent_term_count"] = consecutive_wins

    # 2. regional_challenger_tide
    for idx, row in df[df["year"] == target_year].iterrows():
        reg = row["region"]
        regional_elections = df[(df["region"] == reg) & (df["year"] >= target_year - 3) & (df["year"] < target_year) & df["target_outcome"].notna()]
        if regional_elections.empty:
            df.loc[idx, "regional_challenger_tide"] = 0.5
        else:
            challenger_weight_sum = (1 - regional_elections["target_outcome"]).sum()
            df.loc[idx, "regional_challenger_tide"] = challenger_weight_sum / len(regional_elections)

    # 2.5. regional_tide_36mo and regional_tide_24mo (rolling date based)
    region_lookup = {}
    for idx, row in df.iterrows():
        reg = row["region"]
        if reg not in region_lookup:
            region_lookup[reg] = []
        if pd.notna(row["target_outcome"]):
            region_lookup[reg].append((row["date_parsed"], row["target_outcome"], row["election_id"]))

    for idx, row in df[df["year"] == target_year].iterrows():
        curr_date = row["date_parsed"]
        curr_reg = row["region"]
        curr_id = row["election_id"]
        
        candidates = region_lookup.get(curr_reg, [])
        
        # 36 months candidates
        outcomes_36 = [
            out for dt, out, eid in candidates 
            if curr_date - pd.Timedelta(days=1095) <= dt < curr_date and eid != curr_id
        ]
        
        # 24 months candidates
        outcomes_24 = [
            out for dt, out, eid in candidates 
            if curr_date - pd.Timedelta(days=730) <= dt < curr_date and eid != curr_id
        ]
        
        t_36 = np.mean([1 - o for o in outcomes_36]) if outcomes_36 else 0.40
        t_24 = np.mean([1 - o for o in outcomes_24]) if outcomes_24 else 0.40
        
        df.loc[idx, "regional_tide_36mo"] = t_36
        df.loc[idx, "regional_tide_24mo"] = t_24

    # 3. global_challenger_tide
    global_elections = df[(df["year"] >= target_year - 3) & (df["year"] < target_year) & df["target_outcome"].notna()]
    global_tide = 0.5
    if not global_elections.empty:
        challenger_weight_sum = (1 - global_elections["target_outcome"]).sum()
        global_tide = challenger_weight_sum / len(global_elections)
    df.loc[df["year"] == target_year, "global_challenger_tide"] = global_tide

    # 4. regime_challenger_tide
    for idx, row in df[df["year"] == target_year].iterrows():
        reg_t = row["regime_type"]
        if pd.isna(reg_t):
            df.loc[idx, "regime_challenger_tide"] = 0.5
            continue
        same_regime_elections = df[(df["regime_type"] == reg_t) & (df["year"] >= target_year - 3) & (df["year"] < target_year) & df["target_outcome"].notna()]
        if same_regime_elections.empty:
            df.loc[idx, "regime_challenger_tide"] = 0.5
        else:
            challenger_weight_sum = (1 - same_regime_elections["target_outcome"]).sum()
            df.loc[idx, "regime_challenger_tide"] = challenger_weight_sum / len(same_regime_elections)

    # 5. spatial_lag_outcome_{wt} (concurrent year outcomes are excluded from the lag to avoid simultaneity bias)
    weight_types = ["trade", "unga_voting", "alliance", "contiguity"]
    for wt in weight_types:
        for idx, row in df[df["year"] == target_year].iterrows():
            source_code = row["country_code"]
            src_weights = weights_cache.get(wt, {}).get(target_year, {}).get(source_code, {})
            if not src_weights:
                df.loc[idx, f"spatial_lag_outcome_{wt}"] = np.nan
                continue
            other_elections = df[(df["year"] < target_year) & (df["year"] >= target_year - 4) & (df["country_code"] != source_code)]
            if other_elections.empty:
                df.loc[idx, f"spatial_lag_outcome_{wt}"] = np.nan
                continue
            total_w_o = 0.0
            sum_w_o = 0.0
            for _, other in other_elections.iterrows():
                tgt = other["country_code"]
                if tgt in src_weights:
                    w = src_weights[tgt]
                    if pd.notna(other["target_outcome"]):
                        outcome = other["target_outcome"]
                        is_pred = other.get("is_predicted", 0)
                        conf = other.get("prediction_confidence", 1.0)
                        if pd.isna(conf) or is_pred == 0:
                            conf = 1.0
                        # Calculate soft outcome weighted by confidence
                        soft_outcome = 0.5 + (outcome - 0.5) * (2.0 * (conf - 0.5))
                        sum_w_o += w * soft_outcome
                        total_w_o += w
            df.loc[idx, f"spatial_lag_outcome_{wt}"] = sum_w_o / total_w_o if total_w_o > 0 else np.nan

    # 6. Recalculate cross-pollination features for target_year
    df_sorted = df.sort_values(by="election_date")
    for idx, row in df[df["year"] == target_year].iterrows():
        code = row["country_code"]
        curr_date = row["election_date"]
        is_pres = row["is_presidential"]
        
        # Filter for prior alternate elections
        past_alt = df_sorted[
            (df_sorted["country_code"] == code) & 
            (df_sorted["election_date"] < curr_date) & 
            (df_sorted["is_presidential"] != is_pres)
        ]
        
        if not past_alt.empty:
            last_el = past_alt.iloc[-1]
            df.loc[idx, "last_alt_incumbent_won"] = last_el["target_outcome"]
            df.loc[idx, "last_alt_margin"] = last_el["prior_margin_of_victory"]
            df.loc[idx, "months_since_last_alt"] = months_between(last_el["election_date"], curr_date)


def recalculate_global_features(df):
    """
    Recalculate global indicators (global_incumbent_punishment_index and PCA-based latent global variables)
    now that predictions for a target year have been populated.
    """
    import numpy as np
    import pandas as pd
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    # 1. global_incumbent_punishment_index
    df["global_incumbent_punishment_index"] = df.groupby("year")["target_outcome"].transform(
        lambda x: 1.0 - x.dropna().mean() if not x.dropna().empty else 0.5
    ).fillna(0.5)

    # 2. global annual indicators PCA
    annual_indicators = df.groupby("year").agg({
        "target_outcome": lambda x: 1.0 - x.dropna().mean() if not x.dropna().empty else np.nan,
        "polyarchy_delta_1yr": "mean",
        "inflation": "mean",
        "gdp_growth": "mean",
        "protest_pressure": "mean"
    }).reset_index()
    
    for col in ["target_outcome", "polyarchy_delta_1yr", "inflation", "gdp_growth", "protest_pressure"]:
        if col in annual_indicators.columns:
            annual_indicators[col] = annual_indicators[col].fillna(annual_indicators[col].mean()).fillna(0.0)
        else:
            annual_indicators[col] = 0.0
            
    scaler = StandardScaler()
    X = scaler.fit_transform(annual_indicators[["target_outcome", "polyarchy_delta_1yr", "inflation", "gdp_growth", "protest_pressure"]])
    
    pca_climate = PCA(n_components=1)
    annual_indicators["latent_global_political_climate"] = pca_climate.fit_transform(X)[:, 0]
    
    X_econ = StandardScaler().fit_transform(annual_indicators[["inflation", "gdp_growth"]])
    pca_econ = PCA(n_components=1)
    annual_indicators["latent_global_economic_anxiety"] = pca_econ.fit_transform(X_econ)[:, 0]
    
    annual_indicators["latent_global_anti_incumbent_index"] = annual_indicators["target_outcome"]
    annual_indicators["latent_global_democracy_trend"] = annual_indicators["polyarchy_delta_1yr"]
    
    climate_map = dict(zip(annual_indicators["year"], annual_indicators["latent_global_political_climate"]))
    econ_map = dict(zip(annual_indicators["year"], annual_indicators["latent_global_economic_anxiety"]))
    anti_inc_map = dict(zip(annual_indicators["year"], annual_indicators["latent_global_anti_incumbent_index"]))
    dem_trend_map = dict(zip(annual_indicators["year"], annual_indicators["latent_global_democracy_trend"]))
    
    df["latent_global_political_climate"] = df["year"].map(climate_map)
    df["latent_global_economic_anxiety"] = df["year"].map(econ_map)
    df["latent_global_anti_incumbent_index"] = df["year"].map(anti_inc_map)
    df["latent_global_democracy_trend"] = df["year"].map(dem_trend_map)


def predict_with_cascading(df, train_cutoff_year=None, conn=None, return_df=False):
    """
    Predict outcomes for elections with unknown results, using cascading logic:
    elections are predicted in chronological order, and each prediction is fed back 
    into the dataset as a "soft" outcome (weighted by model confidence) for subsequent
    predictions.
    """
    df = df.copy()
    threshold = TUNED_PARAMS.get("classification_threshold", 0.50)
    
    # Separate known-outcome elections (training set) from unknowns (prediction set)
    df_known = df[df["target_outcome"].notna()].copy()
    df_unknown = df[df["target_outcome"].isna()].copy()
    
    if df_unknown.empty:
        print("No elections to predict (all have known outcomes).")
        return pd.DataFrame()
    
    if train_cutoff_year is not None:
        df_train = df_known[df_known["year"] <= train_cutoff_year].copy()
    else:
        df_train = df_known.copy()
    
    if df_train.empty:
        print("No training data available.")
        return pd.DataFrame()
        
    exec_features = [f for f in FEATURES if f not in LEG_SPECIFIC]
    leg_features = [f for f in FEATURES if f not in EXEC_SPECIFIC]
    
    exec_params = TUNED_PARAMS.get("executive_model", {
        "xgb_weight": 0.9,
        "T": 0.78,
        "classification_threshold": 0.50
    })
    leg_params = TUNED_PARAMS.get("legislative_model", {
        "xgb_weight": 0.5,
        "T": 0.80,
        "classification_threshold": 0.50
    })
    
    # Train separate models for Executive and Legislative
    train_exec = df_train[df_train["is_presidential"] == 1]
    train_leg = df_train[df_train["is_presidential"] == 0]
    
    ex_xgb = None
    ex_cb = None
    if not train_exec.empty:
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
        
    leg_xgb = None
    leg_cb = None
    if not train_leg.empty:
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

    if conn is None:
        conn = get_connection()
        
    cursor = conn.cursor()
    cursor.execute("SELECT country_code, region FROM countries;")
    region_cache = dict(cursor.fetchall())
    
    weights_cache = load_weights_cache(cursor)

    # Sort unknowns chronologically for cascading
    df_unknown = df_unknown.sort_values(by=["year", "country_code"]).reset_index(drop=True)
    unknown_years = sorted(df_unknown["year"].unique())
    
    df["is_predicted"] = 0
    df["prediction_confidence"] = 1.0
    
    df["predicted_outcome_val"] = np.nan
    df["final_confidence_val"] = np.nan
    df["raw_confidence_val"] = np.nan
    df["raw_prob_val"] = np.nan
    
    print(f"Starting cascading predictions loop for {len(unknown_years)} target years: {unknown_years}")
    for idx_yr, yr in enumerate(unknown_years):
        print(f"  [Cascading] Processing target year {yr} ({idx_yr + 1}/{len(unknown_years)})...")
        # First, update the features for all elections in this year using prior outcomes
        update_dynamic_features_for_year(df, yr, weights_cache, region_cache)
        
        # Identify elections in this year that need prediction
        year_elections = df[(df["year"] == yr) & df["target_outcome"].isna()]
        
        if year_elections.empty:
            continue
            
        for idx, row in year_elections.iterrows():
            code = row["country_code"]
            completeness = row["data_completeness"]
            is_pres = row["is_presidential"]
            
            # Predict using relevant model
            if is_pres == 1:
                # Executive model
                X_pred = pd.DataFrame([row[exec_features]], columns=exec_features)
                prob_xgb = ex_xgb.predict_proba(X_pred)[:, 1][0] if ex_xgb is not None else 0.5
                prob_cb = ex_cb.predict_proba(X_pred)[:, 1][0] if ex_cb is not None else 0.5
                xgb_w = exec_params["xgb_weight"]
                T_val = exec_params["T"]
                threshold = exec_params["classification_threshold"]
            else:
                # Legislative model
                X_pred = pd.DataFrame([row[leg_features]], columns=leg_features)
                prob_xgb = leg_xgb.predict_proba(X_pred)[:, 1][0] if leg_xgb is not None else 0.5
                prob_cb = leg_cb.predict_proba(X_pred)[:, 1][0] if leg_cb is not None else 0.5
                xgb_w = leg_params["xgb_weight"]
                T_val = leg_params["T"]
                threshold = leg_params["classification_threshold"]
                
            prob = xgb_w * prob_xgb + (1.0 - xgb_w) * prob_cb
            prob = calibrate_probability(prob, T=T_val)
            raw_confidence = max(prob, 1 - prob)
            
            # Set target_outcome to the soft probability outcome for future steps
            df.loc[idx, "target_outcome"] = prob
            df.loc[idx, "is_predicted"] = 1
            df.loc[idx, "prediction_confidence"] = raw_confidence
            
            # Calculate Feed-Forward Quality Penalty
            # 1. Self prior penalty
            prior_self = df[(df["country_code"] == code) & (df["year"] < yr)].sort_values(by="year", ascending=False)
            predicted_self_frac = 0.0
            if not prior_self.empty:
                if prior_self.iloc[0]["is_predicted"] == 1:
                    predicted_self_frac = 1.0
            
            # 2. Neighbor prior penalty (spatial lags)
            predicted_spatial_weight = 0.0
            total_spatial_weight = 0.0
            other_elections = df[(df["year"] < yr) & (df["year"] >= yr - 4) & (df["country_code"] != code)]
            
            for wt in ["trade", "alliance"]:
                src_weights = weights_cache.get(wt, {}).get(yr, {}).get(code, {})
                if src_weights:
                    matching_others = other_elections[other_elections["country_code"].isin(src_weights)]
                    for _, other in matching_others.iterrows():
                        w = src_weights[other["country_code"]]
                        total_spatial_weight += w
                        if other["is_predicted"] == 1:
                            predicted_spatial_weight += w
            predicted_spatial_frac = (predicted_spatial_weight / total_spatial_weight) if total_spatial_weight > 0 else 0.0
            
            # 3. Region prior penalty (regional tide in last 36 months)
            region = region_cache.get(code, "Unknown")
            prior_region = df[(df["region"] == region) & (df["year"] < yr) & (df["year"] >= yr - 3) & df["target_outcome"].notna()]
            if not prior_region.empty:
                predicted_region_frac = len(prior_region[prior_region["is_predicted"] == 1]) / len(prior_region)
            else:
                predicted_region_frac = 0.0
                
            ff_penalty = 1.0 - 0.10 * predicted_self_frac - 0.10 * predicted_spatial_frac - 0.05 * predicted_region_frac
            ff_penalty = max(0.5, min(1.0, ff_penalty))
            
            # Apply Feed-Forward quality penalty to data completeness
            completeness = row["data_completeness"] * ff_penalty
            df.loc[idx, "data_completeness"] = completeness
            
            # Calculate Localized Cascade Penalty for raw confidence adjustment
            predicted_weight_sum = 0.0
            total_weight_sum = 0.0
            for wt in ["trade", "alliance"]:
                src_weights = weights_cache.get(wt, {}).get(yr, {}).get(code, {})
                if src_weights:
                    matching_others = other_elections[other_elections["country_code"].isin(src_weights)]
                    for _, other in matching_others.iterrows():
                        tgt = other["country_code"]
                        w = src_weights[tgt]
                        total_weight_sum += w
                        if other["is_predicted"] == 1:
                            conf = other["prediction_confidence"]
                            predicted_weight_sum += w * (1.0 - conf)
            
            if total_weight_sum > 0:
                avg_unreliability = predicted_weight_sum / total_weight_sum
                cascade_penalty = 1.0 - 0.2 * avg_unreliability
            else:
                cascade_penalty = 1.0
                
            # Apply confidence penalty using probability shrinkage towards 0.5
            # This guarantees adjusted_confidence >= 50%
            K = completeness * cascade_penalty
            adjusted_confidence = 0.5 + (raw_confidence - 0.5) * K
            predicted_outcome = 1 if prob >= threshold else 0
            
            df.loc[idx, "predicted_outcome_val"] = predicted_outcome
            df.loc[idx, "final_confidence_val"] = adjusted_confidence
            df.loc[idx, "raw_confidence_val"] = raw_confidence
            df.loc[idx, "raw_prob_val"] = prob
            
        # Recalculate global features and PCA after predicting each target year
        recalculate_global_features(df)
            
    results = []
    df_pred_set = df[df["election_id"].isin(df_unknown["election_id"])].sort_values(by=["year", "country_code"])
    
    for idx, row in df_pred_set.iterrows():
        prob = row["raw_prob_val"]
        predicted_outcome = int(row["predicted_outcome_val"])
        raw_confidence = row["raw_confidence_val"]
        final_confidence = row["final_confidence_val"]
        
        orig_row = df_unknown[df_unknown["election_id"] == row["election_id"]].iloc[0]
        target_val = orig_row["target_outcome"]
        
        results.append({
            "election_id": row["election_id"],
            "source": row["source"],
            "country_code": row["country_code"],
            "region": row["region"],
            "year": row["year"],
            "election_type": "Executive" if row["is_presidential"] == 1 else "Legislative",
            "is_scheduled": row["is_scheduled"],
            "clean_index": row["clean_index"],
            "gdp_growth": row.get("gdp_growth"),
            "raw_probability": prob,
            "predicted_outcome": predicted_outcome,
            "predicted_winner": "Incumbent" if predicted_outcome == 1 else "Challenger",
            "raw_confidence": raw_confidence,
            "data_completeness": row["data_completeness"],
            "adjusted_confidence": final_confidence,
            "data_source_flags": row["data_source_flags"],
            "target_outcome": target_val,
            "election_date": row.get("election_date")
        })
        
    if return_df:
        return pd.DataFrame(results), df
    return pd.DataFrame(results)


# 6. Feature Importance Report
def get_feature_importance_report(df, model_type="both"):
    """Train on known data and return ranked feature importances for a model type."""
    df_known = df[df["target_outcome"].notna()].copy()
    if df_known.empty:
        return pd.DataFrame()
        
    exec_features = [f for f in FEATURES if f not in LEG_SPECIFIC]
    leg_features = [f for f in FEATURES if f not in EXEC_SPECIFIC]
    
    if model_type == "Executive":
        train_df = df_known[df_known["is_presidential"] == 1]
        feats = exec_features
    elif model_type == "Legislative":
        train_df = df_known[df_known["is_presidential"] == 0]
        feats = leg_features
    else:
        train_df = df_known
        feats = FEATURES
        
    if train_df.empty:
        return pd.DataFrame()
        
    if model_type in ["Executive", "Legislative"]:
        X = train_df[feats]
        y = train_df["target_outcome"]
        model = xgb.XGBClassifier(
            max_depth=3,
            learning_rate=0.03,
            n_estimators=150,
            eval_metric='logloss',
            random_state=42
        ).fit(X, y)
        
        importance_df = pd.DataFrame({
            "feature": feats,
            "importance": model.feature_importances_
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        return importance_df
    else:
        # Fit both and average
        train_ex = df_known[df_known["is_presidential"] == 1]
        train_lg = df_known[df_known["is_presidential"] == 0]
        
        imp_dict = {f: 0.0 for f in FEATURES}
        counts = {f: 0 for f in FEATURES}
        
        if not train_ex.empty:
            m_ex = xgb.XGBClassifier(max_depth=3, learning_rate=0.03, n_estimators=150, eval_metric='logloss', random_state=42).fit(train_ex[exec_features], train_ex["target_outcome"])
            for f, imp in zip(exec_features, m_ex.feature_importances_):
                imp_dict[f] += imp
                counts[f] += 1
                
        if not train_lg.empty:
            m_lg = xgb.XGBClassifier(max_depth=3, learning_rate=0.03, n_estimators=150, eval_metric='logloss', random_state=42).fit(train_lg[leg_features], train_lg["target_outcome"])
            for f, imp in zip(leg_features, m_lg.feature_importances_):
                imp_dict[f] += imp
                counts[f] += 1
                
        # Average importances where features were seen
        final_imps = []
        for f in FEATURES:
            if counts[f] > 0:
                final_imps.append(imp_dict[f] / counts[f])
            else:
                final_imps.append(0.0)
                
        importance_df = pd.DataFrame({
            "feature": FEATURES,
            "importance": final_imps
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        return importance_df


def main():
    conn = get_connection()
    print("Building optimized modeling dataset from SQLite...")
    df = build_modeling_dataset(conn, include_upcoming=True)
    print(f"Dataset compiled. Rows: {len(df)} (NELDA: {len(df[df['source']=='nelda'])}, Upcoming: {len(df[df['source']=='upcoming'])})")
    
    # Show data completeness stats
    print(f"\nData Completeness Stats:")
    print(f"  Mean: {df['data_completeness'].mean():.2%}")
    print(f"  Min:  {df['data_completeness'].min():.2%}")
    print(f"  Max:  {df['data_completeness'].max():.2%}")
    
    # Compute Moran's I on trade networks
    compute_moran_metrics(df, "trade", 2020)
    
    # Evaluate predictive models (on known-outcome data)
    evaluate_predictive_models(df)
    
    # Feature importance
    print("\n--- Feature Importance Report ---")
    imp_df = get_feature_importance_report(df)
    for _, row in imp_df.head(10).iterrows():
        print(f"  {row['feature']:40s} {row['importance']:.4f}")
    
    # Cascading predictions for upcoming elections
    df_unknown = df[df["target_outcome"].isna()]
    if not df_unknown.empty:
        print(f"\n--- Cascading Predictions for {len(df_unknown)} Upcoming Elections ---")
        preds = predict_with_cascading(df)
        if not preds.empty:
            for _, p in preds.iterrows():
                comp_icon = "🟢" if p["data_completeness"] > 0.8 else ("🟡" if p["data_completeness"] > 0.5 else "🔴")
                print(f"  {comp_icon} {p['country_code']} {p['year']} {p['election_type']:13s} -> {p['predicted_winner']:10s} (conf: {p['adjusted_confidence']:.1%}, data: {p['data_completeness']:.0%})")
    
    conn.close()

if __name__ == "__main__":
    main()
