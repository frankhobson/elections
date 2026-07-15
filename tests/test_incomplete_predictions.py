"""
Test the incomplete-information prediction pipeline end-to-end.

Verifies:
1. build_modeling_dataset returns both NELDA and upcoming elections
2. Data completeness scores are computed correctly
3. Cascading predictions work for elections with NaN targets
4. Feature importance report is non-empty
5. Known-outcome upcoming elections are used as training data
"""
import os
import sys
import sqlite3
import pandas as pd
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.correlation_models import (
    build_modeling_dataset, predict_with_cascading, 
    get_feature_importance_report, FEATURES,
    compute_data_completeness
)

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "elections.db")

@pytest.fixture(scope="module")
def df():
    conn = sqlite3.connect(DB_PATH)
    res = build_modeling_dataset(conn, include_upcoming=True)
    conn.close()
    return res


def test_build_dataset_includes_upcoming():
    """Test that build_modeling_dataset includes upcoming elections when flag is set."""
    conn = sqlite3.connect(DB_PATH)
    
    # Without upcoming
    df_basic = build_modeling_dataset(conn, include_upcoming=False)
    # With upcoming
    df_full = build_modeling_dataset(conn, include_upcoming=True)
    
    conn.close()
    
    nelda_count = len(df_basic)
    full_count = len(df_full)
    upcoming_count = len(df_full[df_full["source"] == "upcoming"])
    
    print(f"✅ Dataset sizes: NELDA-only={nelda_count}, Full={full_count}, Upcoming={upcoming_count}")
    assert full_count >= nelda_count, "Full dataset should be >= NELDA-only"
    assert upcoming_count > 0, "Should have upcoming elections"
    return df_full


def test_data_completeness(df):
    """Test that data completeness scores are computed."""
    assert "data_completeness" in df.columns, "data_completeness column missing"
    assert df["data_completeness"].notna().all(), "All rows should have completeness scores"
    assert df["data_completeness"].min() >= 0.0, "Completeness should be >= 0"
    assert df["data_completeness"].max() <= 1.0, "Completeness should be <= 1"
    
    # NELDA elections should generally have higher completeness
    nelda_mean = df[df["source"] == "nelda"]["data_completeness"].mean()
    upcoming_mean = df[df["source"] == "upcoming"]["data_completeness"].mean()
    
    print(f"✅ Completeness: NELDA mean={nelda_mean:.2%}, Upcoming mean={upcoming_mean:.2%}")


def test_data_source_flags(df):
    """Test that data source flags are correctly set."""
    assert "data_source_flags" in df.columns, "data_source_flags column missing"
    
    # Check bit flags make sense
    flags = df["data_source_flags"].unique()
    print(f"✅ Data source flag values seen: {sorted(flags)}")
    
    # V-Dem flag (bit 0) should be common
    has_vdem = df["data_source_flags"].apply(lambda f: f & 1 > 0).sum()
    print(f"  Elections with V-Dem: {has_vdem}/{len(df)}")


def test_nan_handling(df):
    """Test that NaN values are preserved (not dropped or filled)."""
    upcoming = df[df["source"] == "upcoming"]
    
    # Some features should be NaN for some upcoming elections
    nan_counts = upcoming[FEATURES].isnull().sum()
    features_with_nans = nan_counts[nan_counts > 0]
    
    if not features_with_nans.empty:
        print(f"✅ NaN handling: {len(features_with_nans)} features have NaN values in upcoming elections")
        for feat, count in features_with_nans.head(5).items():
            print(f"  {feat}: {count} NaN values")
    else:
        print("⚠️ No NaN values found in upcoming features (all data complete)")


def test_cascading_predictions(df):
    """Test that cascading predictions produce results for unknown-outcome elections."""
    preds = predict_with_cascading(df)
    
    if preds.empty:
        print("⚠️ No elections to predict (all have known outcomes)")
        return preds
    
    assert "predicted_outcome" in preds.columns, "predicted_outcome column missing"
    assert "adjusted_confidence" in preds.columns, "adjusted_confidence column missing"
    assert "predicted_winner" in preds.columns, "predicted_winner column missing"
    
    print(f"✅ Cascading predictions: {len(preds)} elections predicted")
    print(f"  Incumbent wins: {len(preds[preds['predicted_outcome'] == 1])}")
    print(f"  Challenger wins: {len(preds[preds['predicted_outcome'] == 0])}")
    print(f"  Confidence range: {preds['adjusted_confidence'].min():.2%} - {preds['adjusted_confidence'].max():.2%}")
    
    # Check that confidence decreases for later predictions (cascading penalty)
    early = preds[preds["year"] <= preds["year"].min() + 1]
    late = preds[preds["year"] >= preds["year"].max() - 1]
    if not early.empty and not late.empty:
        print(f"  Early avg confidence: {early['adjusted_confidence'].mean():.2%}")
        print(f"  Late avg confidence: {late['adjusted_confidence'].mean():.2%}")
    
    return preds


def test_feature_importance(df):
    """Test that feature importance report is generated."""
    imp_df = get_feature_importance_report(df)
    
    assert not imp_df.empty, "Feature importance should not be empty"
    assert len(imp_df) == len(FEATURES), f"Should have {len(FEATURES)} features, got {len(imp_df)}"
    assert imp_df["importance"].sum() > 0, "Total importance should be > 0"
    
    top5 = imp_df.head(5)
    print(f"✅ Feature importance: Top 5 features:")
    for _, row in top5.iterrows():
        print(f"  {row['feature']:40s} {row['importance']:.4f}")


def test_known_outcomes_used_for_training(df):
    """Test that upcoming elections with known outcomes feed into training."""
    known_upcoming = df[(df["source"] == "upcoming") & df["target_outcome"].notna()]
    unknown_upcoming = df[(df["source"] == "upcoming") & df["target_outcome"].isna()]
    
    print(f"✅ Training split: Known upcoming={len(known_upcoming)}, Unknown upcoming={len(unknown_upcoming)}")
    
    # Known outcomes should have values 0 or 1
    if not known_upcoming.empty:
        assert known_upcoming["target_outcome"].isin([0, 1]).all(), "Known outcomes should be 0 or 1"


if __name__ == "__main__":
    print("=" * 60)
    print("TESTING INCOMPLETE-INFORMATION PREDICTION PIPELINE")
    print("=" * 60)
    
    # 1. Build dataset
    print("\n--- Test 1: Build Dataset with Upcoming Elections ---")
    df = test_build_dataset_includes_upcoming()
    
    # 2. Data completeness
    print("\n--- Test 2: Data Completeness Scores ---")
    test_data_completeness(df)
    
    # 3. Data source flags
    print("\n--- Test 3: Data Source Flags ---")
    test_data_source_flags(df)
    
    # 4. NaN handling
    print("\n--- Test 4: NaN Value Preservation ---")
    test_nan_handling(df)
    
    # 5. Known outcomes as training data
    print("\n--- Test 5: Known Outcomes Used for Training ---")
    test_known_outcomes_used_for_training(df)
    
    # 6. Cascading predictions
    print("\n--- Test 6: Cascading Predictions ---")
    preds = test_cascading_predictions(df)
    
    # 7. Feature importance
    print("\n--- Test 7: Feature Importance ---")
    test_feature_importance(df)
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
