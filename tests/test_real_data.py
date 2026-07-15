import os
import sys
import sqlite3
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "elections.db")

def get_connection():
    return sqlite3.connect(DB_PATH)

def test_countries_count():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM countries;")
    count = cursor.fetchone()[0]
    conn.close()
    assert count >= 170, f"Expected >= 170 countries, found {count}"

def test_vdem_indicators_range():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT clean_elections_index, polyarchy_index, regime_type FROM vdem_indicators;")
    rows = cursor.fetchall()
    conn.close()
    
    assert len(rows) > 0, "No records found in vdem_indicators."
    for clean, poly, regime in rows:
        assert 0.0 <= clean <= 1.0, f"Clean Elections Index {clean} out of bounds."
        assert 0.0 <= poly <= 1.0, f"Polyarchy Index {poly} out of bounds."
        assert regime in [0, 1, 2, 3], f"Regime Type {regime} is invalid."

def test_vdem_year_coverage():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT MIN(year), MAX(year) FROM vdem_indicators;")
    min_year, max_year = cursor.fetchone()
    conn.close()
    assert min_year == 1990, f"Expected min year 1990, found {min_year}"
    assert max_year == 2025, f"Expected max year 2025, found {max_year}"

def test_elections_count():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM elections;")
    count = cursor.fetchone()[0]
    conn.close()
    assert count >= 1500, f"Expected >= 1500 elections, found {count}"

def test_nelda_indicators_values():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT nelda11_fraud_allegations, nelda15_opposition_boycott, nelda23_incumbent_won, nelda51_monitors_present 
        FROM nelda_indicators;
    """)
    rows = cursor.fetchall()
    conn.close()
    
    for fraud, boycott, incumbent, monitors in rows:
        assert fraud in ["Yes", "No", "N/A"], f"Invalid fraud flag: {fraud}"
        assert boycott in ["Yes", "No", "N/A"], f"Invalid boycott flag: {boycott}"
        assert incumbent in ["Yes", "No", "N/A"], f"Invalid incumbent flag: {incumbent}"
        assert monitors in ["Yes", "No", "N/A"], f"Invalid monitors flag: {monitors}"

def test_nelda_incumbent_won_coverage():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM nelda_indicators WHERE nelda23_incumbent_won != 'N/A';")
    non_na_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM nelda_indicators;")
    total_count = cursor.fetchone()[0]
    conn.close()
    
    coverage = non_na_count / total_count
    assert coverage >= 0.70, f"Expected >= 70% incumbent won coverage, found {coverage:.2%}"

def test_spatial_weights_types():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT weight_type FROM spatial_weights;")
    types = {row[0] for row in cursor.fetchall()}
    conn.close()
    
    expected_types = {"trade", "unga_voting", "alliance", "contiguity"}
    assert expected_types.issubset(types), f"Missing weight types. Found: {types}"

def test_spatial_weights_bounds():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT weight_value FROM spatial_weights;")
    weights = [row[0] for row in cursor.fetchall()]
    conn.close()
    
    assert len(weights) > 0, "No weights found."
    for w in weights:
        assert 0.0 <= w <= 1.0, f"Weight value {w} out of bounds [0, 1]."

def test_spatial_weights_self_loops():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM spatial_weights WHERE country_code_source = country_code_target;")
    count = cursor.fetchone()[0]
    conn.close()
    assert count == 0, f"Found {count} self loops in spatial_weights."

def test_macro_indicators_types():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT gdp_growth, inflation, gdp_growth_deviation, inflation_deviation FROM macro_indicators;")
    rows = cursor.fetchall()
    conn.close()
    
    assert len(rows) > 0, "No macroeconomic records found."
    for gdp, inf, gdp_dev, inf_dev in rows:
        assert isinstance(gdp, float) or gdp is None, f"GDP growth is {type(gdp)} instead of float or None."
        assert isinstance(inf, float) or inf is None, f"Inflation is {type(inf)} instead of float or None."
        assert isinstance(gdp_dev, float) or gdp_dev is None, f"GDP deviation is {type(gdp_dev)} instead of float or None."
        assert isinstance(inf_dev, float) or inf_dev is None, f"Inflation deviation is {type(inf_dev)} instead of float or None."

def test_gdelt_events_real():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM gdelt_events;")
    count = cursor.fetchone()[0]
    
    cursor.execute("SELECT event_id, country_code, event_date, cameo_code, goldstein_scale, avg_tone, news_volume FROM gdelt_events LIMIT 10;")
    rows = cursor.fetchall()
    conn.close()
    
    assert count > 0, "gdelt_events table is empty. GDELT Ingestion failed."
    for ev_id, ccode, date_str, cameo, goldstein, tone, vol in rows:
        assert isinstance(ev_id, str) and ev_id != "", "Invalid event ID."
        assert len(ccode) == 3, f"Expected ISO-3 country code, got {ccode}."
        assert len(date_str) == 10 and date_str.count("-") == 2, f"Invalid date format: {date_str}."
        assert isinstance(cameo, str) and cameo != "", "Invalid CAMEO event code."
        assert isinstance(goldstein, float), "Goldstein scale must be float."
        assert isinstance(tone, float), "Average tone must be float."
        assert isinstance(vol, int), "News volume must be integer."
