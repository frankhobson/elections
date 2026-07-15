import os
import sys
import sqlite3
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spatial.calc_weights import get_row_normalized_weights
from esda.moran import Moran

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "elections.db")

def test_weights_normalization():
    # Test normalization for all network layers in year 2020
    for wt in ["unga_voting", "trade", "alliance", "contiguity"]:
        w_obj, countries = get_row_normalized_weights(wt, year=2020)
        
        assert len(countries) > 10, f"Expected at least 10 countries, got {len(countries)}."
        
        # Verify row normalization: sum of weights for each country should be approximately 1.0
        for country in countries:
            w_sum = sum(w_obj.weights[country])
            assert np.isclose(w_sum, 1.0), f"Row sum for {country} in '{wt}' is {w_sum}, expected 1.0."

def test_morans_i_calculation():
    # Test that we can calculate Moran's I on V-Dem data for 2020
    w_obj, countries = get_row_normalized_weights("trade", year=2020)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT country_code, clean_elections_index FROM vdem_indicators WHERE year = 2020;")
    res = dict(cursor.fetchall())
    conn.close()
    
    # Filter array to match weights countries
    y = np.array([res.get(c, 0.5) for c in countries])
    
    mi = Moran(y, w_obj)
    
    assert isinstance(mi.I, float), "Moran's I value is not a float."
    assert not np.isnan(mi.I), "Moran's I is NaN."
    print(f"Test Moran's I value: {mi.I:.4f}")
