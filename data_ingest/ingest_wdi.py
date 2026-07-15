import os
import sqlite3
import urllib.request
import json
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "elections.db")

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def fetch_wdi_indicator(indicator_code):
    """Fetches indicator data from the World Bank API."""
    url = f"https://api.worldbank.org/v2/country/all/indicator/{indicator_code}?date=1990:2025&format=json&per_page=20000"
    print(f"Fetching WDI data from: {url}")
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            res_data = json.loads(response.read().decode())
            if len(res_data) > 1 and isinstance(res_data[1], list):
                return res_data[1]
    except Exception as e:
        print(f"World Bank API fetch failed for {indicator_code}: {e}")
    return None

def load_or_generate_wdi():
    conn = get_connection()
    cursor = conn.cursor()
    
    # Get all valid country codes in database
    cursor.execute("SELECT country_code FROM countries;")
    valid_codes = {row[0] for row in cursor.fetchall()}
    
    # Indicators dictionary
    indicators = {
        "gdp_growth": "NY.GDP.MKTP.KD.ZG",
        "inflation": "FP.CPI.TOTL.ZG",
        "unemployment": "SL.UEM.TOTL.ZS",
        "gdp_per_capita_growth": "NY.GDP.PCAP.KD.ZG",
        "net_migration": "SM.POP.NETM",
        "refugee_asylum": "SM.POP.REFG",
        "refugee_origin": "SM.POP.REFG.OR"
    }
    
    data_dicts = {}
    for key, ind_code in indicators.items():
        data = fetch_wdi_indicator(ind_code)
        val_dict = {}
        if data:
            for item in data:
                code = item.get("countryiso3code")
                year = item.get("date")
                val = item.get("value")
                if code and year and val is not None:
                    val_dict[(code, int(year))] = float(val)
        data_dicts[key] = val_dict
        
    records = []
    if data_dicts["gdp_growth"] and data_dicts["inflation"]:
        print("Successfully retrieved baseline World Bank data. Parsing JSON...")
        for code in valid_codes:
            for year in range(1990, 2026):
                has_any = False
                row_dict = {"country_code": code, "year": year}
                for key in indicators.keys():
                    val = data_dicts[key].get((code, year))
                    row_dict[key] = val
                    if val is not None:
                        has_any = True
                if has_any:
                    records.append(row_dict)
    
    # No synthetic fallback — if API failed, warn and exit
    if not records:
        print("WARNING: WDI API unavailable and no data fetched. Macro indicators will be empty.")
        print("The modeling pipeline will treat missing macro data as NaN (incomplete information).")
        conn.close()
        return

    # Convert to DataFrame to calculate 5-year moving average deviations
    df = pd.DataFrame(records)
    
    # Sort by country and year to compute rolling average accurately
    df = df.sort_values(by=["country_code", "year"]).reset_index(drop=True)
    
    # Compute 5-year moving average deviations
    df["gdp_growth_ma5"] = df.groupby("country_code")["gdp_growth"].shift(1).rolling(5, min_periods=1).mean()
    df["inflation_ma5"] = df.groupby("country_code")["inflation"].shift(1).rolling(5, min_periods=1).mean()
    
    df["gdp_growth_deviation"] = df["gdp_growth"] - df["gdp_growth_ma5"].fillna(df["gdp_growth"])
    df["inflation_deviation"] = df["inflation"] - df["inflation_ma5"].fillna(df["inflation"])
    
    df["gdp_growth_deviation"] = df["gdp_growth_deviation"].fillna(0.0)
    df["inflation_deviation"] = df["inflation_deviation"].fillna(0.0)
    
    # Convert back to list of tuples for SQL insertion
    insert_rows = []
    for _, r in df.iterrows():
        def val_or_none(v):
            return float(v) if pd.notna(v) and v is not None else None
            
        insert_rows.append((
            r["country_code"],
            int(r["year"]),
            val_or_none(r.get("gdp_growth")),
            val_or_none(r.get("inflation")),
            val_or_none(r.get("gdp_growth_deviation")),
            val_or_none(r.get("inflation_deviation")),
            val_or_none(r.get("unemployment")),
            val_or_none(r.get("gdp_per_capita_growth")),
            val_or_none(r.get("net_migration")),
            val_or_none(r.get("refugee_asylum")),
            val_or_none(r.get("refugee_origin"))
        ))

    # Insert into Database
    cursor.executemany("""
    INSERT OR REPLACE INTO macro_indicators (
        country_code, year, gdp_growth, inflation, gdp_growth_deviation, inflation_deviation,
        unemployment, gdp_per_capita_growth, net_migration, refugee_asylum, refugee_origin
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """, insert_rows)

    conn.commit()
    
    cursor.execute("SELECT COUNT(*) FROM macro_indicators;")
    m_count = cursor.fetchone()[0]
    print(f"Ingested {m_count} country-year macroeconomic indicator records successfully.")
    
    conn.close()

if __name__ == "__main__":
    load_or_generate_wdi()
