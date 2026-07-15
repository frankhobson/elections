import os
import sqlite3
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "elections.db")
OIL_PATH = os.path.join(BASE_DIR, "raw_data", "brent_oil_prices.csv")
FED_PATH = os.path.join(BASE_DIR, "raw_data", "fed_funds_rate.csv")
WDI_PATH = os.path.join(BASE_DIR, "raw_data", "WDICSV.csv")

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def ingest_global():
    print("Starting Ingestion of Global Geopolitical Anchors...")
    
    # 1. Process Brent Crude Oil prices
    if not os.path.exists(OIL_PATH):
        print(f"Error: Brent Oil CSV not found at {OIL_PATH}")
        return
    df_oil = pd.read_csv(OIL_PATH)
    df_oil["year"] = pd.to_datetime(df_oil["observation_date"], errors='coerce').dt.year
    df_oil["price"] = pd.to_numeric(df_oil["DCOILBRENTEU"], errors='coerce')
    df_oil = df_oil.dropna()
    oil_avg = df_oil.groupby("year")["price"].mean().to_dict()
    print(f"Loaded Brent oil prices for {len(oil_avg)} years.")

    # 2. Process Federal Funds Effective Rate
    if not os.path.exists(FED_PATH):
        print(f"Error: Fed Funds CSV not found at {FED_PATH}")
        return
    df_fed = pd.read_csv(FED_PATH)
    df_fed["year"] = pd.to_datetime(df_fed["observation_date"], errors='coerce').dt.year
    df_fed["rate"] = pd.to_numeric(df_fed["FEDFUNDS"], errors='coerce')
    df_fed = df_fed.dropna()
    fed_avg = df_fed.groupby("year")["rate"].mean().to_dict()
    print(f"Loaded Fed funds rate for {len(fed_avg)} years.")

    # 3. Process US Presidency Ideology (Democrat=1, Republican=0)
    pres_dict = {}
    for yr in range(1990, 2029):
        # Clinton: 1993-2000, Obama: 2009-2016, Biden: 2021-2024
        if (1993 <= yr <= 2000) or (2009 <= yr <= 2016) or (2021 <= yr <= 2024):
            pres_dict[yr] = 1
        else:
            pres_dict[yr] = 0

    # 4. Insert Global Indicators into database
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM global_indicators;")
    
    global_rows = []
    # Loop up to 2028 (with forward-filling for future projections)
    for yr in range(1990, 2029):
        oil_price = oil_avg.get(yr)
        if oil_price is None:
            # Forward-fill projections for 2026-2028 if missing
            oil_price = oil_avg.get(2025, 75.0)
            
        fed_rate = fed_avg.get(yr)
        if fed_rate is None:
            # Forward-fill projections for 2026-2028
            fed_rate = fed_avg.get(2025, 4.5)
            
        pres_ideology = pres_dict[yr]
        global_rows.append((yr, oil_price, fed_rate, pres_ideology))
        
    print(f"Inserting {len(global_rows)} global indicator rows...")
    cursor.executemany("""
        INSERT INTO global_indicators (year, brent_oil_price, fed_funds_rate, us_president_ideology)
        VALUES (?, ?, ?, ?);
    """, global_rows)
    conn.commit()

    # 5. Determine Net Oil Exporter countries
    print("Classifying countries as net oil exporters using WDI and fallback lists...")
    fallback_oil_exporters = {
        "SAU", "RUS", "IRQ", "CAN", "ARE", "KWT", "NOR", "NGA", "AGO", "KAZ", 
        "OMN", "LBY", "VEN", "QAT", "DZA", "AZE", "COL", "ECU", "IRN", "IRQ", "USA"
    }
    
    wdi_exporters = set()
    if os.path.exists(WDI_PATH):
        try:
            # Read in chunks or specific rows to preserve memory
            # We filter by Indicator Code 'TX.VAL.FUEL.ZS.UN'
            chunks = pd.read_csv(WDI_PATH, chunksize=50000)
            fuel_rows = []
            for chunk in chunks:
                filtered = chunk[chunk["Indicator Code"] == "TX.VAL.FUEL.ZS.UN"]
                if not filtered.empty:
                    fuel_rows.append(filtered)
            if fuel_rows:
                df_fuel = pd.concat(fuel_rows, ignore_index=True)
                yr_cols = [str(y) for y in range(1990, 2026) if str(y) in df_fuel.columns]
                df_fuel["avg_fuel"] = df_fuel[yr_cols].apply(pd.to_numeric, errors='coerce').mean(axis=1)
                wdi_exporters = set(df_fuel[df_fuel["avg_fuel"] > 15.0]["Country Code"].unique())
                print(f"Identified {len(wdi_exporters)} oil exporters from WDI (>15% fuel share of exports).")
        except Exception as e:
            print(f"Failed to parse WDI CSV for oil exporters: {e}")
            
    all_oil_exporters = wdi_exporters.union(fallback_oil_exporters)
    
    # Update Countries Table
    cursor.execute("UPDATE countries SET is_oil_exporter = 0;") # Reset
    
    # Fetch valid countries
    cursor.execute("SELECT country_code FROM countries;")
    db_countries = {r[0] for r in cursor.fetchall()}
    
    matching_exporters = [c for c in all_oil_exporters if c in db_countries]
    print(f"Marking {len(matching_exporters)} active database countries as net oil exporters...")
    
    for ccode in matching_exporters:
        cursor.execute("UPDATE countries SET is_oil_exporter = 1 WHERE country_code = ?;", (ccode,))
        
    conn.commit()
    
    # Verification
    cursor.execute("SELECT COUNT(*) FROM global_indicators;")
    g_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM countries WHERE is_oil_exporter = 1;")
    exp_count = cursor.fetchone()[0]
    
    print(f"Global Ingestion successful! global_indicators={g_count} rows, is_oil_exporter={exp_count} countries.")
    conn.close()

if __name__ == "__main__":
    ingest_global()
