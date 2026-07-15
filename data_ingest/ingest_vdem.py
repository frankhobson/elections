import os
import sqlite3
import pandas as pd
import numpy as np
import geopandas as gpd
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "elections.db")
RAW_DIR = os.path.join(BASE_DIR, "raw_data")
VDEM_RAW_PATH = os.path.join(RAW_DIR, "V-Dem-CY-Core-v16.csv")
GADM_PATH = os.path.join(RAW_DIR, "gadm_410-levels.gpkg")

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def get_region_from_coords(lat, lon, country_code):
    # Manual overrides for specific countries
    if country_code in ["TUR", "CYP", "GEO", "ARM", "AZE", "ISR", "PSE", "SAU", "YEM", "OMN", "ARE", "QAT", "KWT", "BHR", "IRQ", "JOR", "LBN", "SYR", "YMD"]:
        return "Middle East"
    if country_code in ["EGY", "LBY", "TUN", "DZA", "MAR", "SML", "ZZB"]:
        return "Africa"
    if country_code in ["HKG"]:
        return "Asia"
    if country_code in ["XKX"]:
        return "Europe"
    
    if -25 <= lon <= 45 and 34 <= lat <= 75:
        return "Europe"
    if -170 <= lon <= -30 and lat > 12:
        return "North America"
    if -100 <= lon <= -30 and lat <= 12:
        return "South America"
    if -20 <= lon <= 55 and lat <= 35:
        return "Africa"
    if 30 <= lon <= 180 and lat > -10:
        return "Asia"
    if 100 <= lon <= 180 and lat <= -10:
        return "Oceania"
    return "Europe" # fallback

def load_or_generate_vdem():
    conn = get_connection()
    cursor = conn.cursor()

    if os.path.exists(VDEM_RAW_PATH):
        print(f"Loading V-Dem data from raw file: {VDEM_RAW_PATH}")
        # Load V-Dem data
        df = pd.read_csv(VDEM_RAW_PATH, usecols=[
            "country_name", "country_text_id", "COWcode", "year",
            "v2xel_frefair", "v2x_polyarchy", "v2x_regime",
            "v2xps_party", "v2eltype_0", "v2eltype_1",
            "v2cafres_mean", "v2casurv_mean", "v2lgbicam",
            "v2exhoshog", "v2xlg_legcon", "v2x_jucon",
            "v2x_execorr", "v2x_pubcorr"
        ])
        
        # Filter years 1990-2025
        df = df[(df["year"] >= 1990) & (df["year"] <= 2025)]
        
        # Build COW code to country_code mapping
        mapping_df = df[["COWcode", "country_text_id"]].dropna().drop_duplicates()
        mapping_dict = {int(row["COWcode"]): row["country_text_id"] for _, row in mapping_df.iterrows()}
        
        # Save to json file for other ingest scripts
        mapping_path = os.path.join(BASE_DIR, "data_ingest", "cow_to_iso.json")
        with open(mapping_path, "w") as f:
            json.dump(mapping_dict, f, indent=4)
        print(f"Saved {len(mapping_dict)} COW to ISO mappings to {mapping_path}")
        
        # Rename columns to match db schema
        df = df.rename(columns={
            "country_text_id": "country_code",
            "v2xel_frefair": "clean_elections_index",
            "v2x_polyarchy": "polyarchy_index",
            "v2x_regime": "regime_type",
            "v2xps_party": "political_polarization",
            "v2eltype_0": "electoral_system_majoritarian",
            "v2eltype_1": "electoral_system_pr",
            "v2cafres_mean": "v2cafres",
            "v2casurv_mean": "v2casurv",
            "v2lgbicam": "v2lgbicam",
            "v2exhoshog": "v2exhoshog",
            "v2xlg_legcon": "v2xlg_legcon",
            "v2x_jucon": "v2x_jucon",
            "v2x_execorr": "v2x_execorr",
            "v2x_pubcorr": "v2x_pubcorr"
        })
        
        # Fill missing values for baseline V-Dem columns
        # Turkmenistan 1990 has null polyarchy and regime type
        df.loc[df["country_code"] == "TKM", "regime_type"] = df.loc[df["country_code"] == "TKM", "regime_type"].fillna(0)
        df.loc[df["country_code"] == "TKM", "polyarchy_index"] = df.loc[df["country_code"] == "TKM", "polyarchy_index"].fillna(0.1)
        df["clean_elections_index"] = df["clean_elections_index"].fillna(0.0)
        df["polyarchy_index"] = df["polyarchy_index"].fillna(0.1)
        df["regime_type"] = df["regime_type"].fillna(0).astype(int)
        
        # Get unique countries in the final dataset
        unique_codes = df["country_code"].unique()
        
        # Load GADM boundaries to get centroids (lats/lons)
        print("Extracting coordinates from GADM geopackage...")
        gdf = gpd.read_file(GADM_PATH, layer="ADM_0")
        gdf["centroid"] = gdf.geometry.centroid
        
        gadm_coords = {}
        for _, row in gdf.iterrows():
            gadm_coords[row["GID_0"]] = (row["centroid"].y, row["centroid"].x)
            
        # Manual overrides/additions for countries not matching GADM codes directly or missing
        manual_coords = {
            "HKG": (22.3, 114.2),
            "SML": (9.6, 44.4),
            "DDR": (52.0, 12.5),
            "XKX": (42.6, 21.1),
            "ZZB": (-6.1, 39.3),
            "PSG": (31.9, 35.2),
            "YMD": (13.8, 47.2)
        }
        
        countries_to_insert = []
        for code in unique_codes:
            # Get real country name from V-Dem
            name = df[df["country_code"] == code]["country_name"].iloc[0]
            # Standardize some V-Dem names
            if name == "United States of America":
                name = "United States"
            elif name == "Burma/Myanmar":
                name = "Myanmar"
                
            lat, lon = manual_coords.get(code, (None, None))
            if lat is None:
                lat, lon = gadm_coords.get(code, (0.0, 0.0))
                
            region = get_region_from_coords(lat, lon, code)
            countries_to_insert.append((code, name, region, lat, lon))
            
        print(f"Inserting {len(countries_to_insert)} countries into the database...")
        cursor.executemany("""
            INSERT OR REPLACE INTO countries (country_code, country_name, region, latitude, longitude)
            VALUES (?, ?, ?, ?, ?);
        """, countries_to_insert)
        
        # Recreate table vdem_indicators to match new schema
        print("Recreating vdem_indicators table with new schema...")
        cursor.execute("DROP TABLE IF EXISTS vdem_indicators;")
        cursor.execute("""
        CREATE TABLE vdem_indicators (
            country_code TEXT NOT NULL,
            year INTEGER NOT NULL,
            clean_elections_index REAL,
            polyarchy_index REAL,
            regime_type INTEGER,
            electoral_system_majoritarian REAL,
            electoral_system_pr REAL,
            political_polarization REAL,
            v2cafres REAL,
            v2casurv REAL,
            v2lgbicam INTEGER,
            v2exhoshog INTEGER,
            v2xlg_legcon REAL,
            v2x_jucon REAL,
            v2x_execorr REAL,
            v2x_pubcorr REAL,
            PRIMARY KEY (country_code, year),
            FOREIGN KEY (country_code) REFERENCES countries(country_code) ON DELETE CASCADE
        );
        """)
        
        # Insert V-Dem Indicators
        vdem_rows = []
        for _, row in df.iterrows():
            def val_or_none(val, val_type=float):
                if pd.isna(val) or val is None:
                    return None
                try:
                    return val_type(val)
                except ValueError:
                    return None

            vdem_rows.append((
                row["country_code"],
                int(row["year"]),
                val_or_none(row["clean_elections_index"]),
                val_or_none(row["polyarchy_index"]),
                val_or_none(row["regime_type"], int) if val_or_none(row["regime_type"], int) is not None else 0,
                val_or_none(row["electoral_system_majoritarian"]),
                val_or_none(row["electoral_system_pr"]),
                val_or_none(row["political_polarization"]),
                val_or_none(row["v2cafres"]),
                val_or_none(row["v2casurv"]),
                val_or_none(row["v2lgbicam"], int),
                val_or_none(row["v2exhoshog"], int),
                val_or_none(row["v2xlg_legcon"]),
                val_or_none(row["v2x_jucon"]),
                val_or_none(row["v2x_execorr"]),
                val_or_none(row["v2x_pubcorr"])
            ))
            
        print(f"Inserting {len(vdem_rows)} country-year V-Dem records...")
        cursor.executemany("""
            INSERT OR REPLACE INTO vdem_indicators (
                country_code, year, clean_elections_index, polyarchy_index, regime_type, 
                electoral_system_majoritarian, electoral_system_pr, political_polarization,
                v2cafres, v2casurv, v2lgbicam, v2exhoshog, v2xlg_legcon, v2x_jucon, v2x_execorr, v2x_pubcorr
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, vdem_rows)
        
    else:
        raise FileNotFoundError(f"Raw V-Dem CSV not found at: {VDEM_RAW_PATH}")

    conn.commit()
    
    cursor.execute("SELECT COUNT(*) FROM countries;")
    c_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM vdem_indicators;")
    v_count = cursor.fetchone()[0]
    print(f"Ingested {c_count} countries and {v_count} country-year V-Dem records successfully.")
    
    conn.close()

if __name__ == "__main__":
    load_or_generate_vdem()
