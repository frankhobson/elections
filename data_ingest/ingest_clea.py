import os
import sqlite3
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "elections.db")
CLEA_PATH = os.path.join(BASE_DIR, "raw_data", "clea_lc_20251015.dta")

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def load_country_mapping():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT country_code, country_name FROM countries;")
    countries = cursor.fetchall()
    conn.close()
    
    # Map cleaned lower-case country names in DB to ISO-3 codes
    name_to_iso = {}
    for code, name in countries:
        name_to_iso[name.lower().strip()] = code
        
    # Manual overrides for mismatches between V-Dem/DB names and CLEA names
    manual_overrides = {
        "russian federation": "RUS",
        "russia": "RUS",
        "great britain": "GBR",
        "uk": "GBR",
        "united kingdom": "GBR",
        "korea": "KOR",
        "south korea": "KOR",
        "republic of korea": "KOR",
        "turkey": "TUR",
        "türkiye": "TUR",
        "gambia": "GMB",
        "the gambia": "GMB",
        "czech republic": "CZE",
        "czechia": "CZE",
        "palestine": "PSE",
        "palestine/west bank": "PSE",
        "palestine/gaza": "PSE",
        "congo, democratic republic": "COD",
        "democratic republic of the congo": "COD",
        "congo, republic of": "COG",
        "republic of the congo": "COG",
        "vietnam": "VNM",
        "viet nam": "VNM",
        "laos": "LAO",
        "lao people's democratic republic": "LAO",
        "cote d'ivoire": "CIV",
        "cote divoire": "CIV",
        "ivory coast": "CIV",
    }
    
    for k, v in manual_overrides.items():
        name_to_iso[k] = v
        
    return name_to_iso

def ingest_clea():
    print(f"Reading CLEA lower chamber dataset from: {CLEA_PATH}")
    # Read only required columns to save memory and speed up ingestion
    cols = ["ctr_n", "yr", "cst", "pty", "seat", "pev1", "vv1", "pv1"]
    try:
        df = pd.read_stata(CLEA_PATH, columns=cols)
    except Exception as e:
        print(f"Failed to read CLEA file: {e}")
        return
        
    print(f"Successfully loaded {len(df):,} rows from CLEA.")
    
    # Clean data: treat negative values as NaN (CLEA missing code standard)
    for col in ["seat", "pev1", "vv1", "pv1"]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df.loc[df[col] < 0, col] = np.nan
        
    df["yr"] = pd.to_numeric(df["yr"], errors='coerce')
    df = df.dropna(subset=["ctr_n", "yr"])
    df["yr"] = df["yr"].astype(int)
    
    # Map country names to ISO codes
    name_to_iso = load_country_mapping()
    df["ctr_n_clean"] = df["ctr_n"].str.lower().str.strip()
    df["country_code"] = df["ctr_n_clean"].map(name_to_iso)
    
    # Filter out unmapped countries
    unmapped_names = df[df["country_code"].isna()]["ctr_n"].unique()
    print(f"Skipping {len(unmapped_names)} countries not present in target database countries table.")
    df = df.dropna(subset=["country_code"])
    
    # Group by (country_code, year) to compute election indicators
    grouped = df.groupby(["country_code", "yr"])
    
    country_year_shares = {}
    country_years = {}
    raw_results = []
    
    print("Computing election aggregates...")
    for (ccode, year), group in grouped:
        # 1. Turnout
        # In CLEA, pev1 (registered voters) and vv1 (valid votes) are constituency level
        const_data = group[["cst", "pev1", "vv1"]].drop_duplicates(subset=["cst"])
        electorate = const_data["pev1"].sum()
        valid_votes = const_data["vv1"].sum()
        
        turnout = valid_votes / electorate if (electorate > 0 and valid_votes > 0) else np.nan
        
        # 2. Party seat shares and votes
        party_seats = group.groupby("pty")["seat"].sum().dropna()
        total_seats = party_seats.sum()
        
        shares_dict = {}
        
        # Effective Number of Parties (seats)
        if total_seats > 0:
            seat_shares = party_seats / total_seats
            effective_parties = 1.0 / (seat_shares ** 2).sum()
            
            # Sort seat shares to get margin and largest party
            sorted_shares = seat_shares.sort_values(ascending=False)
            largest_party_seat_share = sorted_shares.iloc[0]
            
            if len(sorted_shares) > 1:
                margin_of_victory = sorted_shares.iloc[0] - sorted_shares.iloc[1]
            else:
                margin_of_victory = sorted_shares.iloc[0]
                
            shares_dict = seat_shares.to_dict()
        else:
            effective_parties = np.nan
            largest_party_seat_share = np.nan
            margin_of_victory = np.nan
            
        # Fallback to votes if seats are missing/0
        if pd.isna(effective_parties) or pd.isna(largest_party_seat_share):
            party_votes = group.groupby("pty")["pv1"].sum().dropna()
            total_votes = party_votes.sum()
            if total_votes > 0:
                vote_shares = party_votes / total_votes
                effective_parties = 1.0 / (vote_shares ** 2).sum()
                sorted_vote_shares = vote_shares.sort_values(ascending=False)
                largest_party_seat_share = sorted_vote_shares.iloc[0]
                if len(sorted_vote_shares) > 1:
                    margin_of_victory = sorted_vote_shares.iloc[0] - sorted_vote_shares.iloc[1]
                else:
                    margin_of_victory = sorted_vote_shares.iloc[0]
                shares_dict = vote_shares.to_dict()
                    
        # Sanity check: replace inf/nan with clean values
        if not pd.isna(effective_parties) and np.isinf(effective_parties):
            effective_parties = np.nan
            
        if ccode not in country_years:
            country_years[ccode] = []
        country_years[ccode].append(int(year))
        
        country_year_shares[(ccode, int(year))] = shares_dict
        
        raw_results.append({
            "ccode": ccode,
            "year": int(year),
            "turnout": turnout,
            "margin_of_victory": margin_of_victory,
            "effective_parties": effective_parties,
            "incumbent_seat_share": largest_party_seat_share
        })
        
    # Calculate Pedersen volatility index chronologically
    pedersen_indices = {}
    for ccode, years in country_years.items():
        sorted_years = sorted(years)
        for i, yr in enumerate(sorted_years):
            if i == 0:
                pedersen_indices[(ccode, yr)] = None
                continue
            prev_yr = sorted_years[i-1]
            shares_t = country_year_shares[(ccode, yr)]
            shares_t_1 = country_year_shares[(ccode, prev_yr)]
            
            if not shares_t or not shares_t_1:
                pedersen_indices[(ccode, yr)] = None
                continue
                
            all_parties = set(shares_t.keys()).union(set(shares_t_1.keys()))
            diff_sum = 0.0
            for p in all_parties:
                diff_sum += abs(shares_t.get(p, 0.0) - shares_t_1.get(p, 0.0))
            pedersen_indices[(ccode, yr)] = 0.5 * diff_sum
            
    clea_indicators = []
    for res in raw_results:
        ccode = res["ccode"]
        yr = res["year"]
        pedersen = pedersen_indices.get((ccode, yr))
        clea_indicators.append((
            ccode,
            yr,
            float(res["turnout"]) if pd.notna(res["turnout"]) else None,
            float(res["margin_of_victory"]) if pd.notna(res["margin_of_victory"]) else None,
            float(res["effective_parties"]) if pd.notna(res["effective_parties"]) else None,
            float(res["incumbent_seat_share"]) if pd.notna(res["incumbent_seat_share"]) else None,
            float(pedersen) if pedersen is not None else None
        ))
        
    print(f"Generated {len(clea_indicators)} country-year indicator records.")
    
    # Insert into Database
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM clea_indicators;") # Clear existing to avoid duplicate conflicts
    
    print("Inserting records into 'clea_indicators' table...")
    cursor.executemany("""
        INSERT INTO clea_indicators (country_code, year, turnout, margin_of_victory, effective_parties, incumbent_seat_share, pedersen_index)
        VALUES (?, ?, ?, ?, ?, ?, ?);
    """, clea_indicators)
    
    conn.commit()
    
    cursor.execute("SELECT COUNT(*) FROM clea_indicators;")
    row_count = cursor.fetchone()[0]
    print(f"Ingested {row_count} CLEA country-year election records successfully.")
    
    conn.close()

if __name__ == "__main__":
    ingest_clea()
