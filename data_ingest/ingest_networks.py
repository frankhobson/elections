import os
import sqlite3
import pandas as pd
import numpy as np
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "elections.db")
RAW_DIR = os.path.join(BASE_DIR, "raw_data")

# Raw data paths
CONTIGUITY_PATH = os.path.join(RAW_DIR, "contdird.csv")
ALLIANCE_PATH = os.path.join(RAW_DIR, "alliance_v4.1_by_directed_yearly.csv")
TRADE_PATH = os.path.join(RAW_DIR, "Dyadic_COW_4.0.csv")
UNGA_PATH = os.path.join(RAW_DIR, "2026_02_06_ga_voting.csv")

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def load_or_generate_networks():
    conn = get_connection()
    cursor = conn.cursor()

    # Load valid country codes
    cursor.execute("SELECT country_code FROM countries;")
    valid_codes = {row[0] for row in cursor.fetchall()}
    
    # Load COW-to-ISO3 mapping
    mapping_path = os.path.join(BASE_DIR, "data_ingest", "cow_to_iso.json")
    if os.path.exists(mapping_path):
        with open(mapping_path) as f:
            cow_to_iso = {int(k): v for k, v in json.load(f).items()}
    else:
        cow_to_iso = {}

    # Apply manual overrides for small states or historical codes
    manual_overrides = {
        31: "BHS", 54: "DMA", 55: "GRD", 56: "LCA", 57: "VCT", 58: "ATG", 60: "KNA",
        80: "BLZ", 221: "MCO", 223: "LIE", 232: "AND", 325: "SMR", 340: "SRB", 345: "SRB",
        940: "NRU", 946: "KIR", 947: "TUV", 955: "TON", 983: "MHL", 986: "PLW",
        987: "FSM", 990: "WSM", 835: "BRN"
    }
    for k, v in manual_overrides.items():
        cow_to_iso[k] = v

    # Clear existing spatial weights
    cursor.execute("DELETE FROM spatial_weights;")
    conn.commit()

    # 1. Contiguity
    if os.path.exists(CONTIGUITY_PATH):
        print(f"Loading contiguity data from: {CONTIGUITY_PATH}")
        df_cont = pd.read_csv(CONTIGUITY_PATH)
        df_cont = df_cont[(df_cont["year"] >= 1990) & (df_cont["year"] <= 2028)]
        
        # Forward-fill to 2028 if latest year is earlier
        max_year = df_cont["year"].max()
        if max_year < 2028:
            print(f"Forward-filling contiguity from {max_year} to 2028...")
            df_latest = df_cont[df_cont["year"] == max_year]
            extra_dfs = []
            for yr in range(max_year + 1, 2029):
                df_yr = df_latest.copy()
                df_yr["year"] = yr
                extra_dfs.append(df_yr)
            df_cont = pd.concat([df_cont] + extra_dfs, ignore_index=True)
            
        CONTIGUITY_WEIGHTS = {1: 1.0, 2: 1.0, 3: 0.6, 4: 0.4, 5: 0.2}
        
        cont_rows = []
        for _, row in df_cont.iterrows():
            src_cow = int(row["state1no"])
            tgt_cow = int(row["state2no"])
            src_iso = cow_to_iso.get(src_cow)
            tgt_iso = cow_to_iso.get(tgt_cow)
            
            if src_iso and tgt_iso and src_iso in valid_codes and tgt_iso in valid_codes:
                if src_iso == tgt_iso:
                    continue
                conttype = int(row["conttype"])
                weight = CONTIGUITY_WEIGHTS.get(conttype, 0.0)
                if weight > 0.0:
                    cont_rows.append((src_iso, tgt_iso, "contiguity", int(row["year"]), weight))
                    
        print(f"Inserting {len(cont_rows)} contiguity weights...")
        cursor.executemany("""
            INSERT OR REPLACE INTO spatial_weights (country_code_source, country_code_target, weight_type, year, weight_value)
            VALUES (?, ?, ?, ?, ?);
        """, cont_rows)
        conn.commit()
    else:
        print("Warning: Contiguity raw file not found.")

    # 2. Alliances
    if os.path.exists(ALLIANCE_PATH):
        print(f"Loading alliances data from: {ALLIANCE_PATH}")
        df_all = pd.read_csv(ALLIANCE_PATH)
        df_all = df_all[(df_all["year"] >= 1990) & (df_all["year"] <= 2028)]
        
        # Forward-fill to 2028
        max_year = df_all["year"].max()
        if max_year < 2028:
            print(f"Forward-filling alliances from {max_year} to 2028...")
            df_latest = df_all[df_all["year"] == max_year]
            extra_dfs = []
            for yr in range(max_year + 1, 2029):
                df_yr = df_latest.copy()
                df_yr["year"] = yr
                extra_dfs.append(df_yr)
            df_all = pd.concat([df_all] + extra_dfs, ignore_index=True)
            
        all_rows = []
        for _, row in df_all.iterrows():
            src_cow = int(row["ccode1"])
            tgt_cow = int(row["ccode2"])
            src_iso = cow_to_iso.get(src_cow)
            tgt_iso = cow_to_iso.get(tgt_cow)
            
            if src_iso and tgt_iso and src_iso in valid_codes and tgt_iso in valid_codes:
                if src_iso == tgt_iso:
                    continue
                # Calculate alliance strength
                def_val = float(row["defense"]) if not pd.isna(row["defense"]) else 0.0
                neu_val = float(row["neutrality"]) if not pd.isna(row["neutrality"]) else 0.0
                non_val = float(row["nonaggression"]) if not pd.isna(row["nonaggression"]) else 0.0
                ent_val = float(row["entente"]) if not pd.isna(row["entente"]) else 0.0
                
                weight = max(
                    1.0 * def_val,
                    0.5 * neu_val,
                    0.5 * non_val,
                    0.3 * ent_val
                )
                if weight > 0.0:
                    all_rows.append((src_iso, tgt_iso, "alliance", int(row["year"]), weight))
                    
        print(f"Inserting {len(all_rows)} alliance weights...")
        cursor.executemany("""
            INSERT OR REPLACE INTO spatial_weights (country_code_source, country_code_target, weight_type, year, weight_value)
            VALUES (?, ?, ?, ?, ?);
        """, all_rows)
        conn.commit()
    else:
        print("Warning: Alliances raw file not found.")

    # 3. Trade
    if os.path.exists(TRADE_PATH):
        print(f"Loading trade data from: {TRADE_PATH}")
        df_trade = pd.read_csv(TRADE_PATH, usecols=["ccode1", "ccode2", "year", "smoothtotrade"])
        df_trade = df_trade[(df_trade["year"] >= 1990) & (df_trade["year"] <= 2028)]
        df_trade = df_trade.dropna(subset=["smoothtotrade"])
        df_trade = df_trade[df_trade["smoothtotrade"] > 0]
        
        # Forward-fill to 2028
        max_year = df_trade["year"].max()
        if max_year < 2028:
            print(f"Forward-filling trade from {max_year} to 2028...")
            df_latest = df_trade[df_trade["year"] == max_year]
            extra_dfs = []
            for yr in range(max_year + 1, 2029):
                df_yr = df_latest.copy()
                df_yr["year"] = yr
                extra_dfs.append(df_yr)
            df_trade = pd.concat([df_trade] + extra_dfs, ignore_index=True)
            
        # Map codes
        df_trade["src_iso"] = df_trade["ccode1"].map(cow_to_iso)
        df_trade["tgt_iso"] = df_trade["ccode2"].map(cow_to_iso)
        df_trade = df_trade.dropna(subset=["src_iso", "tgt_iso"])
        
        # Filter to valid codes in database
        df_trade = df_trade[df_trade["src_iso"].isin(valid_codes) & df_trade["tgt_iso"].isin(valid_codes)]
        # Filter out self loops
        df_trade = df_trade[df_trade["src_iso"] != df_trade["tgt_iso"]]
        
        # Calculate sum of trade per source-year
        trade_sums = df_trade.groupby(["src_iso", "year"])["smoothtotrade"].sum().reset_index()
        trade_sums = trade_sums.rename(columns={"smoothtotrade": "total_trade"})
        
        df_trade = pd.merge(df_trade, trade_sums, on=["src_iso", "year"])
        df_trade["weight_value"] = df_trade["smoothtotrade"] / df_trade["total_trade"]
        
        # Keep top 15 trade partners per source-year
        df_trade = df_trade.sort_values(["src_iso", "year", "weight_value"], ascending=[True, True, False])
        df_top_trade = df_trade.groupby(["src_iso", "year"]).head(15)
        
        trade_rows = []
        for _, row in df_top_trade.iterrows():
            trade_rows.append((
                row["src_iso"],
                row["tgt_iso"],
                "trade",
                int(row["year"]),
                float(row["weight_value"])
            ))
            
        print(f"Inserting {len(trade_rows)} trade weights...")
        cursor.executemany("""
            INSERT OR REPLACE INTO spatial_weights (country_code_source, country_code_target, weight_type, year, weight_value)
            VALUES (?, ?, ?, ?, ?);
        """, trade_rows)
        conn.commit()
    else:
        print("Warning: Trade raw file not found.")

    # 4. UNGA Voting
    if os.path.exists(UNGA_PATH):
        print(f"Loading UNGA voting data from: {UNGA_PATH}")
        df_unga = pd.read_csv(UNGA_PATH, usecols=["undl_id", "ms_code", "ms_vote", "date"])
        df_unga["year"] = df_unga["date"].astype(str).str[:4].astype(int)
        df_unga = df_unga[(df_unga["year"] >= 1990) & (df_unga["year"] <= 2028)]
        df_unga = df_unga.dropna(subset=["ms_code"])
        
        # Filter to codes that exist in database
        df_unga = df_unga[df_unga["ms_code"].isin(valid_codes)]
        
        vote_map = {"Y": 1, "N": -1, "A": 0, "X": 0}
        
        years = sorted(df_unga["year"].unique())
        unga_rows = []
        
        for yr in years:
            df_yr = df_unga[df_unga["year"] == yr]
            if df_yr.empty:
                continue
                
            # Pivot
            pivoted = df_yr.pivot(index="ms_code", columns="undl_id", values="ms_vote")
            if pivoted.empty:
                continue
                
            # Map vote codes
            pivoted = pivoted.map(lambda x: vote_map.get(x, 0)).fillna(0)
            
            # Compute cosine similarity
            matrix = pivoted.values
            countries = pivoted.index.tolist()
            
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            
            sim = np.dot(matrix, matrix.T) / (norms @ norms.T)
            # Rescale to [0, 1]
            sim = (sim + 1.0) / 2.0
            
            # Exclude self-matches
            np.fill_diagonal(sim, -1.0)
            
            # Extract top 15 similar partners per country
            for i, src in enumerate(countries):
                top_indices = np.argsort(sim[i])[::-1][:15]
                for idx in top_indices:
                    val = sim[i, idx]
                    if val >= 0.0:
                        unga_rows.append((src, countries[idx], "unga_voting", int(yr), float(val)))
                        
        # Forward-fill UNGA voting weights from max calculated year to 2028
        if unga_rows:
            max_unga_yr = max(r[3] for r in unga_rows)
            if max_unga_yr < 2028:
                print(f"Forward-filling UNGA voting weights from {max_unga_yr} to 2028...")
                latest_unga = [r for r in unga_rows if r[3] == max_unga_yr]
                extra_unga = []
                for yr in range(max_unga_yr + 1, 2029):
                    for src, tgt, wt_type, _, val in latest_unga:
                        extra_unga.append((src, tgt, wt_type, yr, val))
                unga_rows.extend(extra_unga)

        print(f"Inserting {len(unga_rows)} UNGA voting weights...")
        cursor.executemany("""
            INSERT OR REPLACE INTO spatial_weights (country_code_source, country_code_target, weight_type, year, weight_value)
            VALUES (?, ?, ?, ?, ?);
        """, unga_rows)
        conn.commit()
    else:
        print("Warning: UNGA voting raw file not found.")

    # 5. Geographic Proximity
    print("Calculating geographic proximity weights...")
    cursor.execute("SELECT country_code, latitude, longitude FROM countries;")
    country_coords = cursor.fetchall()
    
    # Filter out any countries with missing coords
    country_coords = [c for c in country_coords if c[1] is not None and c[2] is not None]
    
    import math
    def haversine_distance(lat1, lon1, lat2, lon2):
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)
        
        a = math.sin(delta_phi/2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda/2.0)**2
        c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
        return 6371.0 * c # km
        
    proximity_rows = []
    for code1, lat1, lon1 in country_coords:
        dists = []
        for code2, lat2, lon2 in country_coords:
            if code1 == code2:
                continue
            d = haversine_distance(lat1, lon1, lat2, lon2)
            weight = 1000.0 / (1000.0 + d)
            dists.append((code2, weight))
            
        # Take top 15 nearest partners
        dists = sorted(dists, key=lambda x: x[1], reverse=True)[:15]
        
        for code2, weight in dists:
            for yr in range(1990, 2029):
                proximity_rows.append((code1, code2, "proximity", yr, weight))
                
    print(f"Inserting {len(proximity_rows)} geographic proximity weights...")
    cursor.executemany("""
        INSERT OR REPLACE INTO spatial_weights (country_code_source, country_code_target, weight_type, year, weight_value)
        VALUES (?, ?, ?, ?, ?);
    """, proximity_rows)
    conn.commit()

    cursor.execute("SELECT COUNT(*), COUNT(DISTINCT weight_type) FROM spatial_weights;")
    w_count, t_count = cursor.fetchone()
    print(f"Ingested {w_count} time-varying dyadic weights across {t_count} network layers successfully.")
    
    conn.close()

if __name__ == "__main__":
    load_or_generate_networks()
