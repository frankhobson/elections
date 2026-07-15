import os
import sqlite3
import pandas as pd
import numpy as np
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "elections.db")
RAW_DIR = os.path.join(BASE_DIR, "raw_data")
NELDA_RAW_PATH = os.path.join(RAW_DIR, "NELDA.csv")

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def load_or_generate_nelda():
    conn = get_connection()
    cursor = conn.cursor()

    if os.path.exists(NELDA_RAW_PATH):
        print(f"Loading NELDA data from raw file: {NELDA_RAW_PATH}")
        df = pd.read_csv(NELDA_RAW_PATH, sep='\t', encoding='latin1')
        
        # Filter to years 1990-2025
        df = df[(df["year"] >= 1990) & (df["year"] <= 2025)]
        
        # Load COW to ISO mapping
        mapping_path = os.path.join(BASE_DIR, "data_ingest", "cow_to_iso.json")
        if os.path.exists(mapping_path):
            with open(mapping_path) as f:
                cow_to_iso = {int(k): v for k, v in json.load(f).items()}
        else:
            cow_to_iso = {}
            
        # Manual overrides for small states or mismatches
        manual_overrides = {
            31: "BHS", 54: "DMA", 55: "GRD", 56: "LCA", 57: "VCT", 58: "ATG", 60: "KNA",
            80: "BLZ", 221: "MCO", 223: "LIE", 232: "AND", 325: "SMR", 340: "SRB", 345: "SRB",
            940: "NRU", 946: "KIR", 947: "TUV", 955: "TON", 983: "MHL", 986: "PLW",
            987: "FSM", 990: "WSM", 835: "BRN"
        }
        for k, v in manual_overrides.items():
            cow_to_iso[k] = v
            
        # Get valid country codes from countries table
        cursor.execute("SELECT country_code FROM countries;")
        valid_codes = {row[0] for row in cursor.fetchall()}
        
        election_rows = []
        nelda_rows = []
        
        for _, row in df.iterrows():
            ccode = int(row["ccode"])
            iso = cow_to_iso.get(ccode)
            if not iso or iso not in valid_codes:
                continue # Skip unmapped or non-Vdem countries
                
            year = int(row["year"])
            
            # Map election type
            types_val = str(row["types"]).strip()
            if types_val == "Executive":
                election_type = "Presidential"
            else:
                election_type = "Legislative"
                
            # Parse date
            mmdd = row["mmdd"]
            if pd.isna(mmdd) or mmdd <= 0:
                election_date = f"{year}-06-15" # fallback mid-year
            else:
                month = int(mmdd // 100)
                day = int(mmdd % 100)
                if month < 1 or month > 12 or day < 1 or day > 31:
                    election_date = f"{year}-06-15"
                else:
                    election_date = f"{year}-{month:02d}-{day:02d}"
            
            # Map is_scheduled from nelda6
            n6 = str(row["nelda6"]).strip().lower()
            is_scheduled = 0 if n6 == "yes" else 1
            
            # Map is_fraudulent from nelda11
            n11 = str(row["nelda11"]).strip().lower()
            is_fraudulent = 1 if n11 == "yes" else 0
            
            # Map incumbent_ran from nelda21
            n21 = str(row["nelda21"]).strip().lower()
            if n21 == "yes":
                incumbent_ran = 1
            elif n21 == "no":
                incumbent_ran = 0
            else:
                incumbent_ran = None
                
            # Map has_successor from nelda22
            n22 = str(row["nelda22"]).strip().lower()
            if n22 == "yes":
                has_successor = 1
            elif n22 == "no":
                has_successor = 0
            else:
                has_successor = None
            
            # Insert election
            cursor.execute("""
                INSERT INTO elections (country_code, election_year, election_date, election_type, is_scheduled, is_fraudulent, incumbent_ran, has_successor)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """, (iso, year, election_date, election_type, is_scheduled, is_fraudulent, incumbent_ran, has_successor))
            
            elect_id = cursor.lastrowid
            
            # Map nelda_indicators values
            # nelda24 (incumbent party lost?): no -> Yes, yes -> No, else -> N/A
            n24 = str(row["nelda24"]).strip().lower()
            if n24 == "no":
                nelda23_incumbent_won = "Yes"
            elif n24 == "yes":
                nelda23_incumbent_won = "No"
            else:
                nelda23_incumbent_won = "N/A"
                
            # nelda11 (fraud allegations)
            nelda11_val = "Yes" if n11 == "yes" else ("No" if n11 == "no" else "N/A")
            
            # nelda15 (opposition boycott)
            n15 = str(row["nelda15"]).strip().lower()
            nelda15_val = "Yes" if n15 == "yes" else ("No" if n15 == "no" else "N/A")
            
            # nelda51 (monitors present)
            n51 = str(row["nelda51"]).strip().lower()
            nelda51_val = "Yes" if n51 == "yes" else ("No" if n51 == "no" else "N/A")
            
            nelda_rows.append((
                elect_id,
                nelda11_val,
                nelda15_val,
                nelda23_incumbent_won,
                nelda51_val
            ))
            
        print(f"Inserting {len(nelda_rows)} elections and indicators into database...")
        cursor.executemany("""
            INSERT OR REPLACE INTO nelda_indicators (election_id, nelda11_fraud_allegations, nelda15_opposition_boycott, nelda23_incumbent_won, nelda51_monitors_present)
            VALUES (?, ?, ?, ?, ?);
        """, nelda_rows)
        
    else:
        raise FileNotFoundError(f"Raw NELDA CSV not found at: {NELDA_RAW_PATH}")

    conn.commit()
    
    cursor.execute("SELECT COUNT(*) FROM elections;")
    e_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM nelda_indicators;")
    n_count = cursor.fetchone()[0]
    print(f"Ingested {e_count} elections and {n_count} election-level NELDA records successfully.")
    
    conn.close()

if __name__ == "__main__":
    load_or_generate_nelda()
