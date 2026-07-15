import os
import re
import sqlite3
import pandas as pd
import numpy as np

BASE_DIR = "/Users/frankhobson/Documents/Antigravity Workspace/Election Correlations and Predictions"
DB_PATH = os.path.join(BASE_DIR, "elections.db")
CSV_PATH = os.path.join(BASE_DIR, "raw_data", "exchange_rates.csv")

def main():
    if not os.path.exists(CSV_PATH):
        print(f"Error: {CSV_PATH} not found. Please place the file first.")
        return
        
    print("Connecting to database...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get valid country codes from countries table
    cursor.execute("SELECT country_code FROM countries")
    valid_country_codes = {row[0] for row in cursor.fetchall()}
    print(f"Loaded {len(valid_country_codes)} valid country codes from database.")
    
    print("Reading exchange rates CSV...")
    # Low_memory=False to avoid mixed type warnings
    df = pd.read_csv(CSV_PATH, low_memory=False)
    
    # Clean column names and row values
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]
    
    # Filter out footer rows and aggregate zones (Country Code must be a 3-character string)
    df = df[df["Country Code"].notna()]
    df["Country Code"] = df["Country Code"].astype(str).str.strip().str.upper()
    df = df[df["Country Code"].str.len() == 3]
    
    # Identify monthly exchange rate columns (e.g. "1990M01 [1990M01]" or "1990M01")
    monthly_cols = []
    col_mapping = {} # col_name -> (year, month)
    
    pattern = re.compile(r"^([0-9]{4})M([0-9]{2})")
    for col in df.columns:
        match = pattern.match(col)
        if match:
            year = int(match.group(1))
            month = int(match.group(2))
            monthly_cols.append(col)
            col_mapping[col] = (year, month)
            
    print(f"Identified {len(monthly_cols)} monthly exchange rate columns.")
    
    # Insert data
    insert_records = []
    skipped_unmapped_countries = set()
    skipped_values_count = 0
    
    for _, row in df.iterrows():
        c_code = row["Country Code"]
        if c_code not in valid_country_codes:
            skipped_unmapped_countries.add(c_code)
            continue
            
        for col in monthly_cols:
            val_str = str(row[col]).strip()
            # World Bank uses ".." to represent missing values
            if val_str in ["", "..", "nan", "None", "."]:
                skipped_values_count += 1
                continue
                
            try:
                # Clean value of any surrounding spaces/quotes
                val = float(val_str)
                if np.isnan(val):
                    skipped_values_count += 1
                    continue
                
                year, month = col_mapping[col]
                insert_records.append((c_code, year, month, val))
            except ValueError:
                skipped_values_count += 1
                continue
                
    print(f"Prepared {len(insert_records)} records for database insertion.")
    if skipped_unmapped_countries:
        print(f"Skipped {len(skipped_unmapped_countries)} country codes not in countries table: {sorted(skipped_unmapped_countries)}")
    print(f"Skipped {skipped_values_count} missing/null/corrupted value points.")
    
    # Insert in batches
    cursor.execute("DELETE FROM exchange_rates")
    cursor.executemany("""
    INSERT OR REPLACE INTO exchange_rates (country_code, year, month, exchange_rate)
    VALUES (?, ?, ?, ?)
    """, insert_records)
    
    conn.commit()
    
    # Verify ingestion
    cursor.execute("SELECT COUNT(*), COUNT(DISTINCT country_code), MIN(year), MAX(year) FROM exchange_rates")
    total_rows, num_countries, min_yr, max_yr = cursor.fetchone()
    print("\n--- Ingestion Verification ---")
    print(f"Total Rows Ingested: {total_rows}")
    print(f"Distinct Countries:  {num_countries}")
    print(f"Year Range:          {min_yr} to {max_yr}")
    
    conn.close()
    print("Ingestion complete.")

if __name__ == "__main__":
    main()
