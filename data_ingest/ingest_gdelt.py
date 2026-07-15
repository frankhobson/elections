import os
import sqlite3
import json
import zipfile
import io
import urllib.request
import concurrent.futures
import time
from datetime import datetime, timedelta

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "elections.db")
CACHE_DIR = os.path.join(BASE_DIR, "raw_data", "gdelt_cache")
MIN_MENTIONS = 5
NUM_WORKERS = 24

# Ensure cache directory exists
os.makedirs(CACHE_DIR, exist_ok=True)

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def load_country_mappings():
    mapping_path = os.path.join(BASE_DIR, "data_ingest", "fips_to_iso.json")
    with open(mapping_path, "r", encoding="utf-8") as f:
        fips_to_iso = json.load(f)
        
    # Invert mapping to get ISO -> list of FIPS
    iso_to_fips = {}
    for fips, iso3 in fips_to_iso.items():
        if iso3 not in iso_to_fips:
            iso_to_fips[iso3] = []
        iso_to_fips[iso3].append(fips)
        
    return fips_to_iso, iso_to_fips

def get_election_windows(iso_to_fips):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT election_id, country_code, election_date FROM elections WHERE election_date IS NOT NULL;")
    elections = cursor.fetchall()
    conn.close()
    
    active_fips_by_date = {}
    
    for elect_id, country_code, date_str in elections:
        try:
            elect_date = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            continue
            
        start_date = elect_date - timedelta(days=180)
        fips_list = iso_to_fips.get(country_code, [])
        if not fips_list:
            continue
            
        curr = start_date
        while curr <= elect_date:
            day_key = curr.strftime("%Y%m%d")
            if day_key not in active_fips_by_date:
                active_fips_by_date[day_key] = set()
            for fips in fips_list:
                active_fips_by_date[day_key].add(fips)
            curr += timedelta(days=1)
            
    return elections, active_fips_by_date

def download_file(filename, url):
    cache_path = os.path.join(CACHE_DIR, filename)
    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
        return cache_path, "cached"
        
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=45) as response:
            data = response.read()
        with open(cache_path, "wb") as f:
            f.write(data)
        return cache_path, "downloaded"
    except Exception as e:
        # Some daily files might be missing on the GDELT server, which is normal.
        return None, str(e)

def parse_and_filter_zip(cache_path, active_fips_by_date, fips_to_iso, min_mentions):
    events_batch = []
    try:
        with zipfile.ZipFile(cache_path) as z:
            namelist = z.namelist()
            if not namelist:
                return []
            with z.open(namelist[0]) as f:
                for line_bytes in f:
                    try:
                        line = line_bytes.decode('utf-8', errors='ignore')
                    except Exception:
                        continue
                    parts = line.strip('\n').split('\t')
                    if len(parts) < 52:
                        continue
                        
                    event_date_str = parts[1] # YYYYMMDD
                    fips = parts[51] # ActionGeo_CountryCode
                    
                    if event_date_str not in active_fips_by_date:
                        continue
                        
                    if fips not in active_fips_by_date[event_date_str]:
                        continue
                        
                    try:
                        mentions = int(parts[31])
                    except ValueError:
                        continue
                    if mentions < min_mentions:
                        continue
                        
                    cameo_code = parts[26]
                    # Filter for target CAMEO event code prefixes:
                    # Protest (14), Military clash/Fight (19), Express approval/Alliance (03),
                    # Sign treaty/Agreement (04), File complaint/Accuse (11), Make public statement (02)
                    if not (cameo_code.startswith("14") or cameo_code.startswith("19") or 
                            cameo_code.startswith("03") or cameo_code.startswith("04") or 
                            cameo_code.startswith("11") or cameo_code.startswith("02")):
                        continue
                        
                    iso3 = fips_to_iso.get(fips)
                    if not iso3:
                        continue
                        
                    event_id = parts[0]
                    try:
                        goldstein = float(parts[30]) if parts[30] else 0.0
                    except ValueError:
                        goldstein = 0.0
                        
                    try:
                        tone = float(parts[34]) if parts[34] else 0.0
                    except ValueError:
                        tone = 0.0
                        
                    try:
                        formatted_date = f"{event_date_str[0:4]}-{event_date_str[4:6]}-{event_date_str[6:8]}"
                    except IndexError:
                        continue
                        
                    actor1_code = parts[6] if len(parts) > 6 else ""
                    actor2_code = parts[14] if len(parts) > 14 else ""
                    
                    events_batch.append((
                        event_id,
                        iso3,
                        formatted_date,
                        cameo_code,
                        goldstein,
                        tone,
                        mentions,
                        actor1_code,
                        actor2_code
                    ))
    except Exception as e:
        print(f"Error parsing {cache_path}: {e}")
        
    return events_batch

def process_file(filename, url, active_fips_by_date, fips_to_iso, min_mentions):
    cache_path, status = download_file(filename, url)
    if not cache_path:
        return []
    return parse_and_filter_zip(cache_path, active_fips_by_date, fips_to_iso, min_mentions)

def insert_events(conn, events):
    if not events:
        return
    cursor = conn.cursor()
    cursor.executemany("""
        INSERT OR REPLACE INTO gdelt_events (
            event_id, country_code, event_date, cameo_code, 
            goldstein_scale, avg_tone, news_volume, actor1_code, actor2_code
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
    """, events)
    conn.commit()

def ingest_gdelt():
    print("Loading country mappings and election windows...")
    fips_to_iso, iso_to_fips = load_country_mappings()
    elections, active_fips_by_date = get_election_windows(iso_to_fips)
    
    # Collect all unique required files
    required_files = {}
    for date_key in active_fips_by_date.keys():
        try:
            dt = datetime.strptime(date_key, "%Y%m%d")
        except ValueError:
            continue
            
        if dt >= datetime(2013, 4, 1):
            filename = f"{date_key}.export.CSV.zip"
            url = f"http://data.gdeltproject.org/events/{filename}"
            required_files[filename] = url
        elif dt >= datetime(2006, 1, 1):
            filename = f"{dt.year}{dt.month:02d}.zip"
            url = f"http://data.gdeltproject.org/events/{filename}"
            required_files[filename] = url
        else:
            filename = f"{dt.year}.zip"
            url = f"http://data.gdeltproject.org/events/{filename}"
            required_files[filename] = url
            
    print(f"Total elections: {len(elections)}")
    print(f"Unique GDELT archives required: {len(required_files)}")
    
    conn = get_connection()
    
    # Use ThreadPoolExecutor to parallelize download and parsing
    completed = 0
    total = len(required_files)
    batch = []
    
    print(f"Starting parallel ingestion with {NUM_WORKERS} workers...")
    
    # Sort required files so we process yearly, monthly, then daily to keep memory footprint steady
    sorted_files = sorted(required_files.keys())
    
    last_report_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        future_to_file = {
            executor.submit(process_file, fname, required_files[fname], active_fips_by_date, fips_to_iso, MIN_MENTIONS): fname
            for fname in sorted_files
        }
        
        for future in concurrent.futures.as_completed(future_to_file):
            fname = future_to_file[future]
            try:
                events = future.result()
                batch.extend(events)
                completed += 1
                
                # Insert in batches to prevent high memory usage
                if len(batch) >= 100000 or completed == total:
                    if batch:
                        insert_events(conn, batch)
                        batch = []
                
                # Print progress update once per minute
                current_time = time.time()
                if current_time - last_report_time >= 60.0 or completed == total:
                    pct = (completed / total) * 100 if total > 0 else 100.0
                    print(f"Progress update: {completed}/{total} files processed ({pct:.1f}% complete).")
                    last_report_time = current_time
            except Exception as e:
                print(f"Error processing future for {fname}: {e}")
                
        # Final flush
        if batch:
            insert_events(conn, batch)
            
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM gdelt_events;")
    total_db_events = cursor.fetchone()[0]
    print(f"GDELT ingestion finished. Total real events in DB: {total_db_events}")
    
    conn.close()

if __name__ == "__main__":
    start_time = datetime.now()
    ingest_gdelt()
    print(f"Ingestion took {datetime.now() - start_time}")
