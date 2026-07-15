"""
Bulk GDELT Download Script

Downloads ALL GDELT 1.0 archive files (2006 through today) to raw_data/gdelt_cache/.
This is a one-time bulk download so that GDELT data is locally available for any
election window without needing repeated network calls.

File structure on GDELT servers:
  - Pre-2006: yearly archives (e.g., 2005.zip)
  - 2006-2013: monthly archives (e.g., 200601.zip)
  - 2013-04-01+: daily archives (e.g., 20130401.export.CSV.zip)

Usage:
    python data_ingest/bulk_download_gdelt.py
"""

import os
import urllib.request
import concurrent.futures
from datetime import datetime, timedelta

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(BASE_DIR, "raw_data", "gdelt_cache")
NUM_WORKERS = 12  # Slightly lower than ingest to be gentler on the server

os.makedirs(CACHE_DIR, exist_ok=True)


def download_file(filename, url):
    """Download a single file if not already cached."""
    cache_path = os.path.join(CACHE_DIR, filename)
    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
        return cache_path, "cached"
    
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=60) as response:
            data = response.read()
        with open(cache_path, "wb") as f:
            f.write(data)
        return cache_path, "downloaded"
    except Exception as e:
        return None, str(e)


def generate_file_list():
    """Generate the complete list of GDELT archive files from 1990 through today."""
    files = {}
    today = datetime.now()
    
    # 1. Yearly archives: 1990-2005
    for year in range(1990, 2006):
        filename = f"{year}.zip"
        url = f"http://data.gdeltproject.org/events/{filename}"
        files[filename] = url
    
    # 2. Monthly archives: 2006-01 through 2013-03
    dt = datetime(2006, 1, 1)
    end_monthly = datetime(2013, 4, 1)
    while dt < end_monthly:
        filename = f"{dt.year}{dt.month:02d}.zip"
        url = f"http://data.gdeltproject.org/events/{filename}"
        files[filename] = url
        # Advance to next month
        if dt.month == 12:
            dt = datetime(dt.year + 1, 1, 1)
        else:
            dt = datetime(dt.year, dt.month + 1, 1)
    
    # 3. Daily archives: 2013-04-01 through today
    dt = datetime(2013, 4, 1)
    while dt <= today:
        date_str = dt.strftime("%Y%m%d")
        filename = f"{date_str}.export.CSV.zip"
        url = f"http://data.gdeltproject.org/events/{filename}"
        files[filename] = url
        dt += timedelta(days=1)
    
    return files


def main():
    print("=" * 60)
    print("GDELT 1.0 Bulk Archive Downloader")
    print("=" * 60)
    
    files = generate_file_list()
    
    # Check how many are already cached
    already_cached = 0
    to_download = {}
    for fname, url in files.items():
        cache_path = os.path.join(CACHE_DIR, fname)
        if os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
            already_cached += 1
        else:
            to_download[fname] = url
    
    print(f"Total archive files: {len(files)}")
    print(f"Already cached:      {already_cached}")
    print(f"To download:         {len(to_download)}")
    print(f"Cache directory:     {CACHE_DIR}")
    print(f"Workers:             {NUM_WORKERS}")
    print()
    
    if not to_download:
        print("All files already cached. Nothing to download.")
        return
    
    # Download in parallel
    completed = 0
    downloaded = 0
    failed = 0
    total = len(to_download)
    sorted_files = sorted(to_download.keys())
    
    print(f"Starting download of {total} files...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        future_to_file = {
            executor.submit(download_file, fname, to_download[fname]): fname
            for fname in sorted_files
        }
        
        for future in concurrent.futures.as_completed(future_to_file):
            fname = future_to_file[future]
            try:
                path, status = future.result()
                completed += 1
                if status == "downloaded":
                    downloaded += 1
                elif status != "cached":
                    failed += 1
                    
                if completed % 100 == 0 or completed == total:
                    print(f"  Progress: {completed}/{total} complete ({downloaded} downloaded, {failed} failed)")
            except Exception as e:
                completed += 1
                failed += 1
                if completed % 100 == 0:
                    print(f"  Progress: {completed}/{total} complete ({downloaded} downloaded, {failed} failed)")
    
    print(f"\n{'=' * 60}")
    print(f"Download complete!")
    print(f"  Downloaded: {downloaded}")
    print(f"  Failed:     {failed} (some daily files may not exist on the server, which is normal)")
    print(f"  Total cached files: {already_cached + downloaded}")
    
    # Report cache size
    total_size = 0
    file_count = 0
    for fname in os.listdir(CACHE_DIR):
        fpath = os.path.join(CACHE_DIR, fname)
        if os.path.isfile(fpath):
            total_size += os.path.getsize(fpath)
            file_count += 1
    
    print(f"  Cache directory size: {total_size / (1024**3):.2f} GB ({file_count} files)")


if __name__ == "__main__":
    start_time = datetime.now()
    main()
    elapsed = datetime.now() - start_time
    print(f"  Total time: {elapsed}")
