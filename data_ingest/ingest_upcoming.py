"""
Ingest Upcoming Elections (2021-2028)

Populates the upcoming_elections table with elections that occurred after NELDA's
coverage period, plus scheduled future elections. These elections may have incomplete
data (no NELDA indicators, potentially missing GDELT/macro/V-Dem coverage).

The script also triggers GDELT ingestion for any new election windows where
GDELT data is available in the local cache.
"""

import os
import sys
import sqlite3
import json
import zipfile
from datetime import datetime, timedelta

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "elections.db")
CACHE_DIR = os.path.join(BASE_DIR, "raw_data", "gdelt_cache")

# Add data_ingest directory to path for sibling imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


# Comprehensive list of national elections 2021 - mid-2028
# Sources: IFES Election Guide, Wikipedia, NDI, ACE Project
# For past elections (2021-mid 2026): outcome is "incumbent" or "challenger"
# For future elections (mid 2026-2028): outcome is null
ELECTIONS_2021_2028 = [
    # ============ 2021 ============
    {"country_name": "Uganda", "country_code": "UGA", "election_date": "2021-01-14", "election_type": "Presidential", "outcome": "incumbent", "is_scheduled": 1},
    {"country_name": "Uganda", "country_code": "UGA", "election_date": "2021-01-14", "election_type": "Legislative", "outcome": "incumbent", "is_scheduled": 1},
    {"country_name": "Portugal", "country_code": "PRT", "election_date": "2021-01-24", "election_type": "Presidential", "outcome": "incumbent", "is_scheduled": 1},
    {"country_name": "Ecuador", "country_code": "ECU", "election_date": "2021-02-07", "election_type": "Presidential", "outcome": "challenger", "is_scheduled": 1},
    {"country_name": "Niger", "country_code": "NER", "election_date": "2021-02-21", "election_type": "Presidential", "outcome": "incumbent", "is_scheduled": 1},
    {"country_name": "Kosovo", "country_code": "XKX", "election_date": "2021-02-14", "election_type": "Legislative", "outcome": "challenger", "is_scheduled": 0},
    {"country_name": "Israel", "country_code": "ISR", "election_date": "2021-03-23", "election_type": "Legislative", "outcome": "incumbent", "is_scheduled": 0},
    {"country_name": "Congo, Rep.", "country_code": "COG", "election_date": "2021-03-21", "election_type": "Presidential", "outcome": "incumbent", "is_scheduled": 1},
    {"country_name": "Djibouti", "country_code": "DJI", "election_date": "2021-04-09", "election_type": "Presidential", "outcome": "incumbent", "is_scheduled": 1},
    {"country_name": "Benin", "country_code": "BEN", "election_date": "2021-04-11", "election_type": "Presidential", "outcome": "incumbent", "is_scheduled": 1},
    {"country_name": "Chad", "country_code": "TCD", "election_date": "2021-04-11", "election_type": "Presidential", "outcome": "incumbent", "is_scheduled": 1},
    {"country_name": "Peru", "country_code": "PER", "election_date": "2021-04-11", "election_type": "Presidential", "outcome": "challenger", "is_scheduled": 1},
    {"country_name": "Albania", "country_code": "ALB", "election_date": "2021-04-25", "election_type": "Legislative", "outcome": "incumbent", "is_scheduled": 1},
    {"country_name": "United Kingdom", "country_code": "GBR", "election_date": "2021-05-06", "election_type": "Legislative", "outcome": "incumbent", "is_scheduled": 1, "notes": "Local and devolved elections"},
    {"country_name": "Syria", "country_code": "SYR", "election_date": "2021-05-26", "election_type": "Presidential", "outcome": "incumbent", "is_scheduled": 1},
    {"country_name": "Mexico", "country_code": "MEX", "election_date": "2021-06-06", "election_type": "Legislative", "outcome": "incumbent", "is_scheduled": 1},
    {"country_name": "Ethiopia", "country_code": "ETH", "election_date": "2021-06-21", "election_type": "Legislative", "outcome": "incumbent", "is_scheduled": 1},
    {"country_name": "Mongolia", "country_code": "MNG", "election_date": "2021-06-09", "election_type": "Presidential", "outcome": "challenger", "is_scheduled": 1},
    {"country_name": "Armenia", "country_code": "ARM", "election_date": "2021-06-20", "election_type": "Legislative", "outcome": "incumbent", "is_scheduled": 0},
    {"country_name": "Iran", "country_code": "IRN", "election_date": "2021-06-18", "election_type": "Presidential", "outcome": "incumbent", "is_scheduled": 1},
    {"country_name": "Moldova", "country_code": "MDA", "election_date": "2021-07-11", "election_type": "Legislative", "outcome": "challenger", "is_scheduled": 0},
    {"country_name": "Zambia", "country_code": "ZMB", "election_date": "2021-08-12", "election_type": "Presidential", "outcome": "challenger", "is_scheduled": 1},
    {"country_name": "Morocco", "country_code": "MAR", "election_date": "2021-09-08", "election_type": "Legislative", "outcome": "challenger", "is_scheduled": 1},
    {"country_name": "Russia", "country_code": "RUS", "election_date": "2021-09-19", "election_type": "Legislative", "outcome": "incumbent", "is_scheduled": 1},
    {"country_name": "Canada", "country_code": "CAN", "election_date": "2021-09-20", "election_type": "Legislative", "outcome": "incumbent", "is_scheduled": 0},
    {"country_name": "Iceland", "country_code": "ISL", "election_date": "2021-09-25", "election_type": "Legislative", "outcome": "incumbent", "is_scheduled": 1},
    {"country_name": "Germany", "country_code": "DEU", "election_date": "2021-09-26", "election_type": "Legislative", "outcome": "challenger", "is_scheduled": 1},
    {"country_name": "Czech Republic", "country_code": "CZE", "election_date": "2021-10-09", "election_type": "Legislative", "outcome": "challenger", "is_scheduled": 1},
    {"country_name": "Iraq", "country_code": "IRQ", "election_date": "2021-10-10", "election_type": "Legislative", "outcome": "challenger", "is_scheduled": 0},
    {"country_name": "Japan", "country_code": "JPN", "election_date": "2021-10-31", "election_type": "Legislative", "outcome": "incumbent", "is_scheduled": 0},
    {"country_name": "Nicaragua", "country_code": "NIC", "election_date": "2021-11-07", "election_type": "Presidential", "outcome": "incumbent", "is_scheduled": 1},
    {"country_name": "Bulgaria", "country_code": "BGR", "election_date": "2021-11-14", "election_type": "Legislative", "outcome": "challenger", "is_scheduled": 0},
    {"country_name": "Honduras", "country_code": "HND", "election_date": "2021-11-28", "election_type": "Presidential", "outcome": "challenger", "is_scheduled": 1},
    {"country_name": "Chile", "country_code": "CHL", "election_date": "2021-12-19", "election_type": "Presidential", "outcome": "challenger", "is_scheduled": 1},
    {"country_name": "Gambia", "country_code": "GMB", "election_date": "2021-12-04", "election_type": "Presidential", "outcome": "incumbent", "is_scheduled": 1},
    {"country_name": "Libya", "country_code": "LBY", "election_date": "2021-12-24", "election_type": "Presidential", "outcome": None, "is_scheduled": 1, "notes": "Postponed indefinitely"},
    
    # ============ 2022 ============
    {"country_name": "Barbados", "country_code": "BRB", "election_date": "2022-01-19", "election_type": "Legislative", "outcome": "incumbent", "is_scheduled": 0},
    {"country_name": "Italy", "country_code": "ITA", "election_date": "2022-01-24", "election_type": "Presidential", "outcome": "incumbent", "is_scheduled": 1, "notes": "Parliamentary election of president"},
    {"country_name": "Costa Rica", "country_code": "CRI", "election_date": "2022-02-06", "election_type": "Presidential", "outcome": "challenger", "is_scheduled": 1},
    {"country_name": "South Korea", "country_code": "KOR", "election_date": "2022-03-09", "election_type": "Presidential", "outcome": "challenger", "is_scheduled": 1},
    {"country_name": "Serbia", "country_code": "SRB", "election_date": "2022-04-03", "election_type": "Presidential", "outcome": "incumbent", "is_scheduled": 1},
    {"country_name": "Serbia", "country_code": "SRB", "election_date": "2022-04-03", "election_type": "Legislative", "outcome": "incumbent", "is_scheduled": 1},
    {"country_name": "Hungary", "country_code": "HUN", "election_date": "2022-04-03", "election_type": "Legislative", "outcome": "incumbent", "is_scheduled": 1},
    {"country_name": "France", "country_code": "FRA", "election_date": "2022-04-24", "election_type": "Presidential", "outcome": "incumbent", "is_scheduled": 1},
    {"country_name": "France", "country_code": "FRA", "election_date": "2022-06-19", "election_type": "Legislative", "outcome": "incumbent", "is_scheduled": 1},
    {"country_name": "Philippines", "country_code": "PHL", "election_date": "2022-05-09", "election_type": "Presidential", "outcome": "incumbent", "is_scheduled": 1, "notes": "Marcos Jr, same party coalition"},
    {"country_name": "Colombia", "country_code": "COL", "election_date": "2022-06-19", "election_type": "Presidential", "outcome": "challenger", "is_scheduled": 1},
    {"country_name": "Australia", "country_code": "AUS", "election_date": "2022-05-21", "election_type": "Legislative", "outcome": "challenger", "is_scheduled": 1},
    {"country_name": "Lebanon", "country_code": "LBN", "election_date": "2022-05-15", "election_type": "Legislative", "outcome": "challenger", "is_scheduled": 1},
    {"country_name": "Kenya", "country_code": "KEN", "election_date": "2022-08-09", "election_type": "Presidential", "outcome": "challenger", "is_scheduled": 1},
    {"country_name": "Angola", "country_code": "AGO", "election_date": "2022-08-24", "election_type": "Presidential", "outcome": "incumbent", "is_scheduled": 1},
    {"country_name": "Sweden", "country_code": "SWE", "election_date": "2022-09-11", "election_type": "Legislative", "outcome": "challenger", "is_scheduled": 1},
    {"country_name": "Italy", "country_code": "ITA", "election_date": "2022-09-25", "election_type": "Legislative", "outcome": "challenger", "is_scheduled": 0},
    {"country_name": "Brazil", "country_code": "BRA", "election_date": "2022-10-30", "election_type": "Presidential", "outcome": "challenger", "is_scheduled": 1},
    {"country_name": "Denmark", "country_code": "DNK", "election_date": "2022-11-01", "election_type": "Legislative", "outcome": "incumbent", "is_scheduled": 0},
    {"country_name": "Slovenia", "country_code": "SVN", "election_date": "2022-04-24", "election_type": "Legislative", "outcome": "challenger", "is_scheduled": 1},
    {"country_name": "Malaysia", "country_code": "MYS", "election_date": "2022-11-19", "election_type": "Legislative", "outcome": "challenger", "is_scheduled": 0},
    {"country_name": "Nepal", "country_code": "NPL", "election_date": "2022-11-20", "election_type": "Legislative", "outcome": "challenger", "is_scheduled": 1},
    {"country_name": "Tunisia", "country_code": "TUN", "election_date": "2022-12-17", "election_type": "Legislative", "outcome": "incumbent", "is_scheduled": 1, "notes": "Boycotted by major parties"},
    
    # ============ 2023 ============
    {"country_name": "Nigeria", "country_code": "NGA", "election_date": "2023-02-25", "election_type": "Presidential", "outcome": "incumbent", "is_scheduled": 1, "notes": "Tinubu, ruling APC party"},
    {"country_name": "Nigeria", "country_code": "NGA", "election_date": "2023-02-25", "election_type": "Legislative", "outcome": "incumbent", "is_scheduled": 1},
    {"country_name": "Estonia", "country_code": "EST", "election_date": "2023-03-05", "election_type": "Legislative", "outcome": "incumbent", "is_scheduled": 1},
    {"country_name": "Finland", "country_code": "FIN", "election_date": "2023-04-02", "election_type": "Legislative", "outcome": "challenger", "is_scheduled": 1},
    {"country_name": "Turkey", "country_code": "TUR", "election_date": "2023-05-28", "election_type": "Presidential", "outcome": "incumbent", "is_scheduled": 1},
    {"country_name": "Turkey", "country_code": "TUR", "election_date": "2023-05-14", "election_type": "Legislative", "outcome": "incumbent", "is_scheduled": 1},
    {"country_name": "Thailand", "country_code": "THA", "election_date": "2023-05-14", "election_type": "Legislative", "outcome": "challenger", "is_scheduled": 1},
    {"country_name": "Greece", "country_code": "GRC", "election_date": "2023-06-25", "election_type": "Legislative", "outcome": "incumbent", "is_scheduled": 0},
    {"country_name": "Guatemala", "country_code": "GTM", "election_date": "2023-08-20", "election_type": "Presidential", "outcome": "challenger", "is_scheduled": 1},
    {"country_name": "Ecuador", "country_code": "ECU", "election_date": "2023-10-15", "election_type": "Presidential", "outcome": "challenger", "is_scheduled": 0},
    {"country_name": "Argentina", "country_code": "ARG", "election_date": "2023-11-19", "election_type": "Presidential", "outcome": "challenger", "is_scheduled": 1},
    {"country_name": "Poland", "country_code": "POL", "election_date": "2023-10-15", "election_type": "Legislative", "outcome": "challenger", "is_scheduled": 1},
    {"country_name": "New Zealand", "country_code": "NZL", "election_date": "2023-10-14", "election_type": "Legislative", "outcome": "challenger", "is_scheduled": 1},
    {"country_name": "Liberia", "country_code": "LBR", "election_date": "2023-11-14", "election_type": "Presidential", "outcome": "challenger", "is_scheduled": 1},
    {"country_name": "Netherlands", "country_code": "NLD", "election_date": "2023-11-22", "election_type": "Legislative", "outcome": "challenger", "is_scheduled": 0},
    {"country_name": "Congo, Dem. Rep.", "country_code": "COD", "election_date": "2023-12-20", "election_type": "Presidential", "outcome": "incumbent", "is_scheduled": 1},
    {"country_name": "Madagascar", "country_code": "MDG", "election_date": "2023-11-16", "election_type": "Presidential", "outcome": "incumbent", "is_scheduled": 1},
    {"country_name": "Zimbabwe", "country_code": "ZWE", "election_date": "2023-08-23", "election_type": "Presidential", "outcome": "incumbent", "is_scheduled": 1},
    {"country_name": "Spain", "country_code": "ESP", "election_date": "2023-07-23", "election_type": "Legislative", "outcome": "challenger", "is_scheduled": 0},
    {"country_name": "Paraguay", "country_code": "PRY", "election_date": "2023-04-30", "election_type": "Presidential", "outcome": "incumbent", "is_scheduled": 1},
    {"country_name": "Cambodia", "country_code": "KHM", "election_date": "2023-07-23", "election_type": "Legislative", "outcome": "incumbent", "is_scheduled": 1},
    {"country_name": "Sierra Leone", "country_code": "SLE", "election_date": "2023-06-24", "election_type": "Presidential", "outcome": "incumbent", "is_scheduled": 1},
    {"country_name": "Switzerland", "country_code": "CHE", "election_date": "2023-10-22", "election_type": "Legislative", "outcome": "challenger", "is_scheduled": 1},
    {"country_name": "Luxembourg", "country_code": "LUX", "election_date": "2023-10-08", "election_type": "Legislative", "outcome": "challenger", "is_scheduled": 1},
    
    # ============ 2024 ============
    {"country_name": "Bangladesh", "country_code": "BGD", "election_date": "2024-01-07", "election_type": "Legislative", "outcome": "incumbent", "is_scheduled": 1, "notes": "Boycotted by opposition"},
    {"country_name": "Taiwan", "country_code": "TWN", "election_date": "2024-01-13", "election_type": "Presidential", "outcome": "incumbent", "is_scheduled": 1, "notes": "DPP won, ruling party retained"},
    {"country_name": "Finland", "country_code": "FIN", "election_date": "2024-01-28", "election_type": "Presidential", "outcome": "challenger", "is_scheduled": 1},
    {"country_name": "El Salvador", "country_code": "SLV", "election_date": "2024-02-04", "election_type": "Presidential", "outcome": "incumbent", "is_scheduled": 1},
    {"country_name": "Indonesia", "country_code": "IDN", "election_date": "2024-02-14", "election_type": "Presidential", "outcome": "incumbent", "is_scheduled": 1, "notes": "Prabowo, Jokowi coalition successor"},
    {"country_name": "Pakistan", "country_code": "PAK", "election_date": "2024-02-08", "election_type": "Legislative", "outcome": "challenger", "is_scheduled": 1},
    {"country_name": "Senegal", "country_code": "SEN", "election_date": "2024-03-24", "election_type": "Presidential", "outcome": "challenger", "is_scheduled": 1},
    {"country_name": "Russia", "country_code": "RUS", "election_date": "2024-03-17", "election_type": "Presidential", "outcome": "incumbent", "is_scheduled": 1},
    {"country_name": "India", "country_code": "IND", "election_date": "2024-06-04", "election_type": "Legislative", "outcome": "incumbent", "is_scheduled": 1},
    {"country_name": "South Africa", "country_code": "ZAF", "election_date": "2024-05-29", "election_type": "Legislative", "outcome": "incumbent", "is_scheduled": 1, "notes": "ANC lost majority but led coalition"},
    {"country_name": "Mexico", "country_code": "MEX", "election_date": "2024-06-02", "election_type": "Presidential", "outcome": "incumbent", "is_scheduled": 1, "notes": "Sheinbaum, same MORENA party"},
    {"country_name": "European Union", "country_code": "EU_", "election_date": "2024-06-09", "election_type": "Legislative", "outcome": "incumbent", "is_scheduled": 1, "notes": "European Parliament elections — skipped, no country code"},
    {"country_name": "United Kingdom", "country_code": "GBR", "election_date": "2024-07-04", "election_type": "Legislative", "outcome": "challenger", "is_scheduled": 0},
    {"country_name": "Iran", "country_code": "IRN", "election_date": "2024-07-05", "election_type": "Presidential", "outcome": "challenger", "is_scheduled": 0, "notes": "Snap election after Raisi death"},
    {"country_name": "Venezuela", "country_code": "VEN", "election_date": "2024-07-28", "election_type": "Presidential", "outcome": "incumbent", "is_scheduled": 1, "notes": "Disputed result"},
    {"country_name": "Rwanda", "country_code": "RWA", "election_date": "2024-07-15", "election_type": "Presidential", "outcome": "incumbent", "is_scheduled": 1},
    {"country_name": "Sri Lanka", "country_code": "LKA", "election_date": "2024-09-21", "election_type": "Presidential", "outcome": "challenger", "is_scheduled": 1},
    {"country_name": "Jordan", "country_code": "JOR", "election_date": "2024-09-10", "election_type": "Legislative", "outcome": "incumbent", "is_scheduled": 1},
    {"country_name": "Austria", "country_code": "AUT", "election_date": "2024-09-29", "election_type": "Legislative", "outcome": "challenger", "is_scheduled": 1},
    {"country_name": "Tunisia", "country_code": "TUN", "election_date": "2024-10-06", "election_type": "Presidential", "outcome": "incumbent", "is_scheduled": 1},
    {"country_name": "Georgia", "country_code": "GEO", "election_date": "2024-10-26", "election_type": "Legislative", "outcome": "incumbent", "is_scheduled": 1, "notes": "Contested results"},
    {"country_name": "Japan", "country_code": "JPN", "election_date": "2024-10-27", "election_type": "Legislative", "outcome": "incumbent", "is_scheduled": 0},
    {"country_name": "United States", "country_code": "USA", "election_date": "2024-11-05", "election_type": "Presidential", "outcome": "challenger", "is_scheduled": 1},
    {"country_name": "Uruguay", "country_code": "URY", "election_date": "2024-11-24", "election_type": "Presidential", "outcome": "challenger", "is_scheduled": 1},
    {"country_name": "Panama", "country_code": "PAN", "election_date": "2024-05-05", "election_type": "Presidential", "outcome": "challenger", "is_scheduled": 1},
    {"country_name": "Dominican Republic", "country_code": "DOM", "election_date": "2024-05-19", "election_type": "Presidential", "outcome": "incumbent", "is_scheduled": 1},
    {"country_name": "Lithuania", "country_code": "LTU", "election_date": "2024-10-27", "election_type": "Legislative", "outcome": "challenger", "is_scheduled": 1},
    {"country_name": "Romania", "country_code": "ROU", "election_date": "2024-12-01", "election_type": "Legislative", "outcome": "challenger", "is_scheduled": 1},
    {"country_name": "Ghana", "country_code": "GHA", "election_date": "2024-12-07", "election_type": "Presidential", "outcome": "challenger", "is_scheduled": 1},
    {"country_name": "Botswana", "country_code": "BWA", "election_date": "2024-10-30", "election_type": "Legislative", "outcome": "challenger", "is_scheduled": 1},
    {"country_name": "Mozambique", "country_code": "MOZ", "election_date": "2024-10-09", "election_type": "Presidential", "outcome": "incumbent", "is_scheduled": 1, "notes": "Disputed, protests followed"},
    {"country_name": "Chad", "country_code": "TCD", "election_date": "2024-05-06", "election_type": "Presidential", "outcome": "incumbent", "is_scheduled": 1, "notes": "Transition referendum"},
    {"country_name": "Croatia", "country_code": "HRV", "election_date": "2024-04-17", "election_type": "Legislative", "outcome": "incumbent", "is_scheduled": 0},
    {"country_name": "Belgium", "country_code": "BEL", "election_date": "2024-06-09", "election_type": "Legislative", "outcome": "challenger", "is_scheduled": 1},
    {"country_name": "Iceland", "country_code": "ISL", "election_date": "2024-11-30", "election_type": "Legislative", "outcome": "challenger", "is_scheduled": 0},
    {"country_name": "Mauritius", "country_code": "MUS", "election_date": "2024-11-10", "election_type": "Legislative", "outcome": "challenger", "is_scheduled": 1},
    {"country_name": "Namibia", "country_code": "NAM", "election_date": "2024-11-27", "election_type": "Presidential", "outcome": "incumbent", "is_scheduled": 1},
    
    # ============ 2025 ============
    {"country_name": "Belarus", "country_code": "BLR", "election_date": "2025-01-26", "election_type": "Presidential", "outcome": "incumbent", "is_scheduled": 1},
    {"country_name": "Germany", "country_code": "DEU", "election_date": "2025-02-23", "election_type": "Legislative", "outcome": "challenger", "is_scheduled": 0},
    {"country_name": "Ecuador", "country_code": "ECU", "election_date": "2025-02-09", "election_type": "Presidential", "outcome": "challenger", "is_scheduled": 1},
    {"country_name": "Cameroon", "country_code": "CMR", "election_date": "2025-02-09", "election_type": "Legislative", "outcome": "incumbent", "is_scheduled": 1},
    {"country_name": "Gabon", "country_code": "GAB", "election_date": "2025-03-15", "election_type": "Presidential", "outcome": "incumbent", "is_scheduled": 1, "notes": "Post-coup transitional vote"},
    {"country_name": "Canada", "country_code": "CAN", "election_date": "2025-04-28", "election_type": "Legislative", "outcome": "challenger", "is_scheduled": 0},
    {"country_name": "South Korea", "country_code": "KOR", "election_date": "2025-06-03", "election_type": "Presidential", "outcome": "challenger", "is_scheduled": 0, "notes": "Snap election after Yoon impeachment"},
    {"country_name": "Philippines", "country_code": "PHL", "election_date": "2025-05-12", "election_type": "Legislative", "outcome": "incumbent", "is_scheduled": 1},
    {"country_name": "Australia", "country_code": "AUS", "election_date": "2025-05-03", "election_type": "Legislative", "outcome": "incumbent", "is_scheduled": 1},
    {"country_name": "Poland", "country_code": "POL", "election_date": "2025-05-18", "election_type": "Presidential", "outcome": "incumbent", "is_scheduled": 1, "notes": "Nawrocki (allied PiS) vs Trzaskowski"},
    {"country_name": "Portugal", "country_code": "PRT", "election_date": "2025-03-09", "election_type": "Legislative", "outcome": "incumbent", "is_scheduled": 0},
    {"country_name": "Chile", "country_code": "CHL", "election_date": "2025-05-04", "election_type": "Legislative", "outcome": "challenger", "is_scheduled": 1, "notes": "Constitutional council"},
    {"country_name": "Togo", "country_code": "TGO", "election_date": "2025-04-13", "election_type": "Legislative", "outcome": "incumbent", "is_scheduled": 1},
    {"country_name": "Ivory Coast", "country_code": "CIV", "election_date": "2025-10-25", "election_type": "Presidential", "outcome": None, "is_scheduled": 1},
    {"country_name": "Singapore", "country_code": "SGP", "election_date": "2025-05-03", "election_type": "Legislative", "outcome": "incumbent", "is_scheduled": 0},
    
    # ============ 2026 (past / confirmed) ============
    {"country_name": "Bolivia", "country_code": "BOL", "election_date": "2025-08-17", "election_type": "Presidential", "outcome": None, "is_scheduled": 1},
    {"country_name": "Czech Republic", "country_code": "CZE", "election_date": "2025-10-04", "election_type": "Legislative", "outcome": None, "is_scheduled": 1},
    {"country_name": "Norway", "country_code": "NOR", "election_date": "2025-09-08", "election_type": "Legislative", "outcome": None, "is_scheduled": 1},
    {"country_name": "Tanzania", "country_code": "TZA", "election_date": "2025-10-15", "election_type": "Presidential", "outcome": None, "is_scheduled": 1},
    {"country_name": "Guinea", "country_code": "GIN", "election_date": "2025-12-31", "election_type": "Presidential", "outcome": None, "is_scheduled": 1, "notes": "Transition date, may slip"},
    
    # ============ 2026 (scheduled) ============
    {"country_name": "Colombia", "country_code": "COL", "election_date": "2026-05-31", "election_type": "Presidential", "outcome": "challenger", "is_scheduled": 1},
    {"country_name": "Brazil", "country_code": "BRA", "election_date": "2026-10-04", "election_type": "Presidential", "outcome": None, "is_scheduled": 1},
    {"country_name": "Japan", "country_code": "JPN", "election_date": "2026-07-25", "election_type": "Legislative", "outcome": None, "is_scheduled": 1, "notes": "House of Councillors"},
    {"country_name": "Mexico", "country_code": "MEX", "election_date": "2027-06-06", "election_type": "Legislative", "outcome": None, "is_scheduled": 1},
    {"country_name": "Philippines", "country_code": "PHL", "election_date": "2028-05-08", "election_type": "Presidential", "outcome": None, "is_scheduled": 1},
    
    # ============ 2027 ============
    {"country_name": "France", "country_code": "FRA", "election_date": "2027-04-11", "election_type": "Presidential", "outcome": None, "is_scheduled": 1},
    {"country_name": "France", "country_code": "FRA", "election_date": "2027-06-13", "election_type": "Legislative", "outcome": None, "is_scheduled": 1},
    {"country_name": "South Korea", "country_code": "KOR", "election_date": "2027-06-09", "election_type": "Legislative", "outcome": None, "is_scheduled": 1, "notes": "Date approximate"},
    {"country_name": "India", "country_code": "IND", "election_date": "2027-05-01", "election_type": "Legislative", "outcome": None, "is_scheduled": 1, "notes": "Date approximate, due by 2029"},
    {"country_name": "Kenya", "country_code": "KEN", "election_date": "2027-08-10", "election_type": "Presidential", "outcome": None, "is_scheduled": 1},
    {"country_name": "Angola", "country_code": "AGO", "election_date": "2027-08-15", "election_type": "Presidential", "outcome": None, "is_scheduled": 1},
    {"country_name": "Iran", "country_code": "IRN", "election_date": "2027-06-01", "election_type": "Legislative", "outcome": None, "is_scheduled": 1, "notes": "Date approximate"},
    {"country_name": "Germany", "country_code": "DEU", "election_date": "2029-09-01", "election_type": "Legislative", "outcome": None, "is_scheduled": 1, "notes": "Next regular cycle; moved to 2029"},
    
    # ============ 2028 ============
    {"country_name": "United States", "country_code": "USA", "election_date": "2028-11-07", "election_type": "Presidential", "outcome": None, "is_scheduled": 1},
    {"country_name": "Russia", "country_code": "RUS", "election_date": "2030-03-01", "election_type": "Presidential", "outcome": None, "is_scheduled": 1, "notes": "2030 cycle, out of range"},
    {"country_name": "Brazil", "country_code": "BRA", "election_date": "2026-10-04", "election_type": "Legislative", "outcome": None, "is_scheduled": 1},
    {"country_name": "Nigeria", "country_code": "NGA", "election_date": "2027-02-25", "election_type": "Presidential", "outcome": None, "is_scheduled": 1},
    {"country_name": "Turkey", "country_code": "TUR", "election_date": "2028-06-18", "election_type": "Presidential", "outcome": None, "is_scheduled": 1},
    {"country_name": "Argentina", "country_code": "ARG", "election_date": "2027-10-27", "election_type": "Legislative", "outcome": None, "is_scheduled": 1},
    {"country_name": "United Kingdom", "country_code": "GBR", "election_date": "2028-12-17", "election_type": "Legislative", "outcome": None, "is_scheduled": 1, "notes": "Must be held by Jan 2030"},
    {"country_name": "Indonesia", "country_code": "IDN", "election_date": "2029-02-14", "election_type": "Presidential", "outcome": None, "is_scheduled": 1, "notes": "2029 cycle"},
    {"country_name": "Chile", "country_code": "CHL", "election_date": "2025-11-16", "election_type": "Presidential", "outcome": None, "is_scheduled": 1},

    {"country_name": "Algeria", "country_code": "DZA", "election_date": "2026-07-02", "election_type": "Legislative", "outcome": None, "is_scheduled": 1},
    {"country_name": "Armenia", "country_code": "ARM", "election_date": "2026-06-07", "election_type": "Legislative", "outcome": None, "is_scheduled": 1},
    {"country_name": "Bosnia and Herzegovina", "country_code": "BIH", "election_date": "2026-10-04", "election_type": "Legislative", "outcome": None, "is_scheduled": 1},
    {"country_name": "Bosnia and Herzegovina", "country_code": "BIH", "election_date": "2026-10-04", "election_type": "Presidential", "outcome": None, "is_scheduled": 1},
    {"country_name": "Bulgaria", "country_code": "BGR", "election_date": "2026-11-01", "election_type": "Presidential", "outcome": None, "is_scheduled": 1},
    {"country_name": "Cape Verde", "country_code": "CPV", "election_date": "2026-11-15", "election_type": "Presidential", "outcome": None, "is_scheduled": 1},
    {"country_name": "Estonia", "country_code": "EST", "election_date": "2026-08-30", "election_type": "Presidential", "outcome": None, "is_scheduled": 1},
    {"country_name": "Fiji", "country_code": "FJI", "election_year": 2026, "election_type": "Legislative", "outcome": None, "is_scheduled": 1},
    {"country_name": "Haiti", "country_code": "HTI", "election_date": "2026-08-30", "election_type": "Legislative", "outcome": None, "is_scheduled": 1},
    {"country_name": "Haiti", "country_code": "HTI", "election_date": "2026-08-30", "election_type": "Presidential", "outcome": None, "is_scheduled": 1},
    {"country_name": "Israel", "country_code": "ISR", "election_date": "2026-10-27", "election_type": "Legislative", "outcome": None, "is_scheduled": 1},
    {"country_name": "Kosovo", "country_code": "XKX", "election_date": "2026-06-07", "election_type": "Legislative", "outcome": None, "is_scheduled": 1},
    {"country_name": "Kosovo", "country_code": "XKX", "election_year": 2026, "election_type": "Presidential", "outcome": None, "is_scheduled": 1},
    {"country_name": "Latvia", "country_code": "LVA", "election_date": "2026-10-03", "election_type": "Legislative", "outcome": None, "is_scheduled": 1},
    {"country_name": "Malta", "country_code": "MLT", "election_date": "2026-05-30", "election_type": "Legislative", "outcome": None, "is_scheduled": 1},
    {"country_name": "Morocco", "country_code": "MAR", "election_date": "2026-09-23", "election_type": "Legislative", "outcome": None, "is_scheduled": 1},
    {"country_name": "New Zealand", "country_code": "NZL", "election_date": "2026-11-07", "election_type": "Legislative", "outcome": None, "is_scheduled": 1},
    {"country_name": "Russia", "country_code": "RUS", "election_date": "2026-09-20", "election_type": "Legislative", "outcome": None, "is_scheduled": 1},
    {"country_name": "San Marino", "country_code": "SMR", "election_date": "2026-09-01", "election_type": "Presidential", "outcome": None, "is_scheduled": 1},
    {"country_name": "Sao Tome and Principe", "country_code": "STP", "election_date": "2026-09-27", "election_type": "Legislative", "outcome": None, "is_scheduled": 1},
    {"country_name": "Sao Tome and Principe", "country_code": "STP", "election_date": "2026-07-19", "election_type": "Presidential", "outcome": None, "is_scheduled": 1},
    {"country_name": "Somalia", "country_code": "SOM", "election_year": 2026, "election_type": "Legislative", "outcome": None, "is_scheduled": 1},
    {"country_name": "South Sudan", "country_code": "SSD", "election_date": "2026-12-22", "election_type": "Legislative", "outcome": None, "is_scheduled": 1},
    {"country_name": "South Sudan", "country_code": "SSD", "election_date": "2026-12-22", "election_type": "Presidential", "outcome": None, "is_scheduled": 1},
    {"country_name": "Sweden", "country_code": "SWE", "election_date": "2026-09-13", "election_type": "Legislative", "outcome": None, "is_scheduled": 1},
    {"country_name": "Switzerland", "country_code": "CHE", "election_date": "2026-12-01", "election_type": "Presidential", "outcome": None, "is_scheduled": 1},
    {"country_name": "United States", "country_code": "USA", "election_date": "2026-11-03", "election_type": "Legislative", "outcome": None, "is_scheduled": 1},
    {"country_name": "Zambia", "country_code": "ZMB", "election_date": "2026-08-13", "election_type": "Legislative", "outcome": None, "is_scheduled": 1},
    {"country_name": "Zambia", "country_code": "ZMB", "election_date": "2026-08-13", "election_type": "Presidential", "outcome": None, "is_scheduled": 1},
    {"country_name": "Albania", "country_code": "ALB", "election_year": 2027, "election_type": "Presidential", "outcome": None, "is_scheduled": 1},
    {"country_name": "Andorra", "country_code": "AND", "election_year": 2027, "election_type": "Legislative", "outcome": None, "is_scheduled": 1},
    {"country_name": "Angola", "country_code": "AGO", "election_year": 2027, "election_type": "Legislative", "outcome": None, "is_scheduled": 1},
    {"country_name": "Argentina", "country_code": "ARG", "election_date": "2027-10-24", "election_type": "Presidential", "outcome": None, "is_scheduled": 1},
    {"country_name": "Bahrain", "country_code": "BHR", "election_date": "2027-12-01", "election_type": "Legislative", "outcome": None, "is_scheduled": 1},
    {"country_name": "Burundi", "country_code": "BDI", "election_year": 2027, "election_type": "Presidential", "outcome": None, "is_scheduled": 1},
    {"country_name": "China", "country_code": "CHN", "election_year": 2027, "election_type": "Legislative", "outcome": None, "is_scheduled": 1},
    {"country_name": "Dominica", "country_code": "DMA", "election_date": "2027-12-01", "election_type": "Legislative", "outcome": None, "is_scheduled": 1},
    {"country_name": "El Salvador", "country_code": "SLV", "election_date": "2027-02-28", "election_type": "Legislative", "outcome": None, "is_scheduled": 1},
    {"country_name": "El Salvador", "country_code": "SLV", "election_date": "2027-02-28", "election_type": "Presidential", "outcome": None, "is_scheduled": 1},
    {"country_name": "Equatorial Guinea", "country_code": "GNQ", "election_year": 2027, "election_type": "Legislative", "outcome": None, "is_scheduled": 1},
    {"country_name": "Estonia", "country_code": "EST", "election_year": 2027, "election_type": "Legislative", "outcome": None, "is_scheduled": 1},
    {"country_name": "Fiji", "country_code": "FJI", "election_date": "2027-10-01", "election_type": "Presidential", "outcome": None, "is_scheduled": 1},
    {"country_name": "Finland", "country_code": "FIN", "election_date": "2027-04-18", "election_type": "Legislative", "outcome": None, "is_scheduled": 1},
    {"country_name": "Germany", "country_code": "DEU", "election_date": "2027-02-01", "election_type": "Presidential", "outcome": None, "is_scheduled": 1},
    {"country_name": "Greece", "country_code": "GRC", "election_year": 2027, "election_type": "Legislative", "outcome": None, "is_scheduled": 1},
    {"country_name": "Grenada", "country_code": "GRD", "election_year": 2027, "election_type": "Legislative", "outcome": None, "is_scheduled": 1},
    {"country_name": "Guatemala", "country_code": "GTM", "election_date": "2027-06-01", "election_type": "Legislative", "outcome": None, "is_scheduled": 1},
    {"country_name": "Guatemala", "country_code": "GTM", "election_date": "2027-06-01", "election_type": "Presidential", "outcome": None, "is_scheduled": 1},
    {"country_name": "Hong Kong", "country_code": "HKG", "election_date": "2027-03-28", "election_type": "Presidential", "outcome": None, "is_scheduled": 1},
    {"country_name": "India", "country_code": "IND", "election_year": 2027, "election_type": "Presidential", "outcome": None, "is_scheduled": 1},
    {"country_name": "Italy", "country_code": "ITA", "election_date": "2027-12-01", "election_type": "Legislative", "outcome": None, "is_scheduled": 1},
    {"country_name": "Kenya", "country_code": "KEN", "election_year": 2027, "election_type": "Legislative", "outcome": None, "is_scheduled": 1},
    {"country_name": "Kyrgyzstan", "country_code": "KGZ", "election_year": 2027, "election_type": "Presidential", "outcome": None, "is_scheduled": 1},
    {"country_name": "Latvia", "country_code": "LVA", "election_year": 2027, "election_type": "Presidential", "outcome": None, "is_scheduled": 1},
    {"country_name": "Lesotho", "country_code": "LSO", "election_year": 2027, "election_type": "Legislative", "outcome": None, "is_scheduled": 1},
    {"country_name": "Marshall Islands", "country_code": "MHL", "election_year": 2027, "election_type": "Legislative", "outcome": None, "is_scheduled": 1},
    {"country_name": "Micronesia", "country_code": "FSM", "election_year": 2027, "election_type": "Legislative", "outcome": None, "is_scheduled": 1},
    {"country_name": "Micronesia", "country_code": "FSM", "election_year": 2027, "election_type": "Presidential", "outcome": None, "is_scheduled": 1},
    {"country_name": "Mongolia", "country_code": "MNG", "election_date": "2027-06-01", "election_type": "Presidential", "outcome": None, "is_scheduled": 1},
    {"country_name": "Montenegro", "country_code": "MNE", "election_year": 2027, "election_type": "Legislative", "outcome": None, "is_scheduled": 1},
    {"country_name": "Nicaragua", "country_code": "NIC", "election_date": "2027-11-01", "election_type": "Legislative", "outcome": None, "is_scheduled": 1},
    {"country_name": "Nicaragua", "country_code": "NIC", "election_date": "2027-11-01", "election_type": "Presidential", "outcome": None, "is_scheduled": 1},
    {"country_name": "Nigeria", "country_code": "NGA", "election_date": "2027-02-01", "election_type": "Legislative", "outcome": None, "is_scheduled": 1},
    {"country_name": "Oman", "country_code": "OMN", "election_date": "2027-10-01", "election_type": "Legislative", "outcome": None, "is_scheduled": 1},
    {"country_name": "Papua New Guinea", "country_code": "PNG", "election_year": 2027, "election_type": "Legislative", "outcome": None, "is_scheduled": 1},
    {"country_name": "Poland", "country_code": "POL", "election_year": 2027, "election_type": "Legislative", "outcome": None, "is_scheduled": 1},
    {"country_name": "Republic of the Congo", "country_code": "COG", "election_year": 2027, "election_type": "Legislative", "outcome": None, "is_scheduled": 1},
    {"country_name": "Saint Kitts and Nevis", "country_code": "KNA", "election_year": 2027, "election_type": "Legislative", "outcome": None, "is_scheduled": 1},
    {"country_name": "Samoa", "country_code": "WSM", "election_year": 2027, "election_type": "Presidential", "outcome": None, "is_scheduled": 1},
    {"country_name": "Serbia", "country_code": "SRB", "election_year": 2027, "election_type": "Legislative", "outcome": None, "is_scheduled": 1},
    {"country_name": "Serbia", "country_code": "SRB", "election_year": 2027, "election_type": "Presidential", "outcome": None, "is_scheduled": 1},
    {"country_name": "Slovakia", "country_code": "SVK", "election_year": 2027, "election_type": "Legislative", "outcome": None, "is_scheduled": 1},
    {"country_name": "Slovenia", "country_code": "SVN", "election_year": 2027, "election_type": "Presidential", "outcome": None, "is_scheduled": 1},
    {"country_name": "Somalia", "country_code": "SOM", "election_year": 2027, "election_type": "Presidential", "outcome": None, "is_scheduled": 1},
    {"country_name": "Somaliland", "country_code": "SML", "election_date": "2027-03-01", "election_type": "Legislative", "outcome": None, "is_scheduled": 1},
    {"country_name": "Spain", "country_code": "ESP", "election_year": 2027, "election_type": "Legislative", "outcome": None, "is_scheduled": 1},
    {"country_name": "Switzerland", "country_code": "CHE", "election_date": "2027-10-01", "election_type": "Legislative", "outcome": None, "is_scheduled": 1},
    {"country_name": "Tajikistan", "country_code": "TJK", "election_year": 2027, "election_type": "Presidential", "outcome": None, "is_scheduled": 1},
    {"country_name": "Timor-Leste", "country_code": "TLS", "election_year": 2027, "election_type": "Presidential", "outcome": None, "is_scheduled": 1},
    {"country_name": "Tunisia", "country_code": "TUN", "election_year": 2027, "election_type": "Legislative", "outcome": None, "is_scheduled": 1},
    {"country_name": "United Arab Emirates", "country_code": "ARE", "election_year": 2027, "election_type": "Legislative", "outcome": None, "is_scheduled": 1},
    {"country_name": "Vanuatu", "country_code": "VUT", "election_date": "2027-07-01", "election_type": "Presidential", "outcome": None, "is_scheduled": 1},
    {"country_name": "Australia", "country_code": "AUS", "election_year": 2028, "election_type": "Legislative", "outcome": None, "is_scheduled": 1},
    {"country_name": "Austria", "country_code": "AUT", "election_year": 2028, "election_type": "Presidential", "outcome": None, "is_scheduled": 1},
    {"country_name": "Bangladesh", "country_code": "BGD", "election_year": 2028, "election_type": "Presidential", "outcome": None, "is_scheduled": 1},
    {"country_name": "Belarus", "country_code": "BLR", "election_year": 2028, "election_type": "Legislative", "outcome": None, "is_scheduled": 1},
    {"country_name": "Cambodia", "country_code": "KHM", "election_year": 2028, "election_type": "Legislative", "outcome": None, "is_scheduled": 1},
    {"country_name": "China", "country_code": "CHN", "election_year": 2028, "election_type": "Presidential", "outcome": None, "is_scheduled": 1},
    {"country_name": "Croatia", "country_code": "HRV", "election_year": 2028, "election_type": "Legislative", "outcome": None, "is_scheduled": 1},
    {"country_name": "Cuba", "country_code": "CUB", "election_year": 2028, "election_type": "Legislative", "outcome": None, "is_scheduled": 1},
    {"country_name": "Cuba", "country_code": "CUB", "election_year": 2028, "election_type": "Presidential", "outcome": None, "is_scheduled": 1},
    {"country_name": "Cyprus", "country_code": "CYP", "election_year": 2028, "election_type": "Presidential", "outcome": None, "is_scheduled": 1},
    {"country_name": "Democratic Republic of the Congo", "country_code": "COD", "election_date": "2028-12-01", "election_type": "Legislative", "outcome": None, "is_scheduled": 1},
    {"country_name": "Democratic Republic of the Congo", "country_code": "COD", "election_date": "2028-12-01", "election_type": "Presidential", "outcome": None, "is_scheduled": 1},
    {"country_name": "Djibouti", "country_code": "DJI", "election_date": "2028-02-01", "election_type": "Legislative", "outcome": None, "is_scheduled": 1},
    {"country_name": "Dominica", "country_code": "DMA", "election_year": 2028, "election_type": "Presidential", "outcome": None, "is_scheduled": 1},
    {"country_name": "Dominican Republic", "country_code": "DOM", "election_year": 2028, "election_type": "Legislative", "outcome": None, "is_scheduled": 1},
    {"country_name": "Dominican Republic", "country_code": "DOM", "election_year": 2028, "election_type": "Presidential", "outcome": None, "is_scheduled": 1},
    {"country_name": "Eswatini", "country_code": "SWZ", "election_year": 2028, "election_type": "Legislative", "outcome": None, "is_scheduled": 1},
    {"country_name": "Georgia", "country_code": "GEO", "election_date": "2028-10-01", "election_type": "Legislative", "outcome": None, "is_scheduled": 1},
    {"country_name": "Ghana", "country_code": "GHA", "election_date": "2028-12-07", "election_type": "Legislative", "outcome": None, "is_scheduled": 1},
    {"country_name": "Ghana", "country_code": "GHA", "election_date": "2028-12-07", "election_type": "Presidential", "outcome": None, "is_scheduled": 1},
    {"country_name": "Iceland", "country_code": "ISL", "election_year": 2028, "election_type": "Legislative", "outcome": None, "is_scheduled": 1},
    {"country_name": "Iceland", "country_code": "ISL", "election_year": 2028, "election_type": "Presidential", "outcome": None, "is_scheduled": 1},
    {"country_name": "Iran", "country_code": "IRN", "election_year": 2028, "election_type": "Presidential", "outcome": None, "is_scheduled": 1},
    {"country_name": "Israel", "country_code": "ISR", "election_year": 2028, "election_type": "Presidential", "outcome": None, "is_scheduled": 1},
    {"country_name": "Jordan", "country_code": "JOR", "election_year": 2028, "election_type": "Legislative", "outcome": None, "is_scheduled": 1},
    {"country_name": "Kazakhstan", "country_code": "KAZ", "election_year": 2028, "election_type": "Legislative", "outcome": None, "is_scheduled": 1},
    {"country_name": "Kiribati", "country_code": "KIR", "election_year": 2028, "election_type": "Legislative", "outcome": None, "is_scheduled": 1},
    {"country_name": "Kiribati", "country_code": "KIR", "election_year": 2028, "election_type": "Presidential", "outcome": None, "is_scheduled": 1},
    {"country_name": "Kuwait", "country_code": "KWT", "election_year": 2028, "election_type": "Legislative", "outcome": None, "is_scheduled": 1},
    {"country_name": "Lebanon", "country_code": "LBN", "election_date": "2028-05-01", "election_type": "Legislative", "outcome": None, "is_scheduled": 1},
    {"country_name": "Lithuania", "country_code": "LTU", "election_year": 2028, "election_type": "Legislative", "outcome": None, "is_scheduled": 1},
    {"country_name": "Luxembourg", "country_code": "LUX", "election_date": "2028-10-01", "election_type": "Legislative", "outcome": None, "is_scheduled": 1},
    {"country_name": "Madagascar", "country_code": "MDG", "election_date": "2028-11-01", "election_type": "Presidential", "outcome": None, "is_scheduled": 1},
    {"country_name": "Malaysia", "country_code": "MYS", "election_year": 2028, "election_type": "Legislative", "outcome": None, "is_scheduled": 1},
    {"country_name": "Maldives", "country_code": "MDV", "election_date": "2028-09-01", "election_type": "Presidential", "outcome": None, "is_scheduled": 1},
    {"country_name": "Marshall Islands", "country_code": "MHL", "election_year": 2028, "election_type": "Presidential", "outcome": None, "is_scheduled": 1},
    {"country_name": "Mauritania", "country_code": "MRT", "election_year": 2028, "election_type": "Legislative", "outcome": None, "is_scheduled": 1},
    {"country_name": "Moldova", "country_code": "MDA", "election_date": "2028-10-01", "election_type": "Presidential", "outcome": None, "is_scheduled": 1},
    {"country_name": "Monaco", "country_code": "MCO", "election_year": 2028, "election_type": "Legislative", "outcome": None, "is_scheduled": 1},
    {"country_name": "Mongolia", "country_code": "MNG", "election_date": "2028-06-01", "election_type": "Legislative", "outcome": None, "is_scheduled": 1},
    {"country_name": "Montenegro", "country_code": "MNE", "election_year": 2028, "election_type": "Presidential", "outcome": None, "is_scheduled": 1},
    {"country_name": "Nauru", "country_code": "NRU", "election_year": 2028, "election_type": "Legislative", "outcome": None, "is_scheduled": 1},
    {"country_name": "Nauru", "country_code": "NRU", "election_year": 2028, "election_type": "Presidential", "outcome": None, "is_scheduled": 1},
    {"country_name": "Nepal", "country_code": "NPL", "election_year": 2028, "election_type": "Presidential", "outcome": None, "is_scheduled": 1},
    {"country_name": "North Macedonia", "country_code": "MKD", "election_year": 2028, "election_type": "Legislative", "outcome": None, "is_scheduled": 1},
    {"country_name": "Palau", "country_code": "PLW", "election_date": "2028-11-07", "election_type": "Legislative", "outcome": None, "is_scheduled": 1},
    {"country_name": "Palau", "country_code": "PLW", "election_date": "2028-11-07", "election_type": "Presidential", "outcome": None, "is_scheduled": 1},
    {"country_name": "Paraguay", "country_code": "PRY", "election_year": 2028, "election_type": "Legislative", "outcome": None, "is_scheduled": 1},
    {"country_name": "Paraguay", "country_code": "PRY", "election_year": 2028, "election_type": "Presidential", "outcome": None, "is_scheduled": 1},
    {"country_name": "Philippines", "country_code": "PHL", "election_date": "2028-05-08", "election_type": "Legislative", "outcome": None, "is_scheduled": 1},
    {"country_name": "Romania", "country_code": "ROU", "election_year": 2028, "election_type": "Legislative", "outcome": None, "is_scheduled": 1},
    {"country_name": "Sierra Leone", "country_code": "SLE", "election_year": 2028, "election_type": "Legislative", "outcome": None, "is_scheduled": 1},
    {"country_name": "Sierra Leone", "country_code": "SLE", "election_year": 2028, "election_type": "Presidential", "outcome": None, "is_scheduled": 1},
    {"country_name": "Solomon Islands", "country_code": "SLB", "election_year": 2028, "election_type": "Legislative", "outcome": None, "is_scheduled": 1},
    {"country_name": "Taiwan", "country_code": "TWN", "election_date": "2028-01-15", "election_type": "Legislative", "outcome": None, "is_scheduled": 1},
    {"country_name": "Taiwan", "country_code": "TWN", "election_date": "2028-01-15", "election_type": "Presidential", "outcome": None, "is_scheduled": 1},
    {"country_name": "Timor-Leste", "country_code": "TLS", "election_year": 2028, "election_type": "Legislative", "outcome": None, "is_scheduled": 1},
    {"country_name": "Trinidad and Tobago", "country_code": "TTO", "election_year": 2028, "election_type": "Presidential", "outcome": None, "is_scheduled": 1},
    {"country_name": "Turkmenistan", "country_code": "TKM", "election_year": 2028, "election_type": "Legislative", "outcome": None, "is_scheduled": 1},
    {"country_name": "Tuvalu", "country_code": "TUV", "election_year": 2028, "election_type": "Legislative", "outcome": None, "is_scheduled": 1},
    {"country_name": "Ukraine", "country_code": "UKR", "election_year": 2028, "election_type": "Legislative", "outcome": None, "is_scheduled": 1},
    {"country_name": "Ukraine", "country_code": "UKR", "election_year": 2028, "election_type": "Presidential", "outcome": None, "is_scheduled": 1},
    {"country_name": "Zimbabwe", "country_code": "ZWE", "election_year": 2028, "election_type": "Legislative", "outcome": None, "is_scheduled": 1},
    {"country_name": "Zimbabwe", "country_code": "ZWE", "election_year": 2028, "election_type": "Presidential", "outcome": None, "is_scheduled": 1},
]


# Mapping of (country_code, election_date_or_year) -> (incumbent_ran, has_successor)
# Coded based on historical verification of executive candidacy
PRESIDENTIAL_CANDIDACY = {
    # 2021
    ("UGA", "2021-01-14"): (1, 0), # Yoweri Museveni ran
    ("PRT", "2021-01-24"): (1, 0), # Marcelo Rebelo de Sousa ran
    ("ECU", "2021-02-07"): (0, 1), # Lenín Moreno did not run, Guillermo Lasso ran
    ("NER", "2021-02-21"): (0, 1), # Mahamadou Issoufou term-limited, Mohamed Bazoum ran
    ("COG", "2021-03-21"): (1, 0), # Denis Sassou Nguesso ran
    ("DJI", "2021-04-09"): (1, 0), # Ismaïl Omar Guelleh ran
    ("BEN", "2021-04-11"): (1, 0), # Patrice Talon ran
    ("TCD", "2021-04-11"): (1, 0), # Idriss Déby ran
    ("PER", "2021-04-11"): (0, 0), # Francisco Sagasti did not run, no successor
    ("SYR", "2021-05-26"): (1, 0), # Bashar al-Assad ran
    ("MNG", "2021-06-09"): (0, 1), # Battulga term-limited, Khürelsükh ran
    ("IRN", "2021-06-18"): (0, 1), # Rouhani term-limited, Raisi ran
    ("ZMB", "2021-08-12"): (1, 0), # Edgar Lungu ran
    ("NIC", "2021-11-07"): (1, 0), # Daniel Ortega ran
    ("HND", "2021-11-28"): (0, 1), # Juan Orlando Hernández term-limited, Nasry Asfura ran
    ("CHL", "2021-12-19"): (0, 1), # Sebastián Piñera term-limited, Sebastian Sichel ran
    ("GMB", "2021-12-04"): (1, 0), # Adama Barrow ran
    # 2022
    ("ITA", "2022-01-24"): (1, 0), # Sergio Mattarella ran
    ("CRI", "2022-02-06"): (0, 1), # Alvarado Quesada term-limited, Welmer Ramos ran
    ("KOR", "2022-03-09"): (0, 1), # Moon Jae-in term-limited, Lee Jae-myung ran
    ("SRB", "2022-04-03"): (1, 0), # Aleksandar Vučić ran
    ("FRA", "2022-04-24"): (1, 0), # Emmanuel Macron ran
    ("PHL", "2022-05-09"): (0, 1), # Duterte term-limited, Marcos ran (successor)
    ("COL", "2022-06-19"): (0, 1), # Iván Duque term-limited, Federico Gutiérrez ran
    ("KEN", "2022-08-09"): (0, 1), # Kenyatta term-limited, Odinga ran (successor)
    ("AGO", "2022-08-24"): (1, 0), # João Lourenço ran
    ("BRA", "2022-10-30"): (1, 0), # Jair Bolsonaro ran
    # 2023
    ("NGA", "2023-02-25"): (0, 1), # Buhari term-limited, Tinubu ran
    ("TUR", "2023-05-28"): (1, 0), # Erdoğan ran
    ("GTM", "2023-08-20"): (0, 1), # Giammattei term-limited, Manuel Conde ran
    ("ECU", "2023-10-15"): (0, 0), # Lasso did not run, snap election
    ("ARG", "2023-11-19"): (0, 1), # Fernández did not run, Massa ran
    ("LBR", "2023-11-14"): (1, 0), # George Weah ran
    ("COD", "2023-12-20"): (1, 0), # Félix Tshisekedi ran
    ("MDG", "2023-11-16"): (1, 0), # Andry Rajoelina ran
    ("ZWE", "2023-08-23"): (1, 0), # Emmerson Mnangagwa ran
    ("PRY", "2023-04-30"): (0, 1), # Abdo Benítez term-limited, Santiago Peña ran
    ("SLE", "2023-06-24"): (1, 0), # Julius Maada Bio ran
    # 2024
    ("TWN", "2024-01-13"): (0, 1), # Tsai Ing-wen term-limited, Lai Ching-te ran
    ("FIN", "2024-01-28"): (0, 1), # Niinistö term-limited, Stubb ran
    ("SLV", "2024-02-04"): (1, 0), # Nayib Bukele ran
    ("IDN", "2024-02-14"): (0, 1), # Jokowi term-limited, Prabowo ran
    ("SEN", "2024-03-24"): (0, 1), # Macky Sall did not run, Amadou Ba ran
    ("RUS", "2024-03-17"): (1, 0), # Vladimir Putin ran
    ("MEX", "2024-06-02"): (0, 1), # AMLO term-limited, Sheinbaum ran
    ("IRN", "2024-07-05"): (0, 1), # Raisi died, successor ran
    ("VEN", "2024-07-28"): (1, 0), # Maduro ran
    ("RWA", "2024-07-15"): (1, 0), # Kagame ran
    ("LKA", "2024-09-21"): (1, 0), # Wickremesinghe ran
    ("TUN", "2024-10-06"): (1, 0), # Saied ran
    ("USA", "2024-11-05"): (0, 1), # Biden withdrew, Harris ran
    ("URY", "2024-11-24"): (0, 1), # Lacalle Pou term-limited, Delgado ran
    ("PAN", "2024-05-05"): (0, 1), # Cortizo term-limited, Carrizo ran
    ("DOM", "2024-05-19"): (1, 0), # Abinader ran
    ("GHA", "2024-12-07"): (0, 1), # Akufo-Addo term-limited, Bawumia ran
    ("MOZ", "2024-10-09"): (0, 1), # Nyusi term-limited, Chapo ran
    ("TCD", "2024-05-06"): (1, 0), # Mahamat Déby ran
    ("NAM", "2024-11-27"): (0, 1), # Mbumba did not run, Netumbo Nandi-Ndaitwah ran
    # 2025
    ("BLR", "2025-01-26"): (1, 0), # Lukashenko ran
    ("ECU", "2025-02-09"): (1, 0), # Noboa ran
    ("GAB", "2025-03-15"): (1, 0), # Oligui Nguema ran
    ("POL", "2025-05-18"): (0, 1), # Duda term-limited, successor ran
    ("KOR", "2025-06-03"): (1, 0), # Yoon Suk-yeol was incumbent
    # 2026
    ("COL", "2026-05-31"): (0, 1), # Petro term-limited, Iván Cepeda ran
    ("BRA", "2026-10-04"): (1, 0), # Lula running
    ("BGR", "2026-11-01"): (0, 1), # Radev term-limited, successor runs
    ("CPV", "2026-11-15"): (1, 0), # Neves eligible
    ("EST", "2026-08-30"): (1, 0), # Karis eligible
    ("HTI", "2026-08-30"): (0, 0), # No incumbent
    ("SSD", "2026-12-22"): (1, 0), # Salva Kiir running
    ("STP", "2026-07-19"): (1, 0), # Vila Nova eligible
    ("ZMB", "2026-08-13"): (1, 0), # Hichilema running
    # 2027
    ("ARG", "2027-10-24"): (1, 0), # Milei eligible
    ("SLV", "2027-02-28"): (0, 1), # Bukele term-limited
    ("GTM", "2027-06-01"): (0, 1), # Arévalo term-limited
    ("MNG", "2027-06-01"): (0, 1), # Khürelsükh term-limited
    ("NIC", "2027-11-01"): (1, 0), # Ortega running
    ("SRB", "2027"): (0, 1), # Vučić term-limited
    # 2028
    ("COD", "2028-12-01"): (0, 1), # Regular presidential
    ("GHA", "2028-12-07"): (0, 1), # Akufo-Addo term-limited, Bawumia ran
    ("MDG", "2028-11-01"): (0, 1), # Rajoelina term-limited
    ("MDV", "2028-09-01"): (1, 0), # Muizzu running
    ("MDA", "2028-10-01"): (0, 1), # Sandu term-limited
    ("PLW", "2028-11-07"): (1, 0), # Whipps running
    ("TWN", "2028-01-15"): (1, 0)  # Lai Ching-te running
}


def ingest_upcoming_elections():
    """Populate the upcoming_elections table with elections not covered by NELDA."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Get valid country codes
    cursor.execute("SELECT country_code FROM countries;")
    valid_codes = {row[0] for row in cursor.fetchall()}
    
    # Get existing NELDA elections to avoid duplicates
    cursor.execute("SELECT country_code, election_year, election_type FROM elections;")
    existing_nelda = {(r[0], r[1], r[2]) for r in cursor.fetchall()}
    
    # Get existing upcoming elections
    cursor.execute("SELECT country_code, election_year, election_type FROM upcoming_elections;")
    existing_upcoming = {(r[0], r[1], r[2]) for r in cursor.fetchall()}
    
    inserted = 0
    skipped_no_code = 0
    skipped_duplicate = 0
    skipped_out_of_range = 0
    
    for election in ELECTIONS_2021_2028:
        code = election["country_code"]
        
        # Skip entries without valid country codes
        if code not in valid_codes:
            skipped_no_code += 1
            continue
        
        year = int(election["election_date"][:4]) if election.get("election_date") else election.get("election_year", 0)
        
        # Filter to 2021-2028 range
        if year < 2021 or year > 2028:
            skipped_out_of_range += 1
            continue
        
        el_type = election["election_type"]
        
        # Check for duplicates
        if (code, year, el_type) in existing_nelda or (code, year, el_type) in existing_upcoming:
            skipped_duplicate += 1
            continue
        
        incumbent_ran = None
        has_successor = None
        
        # Get executive candidacy features
        if el_type == "Presidential":
            key = (code, election.get("election_date"))
            if key in PRESIDENTIAL_CANDIDACY:
                incumbent_ran, has_successor = PRESIDENTIAL_CANDIDACY[key]
            else:
                # Year-based fallback search
                for k, v in PRESIDENTIAL_CANDIDACY.items():
                    if k[0] == code and str(k[1]).startswith(str(year)):
                        incumbent_ran, has_successor = v
                        break
        
        cursor.execute("""
            INSERT INTO upcoming_elections (country_code, election_year, election_date, election_type, is_scheduled, actual_outcome, notes, incumbent_ran, has_successor)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, (
            code,
            year,
            election.get("election_date"),
            el_type,
            election.get("is_scheduled", 1),
            election.get("outcome"),
            election.get("notes"),
            incumbent_ran,
            has_successor
        ))
        existing_upcoming.add((code, year, el_type))
        inserted += 1
    
    conn.commit()
    
    cursor.execute("SELECT COUNT(*) FROM upcoming_elections;")
    total = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM upcoming_elections WHERE actual_outcome IS NOT NULL;")
    known = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM upcoming_elections WHERE actual_outcome IS NULL;")
    unknown = cursor.fetchone()[0]
    
    print(f"\nUpcoming Elections Ingestion Complete:")
    print(f"  Inserted:           {inserted}")
    print(f"  Skipped (no code):  {skipped_no_code}")
    print(f"  Skipped (duplicate):{skipped_duplicate}")
    print(f"  Skipped (range):    {skipped_out_of_range}")
    print(f"  Total in table:     {total}")
    print(f"  Known outcomes:     {known}")
    print(f"  Unknown (to predict): {unknown}")
    
    conn.close()


def ingest_gdelt_for_upcoming():
    """
    Trigger GDELT ingestion for the 180-day windows before each upcoming election.
    Reuses the existing GDELT parsing infrastructure but only processes cached files.
    """
    # Import from the main GDELT ingest
    from ingest_gdelt import load_country_mappings, parse_and_filter_zip, insert_events, MIN_MENTIONS
    
    conn = get_connection()
    cursor = conn.cursor()
    
    # Get upcoming elections with dates
    cursor.execute("SELECT upcoming_id, country_code, election_date FROM upcoming_elections WHERE election_date IS NOT NULL;")
    elections = cursor.fetchall()
    
    fips_to_iso, iso_to_fips = load_country_mappings()
    
    # Build date-based FIPS lookup for election windows
    active_fips_by_date = {}
    for up_id, country_code, date_str in elections:
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
    
    if not active_fips_by_date:
        print("No upcoming election windows to process for GDELT.")
        conn.close()
        return
    
    # Find required archive files
    required_files = {}
    for date_key in active_fips_by_date.keys():
        try:
            dt = datetime.strptime(date_key, "%Y%m%d")
        except ValueError:
            continue
        
        if dt >= datetime(2013, 4, 1):
            filename = f"{date_key}.export.CSV.zip"
        elif dt >= datetime(2006, 1, 1):
            filename = f"{dt.year}{dt.month:02d}.zip"
        else:
            filename = f"{dt.year}.zip"
        
        cache_path = os.path.join(CACHE_DIR, filename)
        if os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
            required_files[filename] = cache_path
            
    # Ensure tracking table exists
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS processed_gdelt_files (
            filename TEXT PRIMARY KEY,
            processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()
    
    # Ask user whether to parse all files or only new/unprocessed ones
    choice = 'n'
    if sys.stdin.isatty():
        try:
            user_input = input("Parse all cached GDELT files (a) or only new/unprocessed files (n)? [default: n]: ").strip().lower()
            if user_input in ['a', 'all']:
                choice = 'a'
        except (Exception, KeyboardInterrupt):
            pass
            
    if choice == 'n':
        cursor.execute("SELECT filename FROM processed_gdelt_files;")
        processed_files = {r[0] for r in cursor.fetchall()}
        filtered_files = {fname: path for fname, path in required_files.items() if fname not in processed_files}
        print(f"\nFound {len(required_files)} cached files. {len(processed_files)} have already been processed.")
        print(f"Processing remaining {len(filtered_files)} unprocessed files...")
    else:
        cursor.execute("DELETE FROM processed_gdelt_files;")
        conn.commit()
        filtered_files = required_files
        print(f"\nProcessing all {len(filtered_files)} cached GDELT archives...")
        
    if not filtered_files:
        print("All required GDELT archives have already been processed.")
        conn.close()
        return
    
    total_events = 0
    for i, (fname, cache_path) in enumerate(sorted(filtered_files.items())):
        events = parse_and_filter_zip(cache_path, active_fips_by_date, fips_to_iso, MIN_MENTIONS)
        if events:
            insert_events(conn, events)
            total_events += len(events)
            
        # Track that this file has been processed
        cursor.execute("INSERT OR REPLACE INTO processed_gdelt_files (filename) VALUES (?);", (fname,))
        conn.commit()
        
        if (i + 1) % 100 == 0 or (i + 1) == len(filtered_files):
            print(f"  Processed {i+1}/{len(filtered_files)} files ({total_events:,} events)")
    
    print(f"GDELT ingestion for upcoming elections complete: {total_events:,} events added.")
    conn.close()


if __name__ == "__main__":
    ingest_upcoming_elections()
    
    # Only try GDELT if cache exists
    if os.path.exists(CACHE_DIR) and len(os.listdir(CACHE_DIR)) > 0:
        print("\nProcessing GDELT data for upcoming election windows...")
        ingest_gdelt_for_upcoming()
    else:
        print("\nNo GDELT cache found. Run bulk_download_gdelt.py first to download GDELT archives.")
