import os
import sqlite3
import sys

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "elections.db")

def get_connection():
    """Returns a connection to the SQLite database with foreign keys enabled."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def init_db():
    print(f"Initializing database at: {DB_PATH}")
    
    # Drop existing database to ensure clean schema migration
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print("Existing database dropped for schema migration.")
        
    conn = get_connection()
    cursor = conn.cursor()

    # 1. Countries Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS countries (
        country_code TEXT PRIMARY KEY,
        country_name TEXT NOT NULL,
        region TEXT,
        latitude REAL,
        longitude REAL,
        is_oil_exporter INTEGER DEFAULT 0
    );
    """)

    # 2. Elections Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS elections (
        election_id INTEGER PRIMARY KEY AUTOINCREMENT,
        country_code TEXT NOT NULL,
        election_year INTEGER NOT NULL,
        election_date TEXT,
        election_type TEXT,
        is_scheduled INTEGER DEFAULT 1,
        is_fraudulent INTEGER,
        incumbent_ran INTEGER,
        has_successor INTEGER,
        FOREIGN KEY (country_code) REFERENCES countries(country_code) ON DELETE CASCADE
    );
    """)

    # 3. NELDA Indicators Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS nelda_indicators (
        election_id INTEGER PRIMARY KEY,
        nelda11_fraud_allegations TEXT,
        nelda15_opposition_boycott TEXT,
        nelda23_incumbent_won TEXT,
        nelda51_monitors_present TEXT,
        FOREIGN KEY (election_id) REFERENCES elections(election_id) ON DELETE CASCADE
    );
    """)

    # 4. V-Dem Indicators Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS vdem_indicators (
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

    # 5. Spatial Weights Table (Including Year for dynamic networks)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS spatial_weights (
        country_code_source TEXT NOT NULL,
        country_code_target TEXT NOT NULL,
        weight_type TEXT NOT NULL,
        year INTEGER NOT NULL,
        weight_value REAL DEFAULT 0.0,
        PRIMARY KEY (country_code_source, country_code_target, weight_type, year),
        FOREIGN KEY (country_code_source) REFERENCES countries(country_code) ON DELETE CASCADE,
        FOREIGN KEY (country_code_target) REFERENCES countries(country_code) ON DELETE CASCADE
    );
    """)

    # 6. GDELT Events Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS gdelt_events (
        event_id TEXT PRIMARY KEY,
        country_code TEXT NOT NULL,
        event_date TEXT NOT NULL,
        cameo_code TEXT NOT NULL,
        goldstein_scale REAL,
        avg_tone REAL,
        news_volume INTEGER,
        actor1_code TEXT,
        actor2_code TEXT,
        FOREIGN KEY (country_code) REFERENCES countries(country_code) ON DELETE CASCADE
    );
    """)

    # 7. Exchange Rates Table (NEW)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS exchange_rates (
        country_code TEXT NOT NULL,
        year INTEGER NOT NULL,
        month INTEGER NOT NULL,
        exchange_rate REAL NOT NULL,
        PRIMARY KEY (country_code, year, month),
        FOREIGN KEY (country_code) REFERENCES countries(country_code) ON DELETE CASCADE
    );
    """)

    # 8. Macroeconomic Indicators Table (NEW)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS macro_indicators (
        country_code TEXT NOT NULL,
        year INTEGER NOT NULL,
        gdp_growth REAL,
        inflation REAL,
        gdp_growth_deviation REAL,
        inflation_deviation REAL,
        unemployment REAL,
        gdp_per_capita_growth REAL,
        net_migration REAL,
        refugee_asylum REAL,
        refugee_origin REAL,
        PRIMARY KEY (country_code, year),
        FOREIGN KEY (country_code) REFERENCES countries(country_code) ON DELETE CASCADE
    );
    """)

    # 8. Upcoming Elections Table (for post-NELDA elections with potentially incomplete data)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS upcoming_elections (
        upcoming_id INTEGER PRIMARY KEY AUTOINCREMENT,
        country_code TEXT NOT NULL,
        election_year INTEGER NOT NULL,
        election_date TEXT,
        election_type TEXT,
        is_scheduled INTEGER DEFAULT 1,
        actual_outcome TEXT,
        notes TEXT,
        incumbent_ran INTEGER,
        has_successor INTEGER,
        FOREIGN KEY (country_code) REFERENCES countries(country_code) ON DELETE CASCADE
    );
    """)

    # 9. CLEA Indicators Table (NEW)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS clea_indicators (
        country_code TEXT NOT NULL,
        year INTEGER NOT NULL,
        turnout REAL,
        margin_of_victory REAL,
        effective_parties REAL,
        incumbent_seat_share REAL,
        pedersen_index REAL,
        PRIMARY KEY (country_code, year),
        FOREIGN KEY (country_code) REFERENCES countries(country_code) ON DELETE CASCADE
    );
    """)

    # 10. Global Indicators Table (NEW)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS global_indicators (
        year INTEGER PRIMARY KEY,
        brent_oil_price REAL,
        fed_funds_rate REAL,
        us_president_ideology INTEGER
    );
    """)

    # Create Indexes
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_elections_date ON elections(election_year, country_code);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_vdem_year ON vdem_indicators(year, country_code);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_gdelt_date_country ON gdelt_events(event_date, country_code);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_spatial_weights ON spatial_weights(country_code_source, weight_type, year);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_macro_indicators ON macro_indicators(country_code, year);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_upcoming_elections ON upcoming_elections(election_year, country_code);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_clea_indicators ON clea_indicators(country_code, year);")

    conn.commit()

    # Verification
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    print("Database tables initialized successfully:")
    for t in sorted(tables):
        print(f" - {t}")
    
    conn.close()


def migrate_db():
    """Non-destructive migration: adds new tables/indexes to existing DB without dropping data."""
    print(f"Migrating database at: {DB_PATH}")
    if not os.path.exists(DB_PATH):
        print("No existing database found. Running full init instead.")
        init_db()
        return
    
    conn = get_connection()
    cursor = conn.cursor()
    
    # Add upcoming_elections table if it doesn't exist
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS upcoming_elections (
        upcoming_id INTEGER PRIMARY KEY AUTOINCREMENT,
        country_code TEXT NOT NULL,
        election_year INTEGER NOT NULL,
        election_date TEXT,
        election_type TEXT,
        is_scheduled INTEGER DEFAULT 1,
        actual_outcome TEXT,
        notes TEXT,
        incumbent_ran INTEGER,
        has_successor INTEGER,
        FOREIGN KEY (country_code) REFERENCES countries(country_code) ON DELETE CASCADE
    );
    """)
    
    # Add incumbent_ran and has_successor columns to elections and upcoming_elections tables if they don't exist
    cursor.execute("PRAGMA table_info(elections);")
    columns = [row[1] for row in cursor.fetchall()]
    if "incumbent_ran" not in columns:
        print("Adding column incumbent_ran to elections table...")
        cursor.execute("ALTER TABLE elections ADD COLUMN incumbent_ran INTEGER;")
    if "has_successor" not in columns:
        print("Adding column has_successor to elections table...")
        cursor.execute("ALTER TABLE elections ADD COLUMN has_successor INTEGER;")
        
    cursor.execute("PRAGMA table_info(upcoming_elections);")
    columns = [row[1] for row in cursor.fetchall()]
    if "incumbent_ran" not in columns:
        print("Adding column incumbent_ran to upcoming_elections table...")
        cursor.execute("ALTER TABLE upcoming_elections ADD COLUMN incumbent_ran INTEGER;")
    if "has_successor" not in columns:
        print("Adding column has_successor to upcoming_elections table...")
        cursor.execute("ALTER TABLE upcoming_elections ADD COLUMN has_successor INTEGER;")
    
    # Add clea_indicators table if it doesn't exist
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS clea_indicators (
        country_code TEXT NOT NULL,
        year INTEGER NOT NULL,
        turnout REAL,
        margin_of_victory REAL,
        effective_parties REAL,
        incumbent_seat_share REAL,
        pedersen_index REAL,
        PRIMARY KEY (country_code, year),
        FOREIGN KEY (country_code) REFERENCES countries(country_code) ON DELETE CASCADE
    );
    """)
    
    # Add exchange_rates table if it doesn't exist
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS exchange_rates (
        country_code TEXT NOT NULL,
        year INTEGER NOT NULL,
        month INTEGER NOT NULL,
        exchange_rate REAL NOT NULL,
        PRIMARY KEY (country_code, year, month),
        FOREIGN KEY (country_code) REFERENCES countries(country_code) ON DELETE CASCADE
    );
    """)
    
    # Add is_oil_exporter column to countries table if it doesn't exist
    cursor.execute("PRAGMA table_info(countries);")
    columns = [row[1] for row in cursor.fetchall()]
    if "is_oil_exporter" not in columns:
        print("Adding column is_oil_exporter to countries table...")
        cursor.execute("ALTER TABLE countries ADD COLUMN is_oil_exporter INTEGER DEFAULT 0;")
        
    # Add actor1_code and actor2_code columns to gdelt_events table if they don't exist
    cursor.execute("PRAGMA table_info(gdelt_events);")
    columns = [row[1] for row in cursor.fetchall()]
    if "actor1_code" not in columns:
        print("Adding column actor1_code to gdelt_events table...")
        cursor.execute("ALTER TABLE gdelt_events ADD COLUMN actor1_code TEXT;")
    if "actor2_code" not in columns:
        print("Adding column actor2_code to gdelt_events table...")
        cursor.execute("ALTER TABLE gdelt_events ADD COLUMN actor2_code TEXT;")
        
    # Add global_indicators table if it doesn't exist
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS global_indicators (
        year INTEGER PRIMARY KEY,
        brent_oil_price REAL,
        fed_funds_rate REAL,
        us_president_ideology INTEGER
    );
    """)
    

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_upcoming_elections ON upcoming_elections(election_year, country_code);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_clea_indicators ON clea_indicators(country_code, year);")
    
    conn.commit()
    
    # Verify
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    print("Database migration complete. Tables:")
    for t in sorted(tables):
        cursor.execute(f"SELECT COUNT(*) FROM [{t}];")
        count = cursor.fetchone()[0]
        print(f"  - {t}: {count:,} rows")
    
    conn.close()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--migrate":
        migrate_db()
    else:
        init_db()
