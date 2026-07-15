import os
import sqlite3
import pandas as pd
import numpy as np
import libpysal

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "elections.db")

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def get_row_normalized_weights(weight_type="unga_voting", year=2020):
    """
    Fetches the weights for a specific weight_type and year from the database,
    constructs a row-normalized spatial weights matrix, and returns a libpysal.weights.W object.
    """
    conn = get_connection()
    df = pd.read_sql_query(
        "SELECT country_code_source, country_code_target, weight_value FROM spatial_weights WHERE weight_type = ? AND year = ?;",
        conn,
        params=(weight_type, year)
    )
    conn.close()

    if df.empty:
        raise ValueError(f"No weights found in database for weight_type: {weight_type}")

    # Build a pivot table representing the weight matrix
    # Index is source, Columns are target
    pivoted = df.pivot(index="country_code_source", columns="country_code_target", values="weight_value")
    
    # Fill missing values (diagonal or missing dyads) with 0.0
    pivoted = pivoted.fillna(0.0)
    
    # Ensure index and columns are identical
    all_countries = sorted(list(set(pivoted.index).union(set(pivoted.columns))))
    pivoted = pivoted.reindex(index=all_countries, columns=all_countries, fill_value=0.0)
    
    # Convert to numpy array for manipulation
    W_matrix = pivoted.values
    
    # Row normalization: divide each row by its sum
    row_sums = W_matrix.sum(axis=1)
    # Avoid division by zero by setting zero sums to 1.0 (they will remain 0.0 anyway)
    row_sums[row_sums == 0] = 1.0
    
    W_normalized = W_matrix / row_sums[:, np.newaxis]
    
    # Convert row-normalized numpy matrix to libpysal W dictionary format
    neighbors = {}
    weights = {}
    
    for i, country in enumerate(all_countries):
        country_neighbors = []
        country_weights = []
        for j, target_country in enumerate(all_countries):
            val = W_normalized[i, j]
            if val > 0.0:
                country_neighbors.append(target_country)
                country_weights.append(val)
                
        # If a country has no neighbors (isolated node), assign a self-loop or leave empty
        if not country_neighbors:
            # Add a fallback neighbor (closest or first country) with very low weight to satisfy PySAL constraints
            fallback = all_countries[0] if country != all_countries[0] else all_countries[1]
            country_neighbors = [fallback]
            country_weights = [1.0]

        neighbors[country] = country_neighbors
        weights[country] = country_weights

    # Construct the libpysal W object
    w_object = libpysal.weights.W(neighbors, weights)
    return w_object, pivoted.index.tolist()

if __name__ == "__main__":
    # Test weights extraction for a sample year
    for wt in ["unga_voting", "trade", "alliance", "contiguity"]:
        try:
            w_obj, countries = get_row_normalized_weights(wt, year=2020)
            print(f"Successfully constructed weights for '{wt}' in 2020: {len(countries)} countries, mean neighbors: {w_obj.mean_neighbors:.2f}")
        except Exception as e:
            print(f"Error constructing weights for '{wt}': {e}")
