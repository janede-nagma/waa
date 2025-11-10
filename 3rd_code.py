import pandas as pd
import sqlite3
from multiprocessing import Pool

def mapper(row):
    return (row["Month"], row["Temperature_Celsius"])

def run_analysis():
    df = pd.read_csv("forestfires.csv")
    print(f" Loaded {len(df)} rows\n")

    # MapReduce: Average temperature
    rows = [row for _, row in df.iterrows()]
    
    with Pool() as pool:
        mapped = pool.map(mapper, rows)

    temps = {}
    for month, temp in mapped:
        temps.setdefault(month, []).append(temp)

    print("🌡️ Avg Temperature (MapReduce):")
    for m, t in temps.items():
        print(f"{m}: {sum(t)/len(t):.2f}")

    # SQL: Average burned area
    conn = sqlite3.connect(":memory:")
    df.to_sql("forestfires", conn, index=False)

    query = "SELECT Month, AVG(Burned_Area_hectares) AS avg FROM forestfires GROUP BY Month ORDER BY avg DESC"
    print("\nAvg Burned Area (SQL):")
    print(pd.read_sql_query(query, conn).to_string(index=False))

    conn.close()
    print("\n Complete")

if __name__ == "__main__":

    run_analysis()
