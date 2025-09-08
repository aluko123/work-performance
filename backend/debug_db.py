import sqlite3

DB_PATH = 'data/analysis.db'

con = sqlite3.connect(DB_PATH)
cur = con.cursor()

print("--- Distinct Dates in Utterances Table ---")
for row in cur.execute("SELECT DISTINCT date FROM utterances"):
    print(row)

con.close()
