"""
Fix JSON columns that were stored as double-encoded strings.
"""
import os
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://workperf:devpassword@localhost:5432/performance")

engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
session = Session()

print("🔧 Fixing JSON columns...")

# The data is stored as JSON-encoded strings inside JSON columns
# We need to parse the string value to get the actual JSON
result = session.execute(text("""
    UPDATE utterances
    SET 
        predictions = (predictions #>> '{}')::json,
        aggregated_scores = (aggregated_scores #>> '{}')::json,
        sa_labels = (sa_labels #>> '{}')::json
    WHERE predictions IS NOT NULL
"""))

session.commit()
print(f"✅ Fixed {result.rowcount} rows")

# Verify
sample = session.execute(text("SELECT predictions, aggregated_scores, sa_labels FROM utterances LIMIT 1")).first()
print(f"\n📊 Sample row:")
print(f"   predictions type: {type(sample[0])}")
print(f"   aggregated_scores type: {type(sample[1])}")
print(f"   sa_labels type: {type(sample[2])}")

session.close()
