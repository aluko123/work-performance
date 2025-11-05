"""
Migration script to copy data from SQLite to Postgres and generate embeddings.

Usage:
    python scripts/migrate_to_postgres.py

Environment variables required:
    - DATABASE_URL: Postgres connection string
    - OPENAI_API_KEY: For generating embeddings
"""

import os
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from openai import OpenAI
from backend.db_models import Base, Analysis, Utterance
import random

# Configuration
SQLITE_URL = "sqlite:///./data/analysis.db"
POSTGRES_URL = os.getenv("DATABASE_URL")
BATCH_SIZE = 200
EMBEDDING_BATCH_SIZE = 50
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not POSTGRES_URL:
    print("❌ DATABASE_URL environment variable not set")
    sys.exit(1)

if not OPENAI_API_KEY:
    print("❌ OPENAI_API_KEY environment variable not set")
    sys.exit(1)

# Initialize OpenAI client
openai_client = OpenAI(api_key=OPENAI_API_KEY)

print("🔄 Starting migration from SQLite to Postgres")
print(f"   Source: {SQLITE_URL}")
print(f"   Target: {POSTGRES_URL}")

# Create engines
sqlite_engine = create_engine(SQLITE_URL, connect_args={"check_same_thread": False})
postgres_engine = create_engine(POSTGRES_URL, pool_pre_ping=True)

# Create sessions
SqliteSession = sessionmaker(bind=sqlite_engine)
PostgresSession = sessionmaker(bind=postgres_engine)


def generate_embedding(text: str, retry_count=3, backoff_factor=2) -> list[float]:
    """Generate embedding with retry logic and rate limiting."""
    for attempt in range(retry_count):
        try:
            response = openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=text[:8000]  # Limit text length
            )
            return response.data[0].embedding
        except Exception as e:
            if attempt < retry_count - 1:
                wait_time = backoff_factor ** attempt + random.uniform(0, 1)
                print(f"   ⚠️  Retry {attempt + 1}/{retry_count} after {wait_time:.1f}s: {str(e)[:50]}")
                time.sleep(wait_time)
            else:
                print(f"   ❌ Failed to generate embedding after {retry_count} attempts: {e}")
                raise


def migrate_data():
    """Copy all data from SQLite to Postgres."""
    sqlite_session = SqliteSession()
    postgres_session = PostgresSession()
    
    try:
        # Count records in SQLite using raw SQL (old schema doesn't have new columns)
        analyses_count = sqlite_session.execute(text("SELECT COUNT(*) FROM analyses")).scalar()
        utterances_count = sqlite_session.execute(text("SELECT COUNT(*) FROM utterances")).scalar()
        
        print(f"\n📊 Found in SQLite:")
        print(f"   - {analyses_count} analyses")
        print(f"   - {utterances_count} utterances")
        
        # Migrate analyses using raw SQL
        print(f"\n📝 Migrating analyses...")
        analyses = sqlite_session.execute(
            text("SELECT id, source_filename, created_at FROM analyses")
        ).mappings().all()
        
        for i, analysis in enumerate(analyses, 1):
            new_analysis = Analysis(
                id=analysis['id'],
                source_filename=analysis['source_filename'],
                created_at=analysis['created_at']
            )
            postgres_session.merge(new_analysis)
            
            if i % 10 == 0:
                print(f"   - Migrated {i}/{analyses_count} analyses")
        
        postgres_session.commit()
        print(f"   ✅ Migrated {analyses_count} analyses")
        
        # Migrate utterances in batches using raw SQL (old schema doesn't have is_indexed/embedding)
        print(f"\n📝 Migrating utterances (in batches of {BATCH_SIZE})...")
        offset = 0
        migrated_count = 0
        
        while True:
            utterances = sqlite_session.execute(
                text("""
                    SELECT id, analysis_id, date, timestamp, speaker, text, 
                           predictions, aggregated_scores, sa_labels 
                    FROM utterances 
                    LIMIT :limit OFFSET :offset
                """),
                {"limit": BATCH_SIZE, "offset": offset}
            ).mappings().all()
            
            if not utterances:
                break
            
            for utterance in utterances:
                new_utterance = Utterance(
                    id=utterance['id'],
                    analysis_id=utterance['analysis_id'],
                    date=utterance['date'],
                    timestamp=utterance['timestamp'],
                    speaker=utterance['speaker'],
                    text=utterance['text'],
                    predictions=utterance['predictions'],
                    aggregated_scores=utterance['aggregated_scores'],
                    sa_labels=utterance['sa_labels'],
                    is_indexed=False,  # New column, default to False
                    embedding=None  # Will be populated later
                )
                postgres_session.merge(new_utterance)
            
            postgres_session.commit()
            migrated_count += len(utterances)
            print(f"   - Migrated {migrated_count}/{utterances_count} utterances")
            offset += BATCH_SIZE
        
        print(f"   ✅ Migrated {migrated_count} utterances")
        
        # Fix sequence if IDs were explicitly set
        print(f"\n🔧 Resetting sequences...")
        postgres_session.execute(text(
            "SELECT setval('analyses_id_seq', (SELECT COALESCE(MAX(id), 1) FROM analyses))"
        ))
        postgres_session.execute(text(
            "SELECT setval('utterances_id_seq', (SELECT COALESCE(MAX(id), 1) FROM utterances))"
        ))
        postgres_session.commit()
        print(f"   ✅ Sequences reset")
        
        # Verify migration
        pg_analyses_count = postgres_session.query(Analysis).count()
        pg_utterances_count = postgres_session.query(Utterance).count()
        
        print(f"\n✅ Migration verification:")
        print(f"   - Analyses: {analyses_count} → {pg_analyses_count}")
        print(f"   - Utterances: {utterances_count} → {pg_utterances_count}")
        
        if pg_analyses_count != analyses_count or pg_utterances_count != utterances_count:
            raise Exception("❌ Migration count mismatch!")
        
        return utterances_count
        
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        postgres_session.rollback()
        raise
    finally:
        sqlite_session.close()
        postgres_session.close()


def generate_embeddings():
    """Generate embeddings for all utterances."""
    postgres_session = PostgresSession()
    
    try:
        # Count utterances without embeddings
        total_count = postgres_session.query(Utterance).filter(
            Utterance.embedding.is_(None)
        ).count()
        
        print(f"\n🤖 Generating embeddings for {total_count} utterances...")
        print(f"   (Batch size: {EMBEDDING_BATCH_SIZE}, Rate: ~50-100 RPM)")
        
        offset = 0
        generated_count = 0
        start_time = time.time()
        
        while True:
            # Fetch batch
            utterances = postgres_session.query(Utterance).filter(
                Utterance.embedding.is_(None)
            ).offset(offset).limit(EMBEDDING_BATCH_SIZE).all()
            
            if not utterances:
                break
            
            batch_start = time.time()
            
            # Generate embeddings for batch
            for i, utterance in enumerate(utterances):
                try:
                    embedding = generate_embedding(utterance.text)
                    utterance.embedding = embedding
                    generated_count += 1
                    
                    if generated_count % 10 == 0:
                        elapsed = time.time() - start_time
                        rate = generated_count / elapsed * 60 if elapsed > 0 else 0
                        eta = (total_count - generated_count) / (generated_count / elapsed) if generated_count > 0 else 0
                        print(f"   - {generated_count}/{total_count} embeddings ({rate:.1f}/min, ETA: {eta/60:.1f}m)")
                
                except Exception as e:
                    print(f"   ⚠️  Skipping utterance {utterance.id}: {str(e)[:100]}")
            
            postgres_session.commit()
            
            # Rate limiting: aim for ~50-100 requests/min
            batch_elapsed = time.time() - batch_start
            target_batch_time = len(utterances) * 0.6  # 0.6s per request = 100 RPM
            if batch_elapsed < target_batch_time:
                sleep_time = target_batch_time - batch_elapsed
                time.sleep(sleep_time)
            
            offset += EMBEDDING_BATCH_SIZE
        
        elapsed = time.time() - start_time
        print(f"   ✅ Generated {generated_count} embeddings in {elapsed/60:.1f} minutes")
        print(f"   Average rate: {generated_count / elapsed * 60:.1f} embeddings/min")
        
    except Exception as e:
        print(f"❌ Embedding generation failed: {e}")
        postgres_session.rollback()
        raise
    finally:
        postgres_session.close()


def verify_migration():
    """Verify data integrity and perform a test search."""
    postgres_session = PostgresSession()
    
    try:
        print(f"\n🔍 Verifying migration...")
        
        # Count checks
        total = postgres_session.query(Utterance).count()
        with_embeddings = postgres_session.query(Utterance).filter(
            Utterance.embedding.isnot(None)
        ).count()
        
        print(f"   - Total utterances: {total}")
        print(f"   - With embeddings: {with_embeddings}")
        print(f"   - Coverage: {with_embeddings/total*100:.1f}%")
        
        # Sample data check
        sample = postgres_session.query(Utterance).first()
        print(f"\n   Sample utterance:")
        print(f"   - ID: {sample.id}")
        print(f"   - Speaker: {sample.speaker}")
        print(f"   - Text: {sample.text[:50]}...")
        has_embedding = sample.embedding is not None
        print(f"   - Has embedding: {has_embedding}")
        if has_embedding:
            print(f"   - Embedding dimensions: {len(sample.embedding)}")
        
        # Test semantic search
        if with_embeddings > 0:
            print(f"\n🔎 Testing semantic search...")
            test_query = "safety concerns"
            query_embedding = generate_embedding(test_query)
            
            # Convert list to string format for pgvector
            embedding_str = str(query_embedding)
            
            results = postgres_session.execute(text("""
                SELECT speaker, date, text,
                       1 - (embedding <=> CAST(:embedding AS vector)) AS similarity
                FROM utterances
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> CAST(:embedding AS vector)
                LIMIT 3
            """), {"embedding": embedding_str}).mappings().all()
            
            print(f"   Query: '{test_query}'")
            print(f"   Top 3 results:")
            for i, r in enumerate(results, 1):
                print(f"   {i}. [{r['speaker']}] {r['text'][:60]}... (similarity: {r['similarity']:.3f})")
        
        print(f"\n✅ Verification complete!")
        
    finally:
        postgres_session.close()


if __name__ == "__main__":
    try:
        # Step 1: Migrate data
        utterances_count = migrate_data()
        
        # Step 2: Generate embeddings
        generate_embeddings()
        
        # Step 3: Verify
        verify_migration()
        
        print(f"\n🎉 Migration complete!")
        print(f"\n📋 Next steps:")
        print(f"   1. Update .env to set DATABASE_URL to Postgres")
        print(f"   2. Test the application with Postgres")
        print(f"   3. Once confirmed working, remove ChromaDB dependency")
        
    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        sys.exit(1)
