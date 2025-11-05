"""
Simple embedding generation for pgvector.
Replaces the deprecated rag_service.py ChromaDB system.
"""

import os
from typing import List
from openai import OpenAI
import os

from .database import SessionLocal
from . import db_models


def generate_embeddings_for_utterances(utterance_ids: List[int]) -> int:
    """
    Generate and store embeddings for utterances directly in Postgres.
    
    Args:
        utterance_ids: List of utterance IDs to generate embeddings for
        
    Returns:
        Number of utterances successfully indexed
    """
    if not utterance_ids:
        print("No utterances to index.")
        return 0
    
    openai_client = OpenAI()
    embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    session = SessionLocal()
    
    try:
        # Fetch utterances that need indexing
        utterances = session.query(db_models.Utterance).filter(
            db_models.Utterance.id.in_(utterance_ids),
            db_models.Utterance.is_indexed == False
        ).all()
        
        if not utterances:
            print(f"No unindexed utterances found from {len(utterance_ids)} IDs")
            return 0
        
        batch_size = int(os.getenv("BATCH_SIZE", 50))
        total_docs = len(utterances)
        print(f"Generating embeddings for {total_docs} utterances in batches of {batch_size}...")
        
        indexed_count = 0
        
        for i in range(0, total_docs, batch_size):
            batch = utterances[i:i + batch_size]
            
            try:
                # Generate embeddings for the batch in a single API call
                inputs = [(u.text or "")[:8000] for u in batch]
                embedding_response = openai_client.embeddings.create(
                    model=embedding_model,
                    input=inputs
                )
                vectors = [d.embedding for d in embedding_response.data]

                # Assign embeddings back to utterances in order
                for utterance, vec in zip(batch, vectors):
                    utterance.embedding = vec
                    utterance.is_indexed = True
                    indexed_count += 1

                # Commit the batch
                session.commit()
                print(f"Indexed batch {i//batch_size + 1}/{(total_docs + batch_size - 1)//batch_size} ({indexed_count}/{total_docs})")
                
            except Exception as e:
                print(f"Error indexing batch {i//batch_size + 1}: {e}")
                session.rollback()
                continue
        
        print(f"✅ Finished indexing {indexed_count}/{total_docs} utterances.")
        return indexed_count
        
    finally:
        session.close()
