# TODO: Next Steps

**Last Updated:** 2025-01-27  
**Status:** New agent system working, ready for enhancements  
**Timeline:** Half day work tomorrow

---

## ✅ Completed Today

- [x] Built new OpenAI native agent (515 lines vs 1,100+)
- [x] Multi-turn conversations working perfectly
- [x] Added input validation (no more hallucinations)
- [x] Added list_speakers and list_metrics tools
- [x] Switched frontend to /api/chat
- [x] Comprehensive tests added
- [x] Full architecture documented

---

## 🚀 TODO Tomorrow

### Priority 1: Implement Smart Chart Auto-Generation (30-45 min)

**Goal:** Charts automatically generated based on tool results (Option C approach)

**Implementation:**
```python
# In backend/main.py, after agent completes:

def generate_charts_from_tools(tool_results, question):
    """
    Auto-generate charts based on what tools were called.
    No LLM decision needed - just pattern matching.
    """
    charts = []
    
    # Pattern 1: If compare_periods was called → line chart
    if any("compare_periods" in tr for tr in tool_results):
        metric = extract_metric_from_tool_results(tool_results)
        charts.append({
            "type": "line",
            "data": build_time_series(metric),  # Reuse from services.py
            "config": {"title": f"{metric} Over Time"}
        })
    
    # Pattern 2: If multiple speakers in results → bar chart
    speakers = extract_speakers_from_tool_results(tool_results)
    if len(speakers) > 1:
        charts.append({
            "type": "bar",
            "data": build_speaker_comparison(speakers, metric),
            "config": {"title": f"Speaker Comparison - {metric}"}
        })
    
    return charts
```

**Files to modify:**
- [ ] `backend/main.py` - Add chart generation after agent completes
- [ ] Reuse chart logic from `backend/services.py` (get_speaker_trends, get_chart_data)
- [ ] Test: "Has safety improved?" should auto-generate trend chart

**Test queries:**
- "Has safety improved?" → Should get line chart
- "Compare Tasha and Mike" → Should get bar chart
- "How is safety?" → No chart (simple query)

---

### Priority 2: Migrate to Postgres + pgvector (2-3 hours)

**Goal:** Replace SQLite + ChromaDB with single Postgres database

**Why now:**
- Small dataset (2K rows) = easy migration
- Fresh system = no legacy complexity
- Learn pgvector before scale
- Production-ready stack

#### Step 1: Update Docker Compose (15 min)

```yaml
# docker-compose.yml - ADD:

services:
  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_USER: workperf
      POSTGRES_PASSWORD: ${DB_PASSWORD:-devpassword}
      POSTGRES_DB: performance
    ports:
      - "5432:5432"
    volumes:
      - postgres-data:/var/lib/postgresql/data

volumes:
  postgres-data:
```

#### Step 2: Update Database Connection (15 min)

```python
# backend/database.py

# OLD:
# DATABASE_URL = "sqlite:///./data/analysis.db"

# NEW:
DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "postgresql://workperf:devpassword@postgres:5432/performance"
)
```

#### Step 3: Add pgvector Extension (10 min)

```sql
-- migrations/add_pgvector.sql
CREATE EXTENSION IF NOT EXISTS vector;

ALTER TABLE utterances 
ADD COLUMN embedding vector(1536);  -- OpenAI ada-002 dimension

CREATE INDEX ON utterances 
USING ivfflat (embedding vector_cosine_ops);
```

#### Step 4: Migrate Data (30 min)

```python
# scripts/migrate_to_postgres.py

# 1. Export from SQLite
# 2. Import to Postgres
# 3. Generate embeddings for existing utterances
# 4. Store in embedding column
```

#### Step 5: Replace ChromaDB (30 min)

```python
# backend/tools.py - search_utterances

# OLD: ChromaDB query
# NEW: Postgres pgvector query

def search_utterances(query, speaker=None, top_k=5):
    # Generate embedding
    embedding = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    ).data[0].embedding
    
    # Query Postgres with pgvector
    session = SessionLocal()
    query_str = """
        SELECT speaker, date, timestamp, text, 
               1 - (embedding <=> :embedding) as similarity
        FROM utterances
        WHERE speaker = :speaker OR :speaker IS NULL
        ORDER BY embedding <=> :embedding
        LIMIT :top_k
    """
    results = session.execute(query_str, {
        "embedding": embedding,
        "speaker": speaker,
        "top_k": top_k
    })
    
    return [dict(r) for r in results]
```

#### Step 6: Update Indexing (30 min)

```python
# When new documents uploaded, generate embeddings:

for utterance in new_utterances:
    embedding = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=utterance.text
    ).data[0].embedding
    
    utterance.embedding = embedding
    session.add(utterance)
```

#### Step 7: Remove ChromaDB (5 min)

```bash
# requirements.txt - REMOVE:
chromadb

# docker-compose.yml - REMOVE volume:
./data/chroma_db

# backend/tools.py - DELETE:
import chromadb
```

**Testing checklist:**
- [ ] Semantic search works: "What did Tasha say about safety?"
- [ ] SQL queries still work: "How is Tasha safety?"
- [ ] Performance acceptable (<2s for search)
- [ ] All 2K rows migrated successfully

---

### Priority 3: Fix Output Issues (1 hour)

**After charts are done:**

1. **Real Citations from Tool Results**
   ```python
   # Extract from search_utterances results
   citations = [
       {
           "speaker": r["speaker"],
           "date": r["date"],
           "text": r["text"][:500]  # Increase from 200 to 500
       }
       for r in search_results
   ]
   ```

2. **Real Bullets from Answer**
   - Already extracting bullets from markdown
   - Might need to improve extraction logic

3. **Test UI rendering**
   - Citations should show full context (500 chars)
   - Charts should appear immediately after answer
   - Bullets should be meaningful

---

## 📋 Order of Execution Tomorrow

**Morning (2-3 hours):**
1. ✅ Implement smart chart auto-generation (Option C)
2. ✅ Test charts appear correctly in UI
3. ✅ Fix citation truncation (200 → 500 chars)

**Afternoon (2-3 hours):**
1. ✅ Set up Postgres + pgvector in Docker
2. ✅ Migrate 2K rows from SQLite
3. ✅ Replace ChromaDB with pgvector
4. ✅ Test semantic search works
5. ✅ Remove ChromaDB dependency

**Evening (30 min):**
1. ✅ Run full test suite
2. ✅ Test in UI with real queries
3. ✅ Validate no regressions

---

## 📝 Reference Files

**Architecture:**
- [ARCHITECTURE_NEW.md](file:///mnt/c/Users/adeda/Dropbox/PC/Downloads/Work/ARCHITECTURE_NEW.md)
- [MIGRATION_COMPLETE.md](file:///mnt/c/Users/adeda/Dropbox/PC/Downloads/Work/MIGRATION_COMPLETE.md)

**Charts:**
- [CHART_TOOL_DESIGN.md](file:///mnt/c/Users/adeda/Dropbox/PC/Downloads/Work/plans/CHART_TOOL_DESIGN.md)

**Code:**
- `backend/agent.py` - Agent loop
- `backend/tools.py` - Tools (add chart generation here)
- `backend/services.py` - Existing chart functions (reuse these)

---

## 🎯 End State (Tomorrow EOD)

**You'll have:**
- ✅ Conversational agent with perfect context
- ✅ Smart auto-generated charts
- ✅ Postgres + pgvector (production-ready)
- ✅ Single database (no ChromaDB)
- ✅ Real citations with full context
- ✅ No hallucinations (validation prevents)
- ✅ ~600 lines of clean code
- ✅ Industry-standard architecture

**Then you can truly move on from this project.**

---

## 🚨 Important Notes

**Before Postgres migration:**
- Backup `data/analysis.db` (your current SQLite)
- Don't delete ChromaDB until pgvector tested
- Run migration on a copy first

**Testing priorities:**
- Conversations still work (most important)
- Charts appear and are accurate
- Semantic search still finds relevant discussions

---

## 💬 Questions to Answer Tomorrow

1. Do you want citations extracted from search results? Or skip citations entirely?
2. Keep generic follow-up suggestions or generate contextual ones?
3. After Postgres migration, delete old SQLite file immediately or keep as backup?

---

**Status:** Ready to resume. All current work committed and working.  
**Docker:** Services running on ports 8000 (API), 8001 (UI), 6379 (Redis)
