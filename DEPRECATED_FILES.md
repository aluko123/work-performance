# Deprecated Files & Cleanup Plan

**Date:** 2025-01-27  
**Reason:** Migrated from LangChain/LangGraph to OpenAI Native SDK  
**Savings:** ~600 lines of complex orchestration code

---

## 🗑️ Files to Delete (After 1 Week of Stability)

### Core RAG System (REPLACED)
```
backend/rag_graph.py              ~915 lines  → REPLACED by agent.py (180 lines)
backend/rag_service.py            ~100 lines  → REPLACED by tools.py (200 lines)
backend/prompts/rag.py            ~88 lines   → REPLACED by system prompt in agent.py
```

**Total Removed:** ~1,103 lines  
**Total New:** ~515 lines  
**Net Reduction:** 588 lines (-53%)

---

### Deprecated Tests
```
backend/tests/test_rag_graph.py           → Some tests now obsolete
backend/tests/test_chart_generation.py    → Already skipped (imports removed function)
backend/tests/test_conversational_ux.py   → Replaced by test_agent.py
```

**Action:** Can delete after verifying test_agent.py covers same scenarios

---

## ⚠️ Files to Update

### Keep But Simplify
```
backend/main.py
  - Keep: /api/chat (new)
  - Deprecate: /api/get_insights (old)
  - Action: Add deprecation warning to old endpoint after 1 week
```

### Frontend
```
frontend-v2/src/lib/api.ts
  - Changed: streamInsights() now calls /api/chat
  - Old: /api/get_insights
  - Status: UPDATED ✓
```

---

## 📦 Dependencies to Remove

After deleting old RAG files, remove from `requirements.txt`:

```
langchain>=0.3.0
langchain-community>=0.3.0
langchain-core>=0.3.0
langchain-openai>=0.2.0
langgraph>=0.2.0
```

**Impact:** ~5 fewer dependencies, smaller Docker image, faster builds

---

## ✅ Files to KEEP (Still Used)

### Shared Infrastructure
```
backend/database.py           ✅ Database connection (used by tools.py)
backend/db_models.py          ✅ SQLAlchemy models (used by tools.py)
backend/models.py             ✅ Pydantic models (API contracts)
backend/services.py           ✅ May reuse for chart tool
backend/config/chart_config.py ✅ Metric names (used by tools.py)
```

### Document Processing (Independent)
```
backend/document_extractor.py  ✅ File upload processing
backend/parsing.py             ✅ Text extraction
backend/message_parser.py      ✅ Message parsing
backend/worker.py              ✅ ARQ background jobs
```

### Tests (Active)
```
backend/tests/test_agent.py       ✅ NEW - Tests new system
backend/tests/test_services.py    ✅ KEEP - Tests shared functions
backend/tests/test_prompts.py     ✅ KEEP - Tests extraction prompts
backend/tests/test_api.py         ✅ KEEP - API integration tests
backend/tests/conftest.py         ✅ KEEP - Test fixtures
```

---

## Cleanup Timeline

### Week 1 (Current)
- [x] New system deployed (`/api/chat`)
- [x] Frontend switched to new endpoint
- [x] Old endpoint still available (`/api/get_insights`)
- [x] Monitor both endpoints

### Week 2
- [ ] Verify `/api/get_insights` traffic = 0
- [ ] Add deprecation warning to old endpoint
- [ ] Update README with migration notice

### Week 3
- [ ] Remove old endpoint from main.py
- [ ] Delete rag_graph.py, rag_service.py, prompts/rag.py
- [ ] Remove LangChain dependencies
- [ ] Delete obsolete tests
- [ ] Update CI to skip deprecated file coverage

### Week 4
- [ ] Final validation
- [ ] Update documentation
- [ ] Close migration task

---

## Rollback Plan (If Needed)

**If new system fails in Week 1:**

```bash
# Quick rollback (< 1 minute)
git checkout HEAD~N frontend-v2/src/lib/api.ts
docker compose restart frontend

# System reverts to /api/get_insights automatically
```

**Files preserved for rollback:**
- rag_graph.py (archived, not deleted)
- rag_service.py (archived, not deleted)
- prompts/rag.py (archived, not deleted)

---

## Code Coverage Impact

### Before Cleanup
```
Total backend LOC: ~5,000
RAG system: ~1,103 lines
Coverage target: 70% overall
```

### After Cleanup
```
Total backend LOC: ~4,500 (-10%)
RAG system: ~515 lines (-53%)
Coverage target: 80% overall (easier to achieve with less code)
```

### New Coverage Targets
```
backend/agent.py         85%+  (core logic)
backend/tools.py         90%+  (simple functions)
backend/metadata.py      80%+  (caching layer)
```

---

## Migration Validation Checklist

Before deleting old files, verify:

- [ ] All manual test scenarios pass with /api/chat
- [ ] No errors in production logs for 1 week
- [ ] User satisfaction >= old system
- [ ] Performance (latency, token usage) acceptable
- [ ] Full test suite passes (pytest backend/tests/test_agent.py)
- [ ] Code coverage >= 80% on new files
- [ ] Documentation updated
- [ ] Team trained on new architecture

**Only delete after all checkboxes complete!**

---

## Quick Reference

**New System Files:**
- `backend/agent.py` - Main logic
- `backend/tools.py` - Tool implementations
- `backend/metadata.py` - Metadata caching

**Deprecated Files:**
- `backend/rag_graph.py` - Old orchestration
- `backend/rag_service.py` - Old wrapper
- `backend/prompts/rag.py` - Old prompts

**Endpoint Migration:**
- Old: `/api/get_insights`
- New: `/api/chat`
- Frontend: Updated ✓
