## Summary
- What does this change do and why?

## Related Issues
- Closes #<issue-id> / Relates to #<issue-id>

## Changes
- [ ] Backend
  - Key modules touched (e.g., `backend/services.py`, `backend/main.py`)
  - API endpoints added/changed
- [ ] Frontend
  - Files updated (e.g., `frontend/script.js`)
  - UI/UX notes
- [ ] Config/Infra
  - `Dockerfile`, `docker-compose.yml`, env vars

## How to Test
1. Start backend: `uvicorn backend.main:app --reload --port 8000`
2. Serve frontend: `python -m http.server 8001 -d frontend`
3. Verify endpoints:
   - `curl http://localhost:8000/analyses/`
   - `curl "http://localhost:8000/api/trends?metric=comm_clarity&period=daily"`
   - `curl -H "Content-Type: application/json" -d '{"question":"What improved last week?"}' http://localhost:8000/api/get_insights`

## Screenshots
Attach before/after UI images if applicable.

## Checklist
- [ ] Clear, imperative commit messages
- [ ] Description explains rationale and impact
- [ ] Added/updated docs where needed
- [ ] Env vars documented (`.env` keys, defaults)
- [ ] Backward compatible (or migration noted)
- [ ] Performance implications considered
- [ ] Security/privacy reviewed (secrets, PII)
- [ ] DB changes safe; rollback plan documented

## Risks & Rollback
- Risks:
- Rollback steps:

