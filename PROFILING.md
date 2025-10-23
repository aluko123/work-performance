# Profiling Guide

This document explains how to profile the application to identify performance bottlenecks.

## Setup

Profiling tools (`py-spy`, `psutil`) are included in `requirements.txt` and Docker is configured with ptrace capabilities for profiling.

## Quick Start

### 1. Rebuild Docker containers
```bash
docker compose down
docker compose up --build -d
```

### 2. Profile the running backend service

**Generate a flamegraph (visual CPU profile):**
```bash
docker compose exec backend py-spy record -o /app/data/profile.svg --duration 60 --pid 1
```
This creates a 60-second flamegraph at `./data/profile.svg` showing where CPU time is spent.

**Live top-like view:**
```bash
docker compose exec backend py-spy top --pid 1
```
Shows real-time function call stats (like `top` for Python functions).

**Profile specific endpoint under load:**
```bash
# Terminal 1: Start profiling
docker compose exec backend py-spy record -o /app/data/endpoint_profile.svg --pid 1

# Terminal 2: Generate load (install `ab` or `hey` if needed)
ab -n 1000 -c 10 http://localhost:8000/analyses/
```

### 3. Profile the ARQ worker (background jobs)
```bash
docker compose exec arq_worker py-spy record -o /app/data/worker_profile.svg --duration 60 --pid 1
```

## What to Look For

### In Flamegraphs
- **Wide bars at top**: Functions consuming most CPU time
- **Common culprits in your stack**:
  - OpenAI API calls (network I/O)
  - BERT model inference (`transformers` library)
  - Embedding generation (`sentence-transformers`)
  - ChromaDB vector operations
  - SQLAlchemy queries

### Memory Profiling
```bash
# Check container memory usage
docker stats

# Profile memory inside container
docker compose exec backend python -m memory_profiler backend/main.py
```

## Advanced: Continuous Profiling

Add FastAPI middleware for automatic timing (see `backend/main.py`):
```python
import time
from fastapi import Request

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response
```

## Monitoring Recommendations

For production monitoring:
- **Prometheus + Grafana**: Add `/metrics` endpoint with `prometheus-fastapi-instrumentator`
- **Sentry**: Add `sentry-sdk` for error tracking + performance
- **OpenTelemetry**: Distributed tracing across FastAPI → OpenAI → ChromaDB

## Common Bottlenecks in This Application

1. **OpenAI API calls** - Add caching or batch requests
2. **BERT inference** - Consider model quantization or batching
3. **ChromaDB queries** - Optimize index size or query parameters
4. **SQLite I/O** - Consider PostgreSQL for production
5. **PDF parsing** - Move to background worker (already done via ARQ)

## Files Generated

All profile outputs are saved to `./data/` which is volume-mounted, so they persist outside containers:
- `profile.svg` - Main backend flamegraph
- `worker_profile.svg` - ARQ worker flamegraph
- `endpoint_profile.svg` - Specific endpoint profiles

Open `.svg` files in any browser to view interactive flamegraphs.
