# CI/CD Documentation

## Overview

This project uses GitHub Actions for continuous integration. The CI pipeline ensures code quality and functionality before merging changes.

## Workflows

### 1. Backend CI (`backend-ci.yml`)

**Triggers:**
- Push to `main` or `master` branches
- Pull requests to `main` or `master` branches

**Services:**
- PostgreSQL 16 with pgvector extension
- Redis 7

**Steps:**
1. **Setup**: Checkout code, install Python 3.10, install dependencies
2. **Database Setup**: 
   - Verify PostgreSQL connection
   - Install pgvector extension
   - Run migrations with Alembic
3. **Testing**: Run pytest with coverage
4. **Reporting**: Upload coverage reports as artifacts

**Environment Variables Required:**
- `TEST_DATABASE_URL`: Set to `postgresql+psycopg2://testuser:testpass@localhost:5432/testdb`
- `DATABASE_URL`: Same as above (for migrations)
- `REDIS_URL`: Set to `redis://localhost:6379/0`
- `PYTHONPATH`: Set to workspace root
- `OPENAI_API_KEY`: GitHub secret (optional for tests)
- `CHUNKRAI_API_KEY`: GitHub secret (optional for tests)

### 2. Lint (`lint.yml`)

**Triggers:**
- Push to `main` or `master` branches
- Pull requests to `main` or `master` branches

**Steps:**
1. Run ruff for code quality checks
2. Run mypy for type checking

*(Currently configured to continue on error - enable strict mode when ready)*

## Local Development vs CI

### Local Development (Docker Compose)
```bash
docker compose up --build
make test-docker
```
- Uses `pgvector/pgvector:pg16`
- Database: `performance` (user: `workperf`)
- Hostname: `postgres` (Docker network)

### CI Environment
- Uses `pgvector/pgvector:pg16` service
- Database: `testdb` (user: `testuser`, password: `testpass`)
- Hostname: `localhost` (GitHub Actions runner)

### Key Differences

| Aspect | Local | CI |
|--------|-------|-----|
| Database Host | `postgres` | `localhost` |
| Database Name | `performance` | `testdb` |
| User | `workperf` | `testuser` |
| Isolation | Docker volumes | Ephemeral (recreated per run) |
| Test Command | `make test-docker` | `pytest backend/tests -v` |

## Test Isolation Strategy

Tests use table truncation for isolation (see `backend/tests/conftest.py`):
- Before each test: `TRUNCATE TABLE utterances, analyses RESTART IDENTITY CASCADE`
- After each test: Same truncation to clean up

This ensures:
- ✅ Fast test execution
- ✅ No cross-test contamination
- ✅ Predictable test outcomes

## Troubleshooting

### Tests fail with "could not translate host name 'postgres'"

**Cause**: Environment variable `TEST_DATABASE_URL` not set or using wrong hostname

**Solution**: Ensure `TEST_DATABASE_URL` uses `localhost` in CI, `postgres` in Docker Compose

### Database connection errors

**Cause**: Database service not ready or credentials incorrect

**Solution**: 
- Check service health checks are passing
- Verify credentials match between service definition and connection string

### Migration errors about missing pgvector extension

**Cause**: Using standard PostgreSQL image instead of pgvector image

**Solution**: Use `pgvector/pgvector:pg16` image in both local and CI

## Adding GitHub Secrets

For CI to work with external APIs (optional):

1. Go to repository Settings → Secrets and variables → Actions
2. Add:
   - `OPENAI_API_KEY`: Your OpenAI API key
   - `CHUNKRAI_API_KEY`: Your Chunkr.ai API key

These are optional - tests will skip features requiring these if not set.

## Coverage Reports

Coverage reports are uploaded as artifacts after each CI run:
- Navigate to Actions → Select workflow run → Artifacts
- Download `coverage-report` to view detailed coverage

## Best Practices

1. **Run tests locally before pushing**:
   ```bash
   make test-docker
   ```

2. **Check CI logs for failures**: GitHub Actions provides detailed logs

3. **Keep dependencies updated**: Regularly update `requirements.txt`

4. **Maintain test isolation**: Don't rely on test execution order

5. **Use meaningful commit messages**: Helps when reviewing CI history
