# La2etha! backend — local setup

Environment notes and the exact steps to run the backend locally. These capture a
few gotchas discovered on the primary dev box (Windows) so they don't bite again.

## Prerequisites

- **Python 3.11 or 3.12 — NOT 3.13.** The CV stack (insightface, onnxruntime-gpu,
  dlib) has no prebuilt wheels for 3.13, so pip tries to compile from source and
  fails without a C/C++ toolchain. `pyproject.toml` enforces `>=3.11,<3.13`.
  On the dev box, use `py -3.12`.
- **Docker Desktop** (for Postgres + Redis).
- A CUDA GPU (RTX 2060 target) for real inference; without it, onnxruntime falls
  back to CPU (slower, still works).

## Gotcha 1 — TLS-inspecting proxy / antivirus

If `pip install` fails with `CERTIFICATE_VERIFY_FAILED (self-signed certificate in
certificate chain)`, the network is intercepting TLS. Either make it durable once:

```bash
pip config set global.trusted-host "pypi.org files.pythonhosted.org pypi.python.org"
```

or pass `--trusted-host pypi.org --trusted-host files.pythonhosted.org
--trusted-host pypi.python.org` on each `pip install`.

## Gotcha 2 — Postgres port 5433

A native PostgreSQL service may already own port **5432**. To avoid the clash,
`docker-compose.yml` maps our Postgres to host port **5433**, and the `.env` DB
URLs use 5433. If you don't have a native Postgres, 5433 still works fine.

## Setup steps

```bash
# 1. Virtual environment (Python 3.12)
py -3.12 -m venv .venv
.venv/Scripts/activate            # Windows;  source .venv/bin/activate on *nix

# 2. Install (add the trusted-host flags if behind a TLS proxy)
pip install -e ".[dev]"           # add .[eval] later for the dlib clustering baseline

# 3. Services
docker compose up -d              # Postgres+pgvector on :5433, Redis on :6379

# 4. Config
cp .env.example .env              # then set JWT_SECRET to something non-default

# 5. Schema
alembic upgrade head              # creates all tables + pgvector + HNSW indexes

# 6. Run (two terminals)
uvicorn app.main:app --reload     # API + Swagger docs at http://localhost:8000/docs
python -m app.workers.worker      # RQ worker (runs the CV pipeline)
```

## Gotcha 3 — first CV job downloads the model

On the first detection/enrollment job, insightface downloads the `buffalo_l`
model (~300 MB). If that download fails on the same proxy cert, pre-place it
manually: download `buffalo_l.zip` and unzip so the files land in
`~/.insightface/models/buffalo_l/` (it should contain the `.onnx` files).

## Tests

```bash
pytest tests/unit                 # pure-logic CV tests (no DB/GPU needed)
pytest tests/integration          # needs Postgres up (skips cleanly if it's down)
```

## Frontend

The web client lives in the sibling repo `la2etha-frontend/`:

```bash
cd ../la2etha-frontend
npm install                       # trusted-host is not needed for npm here
npm run dev                       # http://localhost:5173  (expects the API on :8000)
```
