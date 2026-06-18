# Resume-enhancer embedding service

Standalone FastAPI micro-service that powers ApplyBoost's deterministic ATS
résumé scoring. It exposes a single embeddings endpoint; the Joblet app calls it
over HTTP and compares the returned vectors in-memory (cosine similarity) to
score how well a CV covers a job's skills. Self-hosting a fixed model makes that
score **deterministic** (same input → same vector → same score) and free.

- **Model:** `sentence-transformers/all-MiniLM-L6-v2` (384-dim), overridable via
  `EMBED_MODEL_NAME`.
- **Endpoint:** `POST /v1/embed` — OpenAI-shaped response
  (`{ data: [{ index, embedding }] }`).
- **Health:** `GET /health`.

## Endpoint

```
POST /v1/embed
{ "input": ["text 1", "text 2"], "input_type": "document" }
→ { "model": "...", "dimensions": 384, "data": [{ "index": 0, "embedding": [...] }] }
```

If `EMBED_AUTH_TOKEN` is set, requests must send `Authorization: Bearer <token>`.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8001
curl http://127.0.0.1:8001/health
```

## Deploy on Render

Use `render.yaml` (Blueprint) or create a **Web Service** from this repo with:

- **Language:** Python 3
- **Build:** `pip install --upgrade pip && pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements.txt`
- **Start:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
- **Health Check Path:** `/health`
- **Env:** `PYTHON_VERSION=3.11.9`, `EMBED_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2`, `EMBED_AUTH_TOKEN=<random secret>`

Notes: the `free` plan (512 MB) is tight for torch — bump to `starter` if boot
OOMs. Free also spins down when idle, so the first request after a lull takes
~30-60 s to wake.

## Wire into the Joblet app (Vercel env)

```
APPLYBOOST_EMBEDDING_URL=https://<this-service>.onrender.com
APPLYBOOST_EMBEDDING_API_KEY=<the EMBED_AUTH_TOKEN value>
APPLYBOOST_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
APPLYBOOST_EMBEDDING_COSINE_THRESHOLD=0.5   # calibrate post-deploy, see joblet1.0/scripts/calibrate-ats-threshold.mjs
```
