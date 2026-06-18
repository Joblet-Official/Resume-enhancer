"""
Local FastAPI embedding service for the jobs hybrid-search pipeline.

Model: BAAI/bge-small-en-v1.5  (384-dim, cosine similarity, normalize=True)
Why this model: it matches OpenAI `text-embedding-3-small` on MTEB (~62.2)
while staying 384-dim — same `vector(384)` column, same `vector_cosine_ops`
HNSW index. No schema change needed.

Endpoints
---------
POST /v1/embed
    Request:
        {
          "input":      ["text 1", "text 2", ...],
          "input_type": "document" | "query"      # default "document"
        }
    Response (OpenAI-compatible shape):
        {
          "model":      "BAAI/bge-small-en-v1.5",
          "dimensions": 384,
          "data": [
            { "index": 0, "embedding": [..384 floats..] },
            ...
          ]
        }

GET /health
    Reports model load status + device (cpu/cuda/mps).

IMPORTANT — BGE asymmetric prefix
---------------------------------
The bge-* family is trained with an asymmetric instruction format:
  - Document side: no prefix.
  - Query side:    "Represent this sentence for searching relevant passages: "
Skipping the query prefix costs ~2-3 MTEB points. The Node client
(`api-handlers/_lib/jobsEmbedding.js`) sends `input_type="query"` for
user-typed search keywords and `input_type="document"` when backfilling
rows — we apply the prefix here based on that flag.

Run
---
    cd services/embed-service
    pip install -r requirements.txt
    uvicorn main:app --host 0.0.0.0 --port 8001

First start downloads the model (~130 MB) into the HuggingFace cache.

Environment
-----------
    EMBED_MODEL_NAME      default "BAAI/bge-small-en-v1.5"
    EMBED_BATCH_SIZE      default 32   (per encode() call; the API can accept
                                       larger lists — the service slices them)
    EMBED_NORMALIZE       "1" (default) — return unit-norm vectors so the
                                       Postgres cosine index (`<=>`) is
                                       semantically equivalent to dot product.
"""

from __future__ import annotations

import os
import time
from typing import List, Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

# `sentence_transformers` lazy-imports torch — keeping this import inside main
# so a misconfigured Python install errors loudly, not on first request.
from sentence_transformers import SentenceTransformer
import torch


MODEL_NAME = os.environ.get("EMBED_MODEL_NAME", "BAAI/bge-small-en-v1.5")
BATCH_SIZE = int(os.environ.get("EMBED_BATCH_SIZE", "32"))
NORMALIZE = os.environ.get("EMBED_NORMALIZE", "1") not in ("0", "false", "False")

# Optional auth. If EMBED_AUTH_TOKEN is set, /v1/embed requires
# `Authorization: Bearer <token>`. Leave unset for an open service (fine for
# local dev or a private network). Set it for a public Render URL so the
# endpoint can't be hammered by anyone who finds it.
AUTH_TOKEN = os.environ.get("EMBED_AUTH_TOKEN", "").strip()

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# The query prefix above is a BGE-family instruction format. Apply it ONLY for
# bge-* models. Symmetric models — e.g. sentence-transformers/all-MiniLM-L6-v2,
# which ApplyBoost runs — must NOT get it: the prefix would distort their
# query-side vectors and silently degrade matching.
IS_BGE = "bge" in MODEL_NAME.lower()

# ---- Device selection -----------------------------------------------------

def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    # Apple Silicon — Metal Performance Shaders.
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = pick_device()

# ---- Model load -----------------------------------------------------------

print(f"[embed-service] loading model={MODEL_NAME!r} device={DEVICE!r}")
t0 = time.time()
model = SentenceTransformer(MODEL_NAME, device=DEVICE)
DIMENSIONS = int(model.get_sentence_embedding_dimension())
print(f"[embed-service] ready in {time.time() - t0:.1f}s — dimensions={DIMENSIONS}")

# Sanity check for the JOBS pipeline, which writes into a vector(384) Postgres
# column — a non-384 model would corrupt that. ApplyBoost does NOT use the DB
# column (it compares vectors in-memory within one request), so any dimension is
# fine there; this is just a heads-up, not a hard failure.
if DIMENSIONS != 384:
    print(
        f"[embed-service] NOTE: model dim={DIMENSIONS} != 384. Fine for ApplyBoost "
        f"(in-memory cosine), but the jobs.embedding vector(384) column needs a "
        f"384-dim model (e.g. BAAI/bge-small-en-v1.5 or all-MiniLM-L6-v2)."
    )
print(f"[embed-service] auth={'on' if AUTH_TOKEN else 'off'} bge_prefix={IS_BGE}")


# ---- API ------------------------------------------------------------------

class EmbedRequest(BaseModel):
    input: List[str] = Field(..., max_length=512)
    input_type: Literal["document", "query"] = "document"


class EmbedItem(BaseModel):
    index: int
    embedding: List[float]


class EmbedResponse(BaseModel):
    model: str
    dimensions: int
    data: List[EmbedItem]


app = FastAPI(title="joblet embed-service", version="1.0")


def require_auth(authorization: str | None = Header(default=None)) -> None:
    """Enforce a bearer token when EMBED_AUTH_TOKEN is set; no-op otherwise."""
    if not AUTH_TOKEN:
        return
    if authorization != f"Bearer {AUTH_TOKEN}":
        raise HTTPException(status_code=401, detail="unauthorized")


@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "model": MODEL_NAME,
        "device": DEVICE,
        "dimensions": DIMENSIONS,
        "normalize": NORMALIZE,
        "bge_prefix": IS_BGE,
        "auth": bool(AUTH_TOKEN),
    }


@app.post("/v1/embed", response_model=EmbedResponse, dependencies=[Depends(require_auth)])
def embed(req: EmbedRequest) -> EmbedResponse:
    if not req.input:
        return EmbedResponse(model=MODEL_NAME, dimensions=DIMENSIONS, data=[])

    # Empty strings produce undefined-norm embeddings; replace with a single
    # space so the model returns a well-defined vector instead of NaN.
    cleaned = [t if isinstance(t, str) and t.strip() else " " for t in req.input]

    # BGE asymmetric instruction: queries get the prefix, documents do not.
    # Only for bge-* models — symmetric models (all-MiniLM) must not be prefixed.
    if req.input_type == "query" and IS_BGE:
        cleaned = [QUERY_PREFIX + t for t in cleaned]

    try:
        vectors = model.encode(
            cleaned,
            batch_size=BATCH_SIZE,
            normalize_embeddings=NORMALIZE,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
    except Exception as e:
        # Surface a clean message to the Node side so it can log / fall back.
        raise HTTPException(status_code=500, detail=f"encode_failed: {e}") from e

    data = [
        EmbedItem(index=i, embedding=vec.tolist())
        for i, vec in enumerate(vectors)
    ]
    return EmbedResponse(model=MODEL_NAME, dimensions=DIMENSIONS, data=data)
