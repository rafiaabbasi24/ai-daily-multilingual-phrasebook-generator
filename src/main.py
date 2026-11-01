import os
import json
import logging
from datetime import datetime
from typing import List, Optional

import sqlite3
from fastapi import FastAPI, HTTPException, Depends, status
from pydantic import BaseModel, Field, validator
from transformers import pipeline, Pipeline
from jina import Document, DocumentArray  # Jina AI embeddings
from sentence_transformers import SentenceTransformer  # fallback for embeddings

# ----------------------------------------------------------------------
# Configuration & Logging
# ----------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATABASE_PATH = os.getenv("PHRASEBOOK_DB", "phrasebook.db")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# ----------------------------------------------------------------------
# Database utilities
# ----------------------------------------------------------------------
def get_connection():
    """Return a new SQLite connection with row factory for dict‑like access."""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Create the required tables if they do not exist."""
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS phrasebooks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_lang TEXT NOT NULL,
                target_lang TEXT NOT NULL,
                context TEXT,
                phrases TEXT NOT NULL,
                translations TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.commit()
    logger.info("Database initialised at %s", DATABASE_PATH)

# ----------------------------------------------------------------------
# Pydantic models (request / response schemas)
# ----------------------------------------------------------------------
class PhrasebookCreateRequest(BaseModel):
    source_lang: str = Field(..., description="ISO 639‑1 code of the source language")
    target_lang: str = Field(..., description="ISO 639‑1 code of the target language")
    context: Optional[List[str]] = Field(
        default_factory=list,
        description="Optional list of sentences describing the situation"
    )
    phrases: List[str] = Field(..., description="List of source‑language phrases to translate")

    @validator("source_lang", "target_lang")
    def lang_must_be_two_chars(cls, v):
        if len(v) != 2:
            raise ValueError("Language codes must be two characters (ISO 639‑1)")
        return v.lower()


class PhrasebookResponse(BaseModel):
    id: int
    source_lang: str
    target_lang: str
    context: List[str]
    phrases: List[str]
    translations: List[str]
    created_at: datetime


class HealthResponse(BaseModel):
    status: str = "ok"


# ----------------------------------------------------------------------
# Core NLP utilities
# ----------------------------------------------------------------------
class Translator:
    """
    Wrapper around HuggingFace translation pipelines.
    The pipeline is lazily loaded per language pair to avoid loading
    unnecessary models.
    """

    _pipelines: dict[tuple[str, str], Pipeline] = {}

    @classmethod
    def _load_pipeline(cls, src: str, tgt: str) -> Pipeline:
        model_name = f"Helsinki-NLP/opus-mt-{src}-{tgt}"
        logger.info("Loading translation model %s", model_name)
        try:
            return pipeline("translation", model=model_name)
        except Exception as exc:
            logger.error("Failed to load model %s: %s", model_name, exc)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Translation model for {src}->{tgt} could not be loaded.",
            ) from exc

    @classmethod
    def get_pipeline(cls, src: str, tgt: str) -> Pipeline:
        key = (src, tgt)
        if key not in cls._pipelines:
            cls._pipelines[key] = cls._load_pipeline(src, tgt)
        return cls._pipelines[key]

    @classmethod
    def translate(cls, phrases: List[str], src: str, tgt: str) -> List[str]:
        pipe = cls.get_pipeline(src, tgt)
        # The pipeline returns a list of dicts with a "translation_text" key
        results = pipe(phrases)
        return [r["translation_text"] for r in results]


class EmbeddingProvider:
    """
    Provides sentence embeddings using Jina AI. If Jina's remote service
    is unavailable, falls back to a local SentenceTransformer model.
    """

    def __init__(self):
        try:
            # Jina's DocumentArray can embed using a remote executor if
            # the environment variable JINA_EMBEDDING_ENDPOINT is set.
            self._use_jina = bool(os.getenv("JINA_EMBEDDING_ENDPOINT"))
            if self._use_jina:
                logger.info("Using Jina AI remote embedding endpoint.")
        except Exception:
            self._use_jina = False

        if not self._use_jina:
            logger.info("Falling back to local SentenceTransformer model: %s", EMBEDDING_MODEL_NAME)
            self._local_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    def embed(self, texts: List[str]) -> List[List[float]]:
        if self._use_jina:
            docs = DocumentArray([Document(text=t) for t in texts])
            docs.embed()  # Uses the default Jina embedding executor
            return [doc.embedding.tolist() for doc in docs]
        else:
            return self._local_model.encode(texts, convert_to_numpy=False).tolist()


embedding_provider = EmbeddingProvider()

# ----------------------------------------------------------------------
# FastAPI application
# ----------------------------------------------------------------------
app = FastAPI(
    title="Context‑Aware Multilingual Phrasebook Generator",
    description="Generate phrasebooks that respect a given context using modern LLMs and embeddings.",
    version="1.0.0",
)


@app.on_event("startup")
def on_startup():
    """Initialise the SQLite database when the service starts."""
    init_db()


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Simple health endpoint used by orchestration tools."""
    return HealthResponse()


# ----------------------------------------------------------------------
# Phrasebook endpoints
# ----------------------------------------------------------------------
def _store_phrasebook(
    source_lang: str,
    target_lang: str,
    context: List[str],
    phrases: List[str],
    translations: List[str],
) -> int:
    """Persist a phrasebook and return its generated identifier."""
    with get_connection() as conn:
        cur = conn.execute(
            """
            INSERT INTO phrasebooks (
                source_lang, target_lang, context, phrases, translations, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                source_lang,
                target_lang,
                json.dumps(context, ensure_ascii=False),
                json.dumps(phrases, ensure_ascii=False),
                json.dumps(translations, ensure_ascii=False),
                datetime.utcnow().isoformat(),
            ),
        )
        conn.commit()
        return cur.lastrowid


def _fetch_phrasebook(phrasebook_id: int) -> Optional[dict]:
    """Retrieve a phrasebook row as a dict, or None if it does not exist."""
    with get_connection() as conn:
        cur = conn.execute(
            "SELECT * FROM phrasebooks WHERE id = ?", (phrasebook_id,)
        )
        row = cur.fetchone()
        if row is None:
            return None
        return dict(row)


@app.post(
    "/phrasebook",
    response_model=PhrasebookResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Phrasebook"],
)
def create_phrasebook(request: PhrasebookCreateRequest):
    """
    Generate a phrasebook for the supplied phrases, respecting the optional context.
    The context is used to compute an embedding that can be later leveraged for
    similarity‑based ranking (future extension). For now, we simply translate
    each phrase.
    """
    logger.info(
        "Received phrasebook generation request: %s -> %s (%d phrases)",
        request.source_lang,
        request.target_lang,
        len(request.phrases),
    )

    # Compute embeddings for the context (currently not used in ranking)
    if request.context:
        try:
            context_embeddings = embedding_provider.embed(request.context)
            logger.debug("Context embeddings computed (first vector length %d)", len(context_embeddings[0]))
        except Exception as exc:
            logger.warning("Failed to compute context embeddings: %s", exc)

    # Translate the phrases
    try:
        translations = Translator.translate(request.phrases, request.source_lang, request.target_lang)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unexpected error during translation")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Translation failed due to an internal error.",
        ) from exc

    phrasebook_id = _store_phrasebook(
        source_lang=request.source_lang,
        target_lang=request.target_lang,
        context=request.context,
        phrases=request.phrases,
        translations=translations,
    )

    response = PhrasebookResponse(
        id=phrasebook_id,
        source_lang=request.source_lang,
        target_lang=request.target_lang,
        context=request.context,
        phrases=request.phrases,
        translations=translations,
        created_at=datetime.utcnow(),
    )
    logger.info("Phrasebook %d created successfully", phrasebook_id)
    return response


@app.get(
    "/phrasebook/{phrasebook_id}",
    response_model=PhrasebookResponse,
    tags=["Phrasebook"],
)
def get_phrasebook(phrasebook_id: int):
    """
    Retrieve a previously generated phrasebook by its identifier.
    """
    logger.info("Fetching phrasebook %d", phrasebook_id)
    row = _fetch_phrasebook(phrasebook_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Phrasebook not found")

    response = PhrasebookResponse(
        id=row["id"],
        source_lang=row["source_lang"],
        target_lang=row["target_lang"],
        context=json.loads(row["context"] or "[]"),
        phrases=json.loads(row["phrases"]),
        translations=json.loads(row["translations"]),
        created_at=datetime.fromisoformat(row["created_at"]),
    )
    return response

# ----------------------------------------------------------------------
# Optional: endpoint to compute similarity scores (future extension)
# ----------------------------------------------------------------------
@app.post(
    "/similarity",
    tags=["Utility"],
    status_code=status.HTTP_200_OK,
)
def compute_similarity(
    texts: List[str] = Field(..., description="List of texts to compare"),
    reference: str = Field(..., description="Reference text for similarity"),
):
    """
    Compute cosine similarity between each text in ``texts`` and the ``reference``.
    This uses the same embedding provider as the phrasebook generation.
    """
    if not texts:
        raise HTTPException(status_code=400, detail="The 'texts' list must not be empty")

    try:
        embeddings = embedding_provider.embed([reference] + texts)
    except Exception as exc:
        logger.exception("Embedding failure")
        raise HTTPException(
            status_code=500,
            detail="Failed to compute embeddings.",
        ) from exc

    import numpy as np

    ref_vec = np.array(embeddings[0])
    other_vecs = np.array(embeddings[1:])
    # Normalise vectors
    ref_norm = ref_vec / np.linalg.norm(ref_vec)
    other_norm = other_vecs / np.linalg.norm(other_vecs, axis=1, keepdims=True)
    cosine_sim = (other_norm @ ref_norm).tolist()

    return {"similarities": cosine_sim}