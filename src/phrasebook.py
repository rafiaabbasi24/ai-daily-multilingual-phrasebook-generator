import sqlite3
import json
import threading
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
from transformers import pipeline, Pipeline
from jina import Document, DocumentArray

# --------------------------------------------------------------------------- #
# Helper functions for embedding handling
# --------------------------------------------------------------------------- #
def _embed_text(text: str, model_name: str = "jinaai/jina-embeddings-v2-base-en") -> np.ndarray:
    """
    Generate a dense vector embedding for the given text using Jina AI embeddings.

    Args:
        text: The input string to embed.
        model_name: The name of the Jina embedding model to use.

    Returns:
        A 1‑D NumPy array containing the embedding.
    """
    doc = Document(text=text)
    da = DocumentArray([doc])
    da.embed(embedding_model=model_name)  # Mutates the DocumentArray in‑place
    return np.asarray(da[0].embedding, dtype=np.float32)


def _serialize_embedding(emb: np.ndarray) -> bytes:
    """Serialize a NumPy embedding to bytes for SQLite storage."""
    return emb.tobytes()


def _deserialize_embedding(blob: bytes, dim: int) -> np.ndarray:
    """Deserialize bytes from SQLite back into a NumPy embedding."""
    return np.frombuffer(blob, dtype=np.float32, count=dim)


# --------------------------------------------------------------------------- #
# Core Phrasebook logic
# --------------------------------------------------------------------------- #
class Phrasebook:
    """
    Core class that handles scenario classification and multilingual phrase generation.
    It uses:
        * Zero‑shot classification (BART‑MNLI) to infer the scenario.
        * A seq2seq model (MT5‑small) to translate English phrases.
        * Jina AI embeddings for similarity look‑ups and caching.
        * SQLite for persistent storage of scenarios, generated phrases and embeddings.
    """

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS scenarios (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL UNIQUE,
        description TEXT
    );
    CREATE TABLE IF NOT EXISTS phrases (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        scenario_id INTEGER NOT NULL,
        language TEXT NOT NULL,
        phrase TEXT NOT NULL,
        embedding BLOB NOT NULL,
        FOREIGN KEY(scenario_id) REFERENCES scenarios(id),
        UNIQUE(scenario_id, language, phrase)
    );
    """

    # A small static mapping of scenario → example English phrases.
    # In a real product this would be a separate data source.
    _BASE_PHRASES: Dict[str, List[str]] = {
        "greeting": [
            "Hello!",
            "Good morning.",
            "Good evening.",
            "How are you?",
            "Nice to meet you."
        ],
        "ordering_food": [
            "I would like to order a pizza.",
            "Can I have a glass of water?",
            "What are today's specials?",
            "Could you bring the check, please?",
            "Do you have any vegetarian options?"
        ],
        "asking_directions": [
            "Excuse me, where is the nearest subway station?",
            "How do I get to the museum?",
            "Is there a pharmacy nearby?",
            "Can you show me the way to the hotel?",
            "Which bus should I take to the airport?"
        ],
    }

    def __init__(self, db_path: str = "data/phrasebook.db"):
        """
        Initialise the Phrasebook.

        Args:
            db_path: Path to the SQLite database file. The directory will be created if missing.
        """
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        # Thread‑safety for SQLite connections
        self._lock = threading.Lock()
        self._init_db()

        # Load transformer pipelines (lazy loading ensures they are only created once)
        self._classifier: Pipeline = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=self._select_device()
        )
        self._translator: Pipeline = pipeline(
            "text2text-generation",
            model="google/mt5-small",
            device=self._select_device()
        )

        # Cache of scenario IDs to avoid repeated DB look‑ups
        self._scenario_id_cache: Dict[str, int] = {}

    @staticmethod
    def _select_device() -> int:
        """Return the appropriate device index for transformers (CPU = -1)."""
        try:
            import torch

            return 0 if torch.cuda.is_available() else -1
        except Exception:
            return -1

    # ----------------------------------------------------------------------- #
    # Database handling
    # ----------------------------------------------------------------------- #
    def _init_db(self) -> None:
        """Create tables if they do not already exist."""
        with self._get_connection() as conn:
            conn.executescript(self._SCHEMA)

    def _get_connection(self) -> sqlite3.Connection:
        """Return a new SQLite connection with row factory set to dict."""
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_scenario(self, name: str, description: Optional[str] = None) -> int:
        """
        Insert a scenario into the DB if it does not exist and return its ID.

        Args:
            name: Scenario name (must be unique).
            description: Optional longer description.

        Returns:
            Integer primary key of the scenario.
        """
        if name in self._scenario_id_cache:
            return self._scenario_id_cache[name]

        with self._lock, self._get_connection() as conn:
            cur = conn.execute(
                "SELECT id FROM scenarios WHERE name = ?", (name,)
            )
            row = cur.fetchone()
            if row:
                scenario_id = row["id"]
            else:
                cur = conn.execute(
                    "INSERT INTO scenarios (name, description) VALUES (?, ?)",
                    (name, description or ""),
                )
                scenario_id = cur.lastrowid
                conn.commit()
            self._scenario_id_cache[name] = scenario_id
            return scenario_id

    # ----------------------------------------------------------------------- #
    # Scenario classification
    # ----------------------------------------------------------------------- #
    def classify_scenario(self, text: str) -> Tuple[str, float]:
        """
        Determine the most likely scenario for a given user utterance.

        Args:
            text: The raw user input.

        Returns:
            A tuple ``(scenario_name, confidence)`` where ``scenario_name`` is one
            of the keys in ``_BASE_PHRASES`` and ``confidence`` is the model's
            probability score (0‑1).
        """
        candidate_labels = list(self._BASE_PHRASES.keys())
        result = self._classifier(
            sequences=text,
            candidate_labels=candidate_labels,
            multi_label=False,
        )
        # The pipeline returns labels ordered by score
        best_label = result["labels"][0]
        best_score = float(result["scores"][0])
        return best_label, best_score

    # ----------------------------------------------------------------------- #
    # Phrase generation
    # ----------------------------------------------------------------------- #
    def _translate_phrase(self, phrase_en: str, target_lang: str) -> str:
        """
        Translate an English phrase to the target language using the MT5 model.

        Args:
            phrase_en: Phrase in English.
            target_lang: ISO‑639‑1 language code (e.g., ``'fr'`` for French).

        Returns:
            Translated phrase as a string.
        """
        prompt = f"translate English to {target_lang}: {phrase_en}"
        result = self._translator(prompt, max_length=64, clean_up_tokenization_spaces=True)
        return result[0]["generated_text"].strip()

    def generate_phrases(
        self,
        scenario: str,
        target_lang: str,
        num_phrases: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        Generate a list of multilingual phrases for a given scenario.

        The method first checks the SQLite cache. If enough cached phrases exist,
        they are returned directly. Otherwise, new translations are generated,
        stored, and then returned.

        Args:
            scenario: Name of the scenario (must exist in ``_BASE_PHRASES``).
            target_lang: Target language ISO‑639‑1 code.
            num_phrases: Desired number of phrases to return (max 20).

        Returns:
            A list of tuples ``(phrase, similarity)`` where ``similarity`` is the
            cosine similarity between the stored embedding and the query embedding.
        """
        if scenario not in self._BASE_PHRASES:
            raise ValueError(f"Unknown scenario '{scenario}'. Available: {list(self._BASE_PHRASES)}")

        num_phrases = min(max(1, num_phrases), 20)  # clamp to sensible range
        scenario_id = self._ensure_scenario(scenario)

        # Compute query embedding (English version of the scenario description)
        query_text = f"{scenario} in {target_lang}"
        query_emb = _embed_text(query_text)

        with self._lock, self._get_connection() as conn:
            # Retrieve cached phrases for this scenario+language
            cur = conn.execute(
                """
                SELECT phrase, embedding FROM phrases
                WHERE scenario_id = ? AND language = ?
                """,
                (scenario_id, target_lang),
            )
            rows = cur.fetchall()

            cached = []
            for row in rows:
                emb = _deserialize_embedding(row["embedding"], dim=query_emb.shape[0])
                # Cosine similarity
                sim = float(
                    np.dot(query_emb, emb) / (np.linalg.norm(query_emb) * np.linalg.norm(emb) + 1e-10)
                )
                cached.append((row["phrase"], sim))

            # If we already have enough cached phrases, sort by similarity and return
            if len(cached) >= num_phrases:
                cached.sort(key=lambda x: x[1], reverse=True)
                return cached[:num_phrases]

            # Need to generate more phrases
            needed = num_phrases - len(cached)
            english_phrases = self._BASE_PHRASES[scenario][:needed]

            new_entries: List[Tuple[str, np.ndarray]] = []
            for en_phrase in english_phrases:
                translated = self._translate_phrase(en_phrase, target_lang)
                emb = _embed_text(translated)
                new_entries.append((translated, emb))

                # Store immediately to avoid race conditions
                conn.execute(
                    """
                    INSERT OR IGNORE INTO phrases (scenario_id, language, phrase, embedding)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        scenario_id,
                        target_lang,
                        translated,
                        _serialize_embedding(emb),
                    ),
                )
            conn.commit()

            # Combine cached and newly generated entries
            all_phrases = cached + [
                (phrase, float(np.dot(query_emb, emb) / (np.linalg.norm(query_emb) * np.linalg.norm(emb) + 1e-10)))
                for phrase, emb in new_entries
            ]
            all_phrases.sort(key=lambda x: x[1], reverse=True)
            return all_phrases[:num_phrases]

    # ----------------------------------------------------------------------- #
    # Utility for external callers (e.g., FastAPI routes)
    # ----------------------------------------------------------------------- #
    def get_scenarios(self) -> List[Dict[str, str]]:
        """
        Return a list of all known scenarios with their descriptions.

        Returns:
            List of dictionaries ``{'name': ..., 'description': ...}``.
        """
        with self._get_connection() as conn:
            cur = conn.execute("SELECT name, description FROM scenarios")
            return [{"name": row["name"], "description": row["description"]} for row in cur.fetchall()]

    def add_custom_scenario(self, name: str, description: str, base_phrases: List[str]) -> None:
        """
        Add a new scenario at runtime. The provided English phrases will be used
        as the source pool for translation.

        Args:
            name: Unique scenario identifier.
            description: Human‑readable description.
            base_phrases: List of English phrases that belong to the scenario.
        """
        if name in self._BASE_PHRASES:
            raise ValueError(f"Scenario '{name}' already exists.")
        self._BASE_PHRASES[name] = base_phrases
        self._ensure_scenario(name, description)

# End of src/phrasebook.py