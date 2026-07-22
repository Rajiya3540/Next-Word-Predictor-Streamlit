"""
src/model/persistence.py
========================
Save and load model weights + tokenizer vocabulary to/from disk.

Public API
----------
model_exists  : True if both model + tokenizer files exist on disk
save          : persist model (.keras) + tokenizer (.npy) to MODELS_DIR
load          : restore model + tokenizer; raises on missing/corrupt files

Schema
------
Tokenizer is stored as a plain Python dict inside a .npy file:
    {
        "schema_version" : int,
        "word_index"     : dict[str, int],
        "index_word"     : dict[int, str],
        "max_len"        : int,
        "vocab_size"     : int,
        "oov_token"      : str,
    }

Both word_index AND index_word are saved explicitly.
The original app2.py only saved word_index and rebuilt index_word
on load with {v:k for k,v in word_index.items()} — an O(n) rebuild
that could also produce wrong types if word_index had been mutated.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model as _keras_load
from tensorflow.keras.preprocessing.text import Tokenizer

import config

logger = logging.getLogger(__name__)

# Bump this when the schema changes to detect stale tokenizer files
_SCHEMA_VERSION  = 1
_REQUIRED_KEYS   = frozenset({
    "schema_version", "word_index", "index_word",
    "max_len", "vocab_size", "oov_token",
})


# ── Check ─────────────────────────────────────────────────────────────────────

def model_exists() -> bool:
    """
    Return True only when BOTH the model file and tokenizer file exist.

    A model without its tokenizer (or vice-versa) is unusable — both
    files must be present before load() is called.
    """
    exists = config.MODEL_PATH.exists() and config.TOKENIZER_PATH.exists()
    logger.debug("model_exists → %s", exists)
    return exists


# ── Save ──────────────────────────────────────────────────────────────────────

def save(
    model      : Sequential,
    tokenizer  : Tokenizer,
    max_len    : int,
    vocab_size : int,
) -> None:
    """
    Save model weights (.keras) and tokenizer data (.npy) to MODELS_DIR.

    Fixes over original app2.py
    ---------------------------
    FIX 1 — index_word not persisted
        BEFORE: only word_index saved; index_word rebuilt on load with
                {v:k for k,v in data["word_index"].items()} — O(n) per load
                and fragile if word_index contains non-string keys.
        AFTER:  both word_index and index_word saved explicitly as plain dicts.

    FIX 2 — no schema version
        BEFORE: no version field; impossible to detect stale tokenizer files
                after code changes.
        AFTER:  schema_version=1 saved; load() validates and warns on mismatch.

    FIX 3 — single source of truth for paths
        BEFORE: path strings scattered across app2.py ("saved_model.keras",
                "tokenizer_vocab.npy") and notebook ("lstm_next_word_model.keras",
                "tokenizer_data.npy") — four different names for two files.
        AFTER:  config.MODEL_PATH and config.TOKENIZER_PATH used everywhere.

    Parameters
    ----------
    model      : Sequential  Trained Keras model.
    tokenizer  : Tokenizer   Fitted Keras tokenizer.
    max_len    : int         Padded sequence length (required for inference).
    vocab_size : int         Effective vocabulary size.

    Raises
    ------
    OSError    If the MODELS_DIR is not writable.
    """
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Save Keras model in the recommended .keras format
    model.save(str(config.MODEL_PATH))
    logger.info("Model saved → %s", config.MODEL_PATH)

    # Save tokenizer as a plain dict — no pickle security concerns because
    # we only store str/int values (no arbitrary Python objects)
    payload = {
        "schema_version" : _SCHEMA_VERSION,
        "word_index"     : dict(tokenizer.word_index),   # str → int
        "index_word"     : dict(tokenizer.index_word),   # int → str
        "max_len"        : int(max_len),
        "vocab_size"     : int(vocab_size),
        "oov_token"      : str(config.OOV_TOKEN),
    }
    np.save(str(config.TOKENIZER_PATH), payload, allow_pickle=True)
    logger.info(
        "Tokenizer saved → %s  (vocab=%d, max_len=%d)",
        config.TOKENIZER_PATH, vocab_size, max_len,
    )


# ── Load ──────────────────────────────────────────────────────────────────────

def load() -> tuple[Sequential, Tokenizer, int, int]:
    """
    Load model and tokenizer from disk.

    Fixes over original app2.py
    ---------------------------
    FIX 1 — no existence check before loading
        BEFORE: load_model() called without checking if file exists →
                unhandled OSError with confusing traceback.
        AFTER:  FileNotFoundError with a clear user-facing message.

    FIX 2 — no schema validation
        BEFORE: data["word_index"] raised KeyError if key was missing;
                data["max_len"] raised KeyError with no explanation.
        AFTER:  all REQUIRED_KEYS checked upfront; ValueError lists
                exactly which keys are missing.

    FIX 3 — index_word not restored correctly
        BEFORE: tokenizer.index_word rebuilt manually from word_index;
                int keys from JSON/npy round-trip could become strings,
                breaking tokenizer.index_word.get(int(idx)) lookups.
        AFTER:  index_word loaded directly from saved dict; keys cast to
                int explicitly to guarantee correct lookup type.

    Returns
    -------
    model      : Sequential  Loaded Keras model.
    tokenizer  : Tokenizer   Reconstructed Keras tokenizer.
    max_len    : int         Padded sequence length for inference.
    vocab_size : int         Effective vocabulary size.

    Raises
    ------
    FileNotFoundError  Model or tokenizer file missing.
    ValueError         Tokenizer file has wrong schema or missing keys.
    """
    # ── Existence checks ──────────────────────────────────────────────────────

    if not config.MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {config.MODEL_PATH}\n"
            "Fix: train the model first, or copy lstm_best_model.keras "
            "into the models/ folder."
        )

    if not config.TOKENIZER_PATH.exists():
        raise FileNotFoundError(
            f"Tokenizer file not found: {config.TOKENIZER_PATH}\n"
            "Fix: train the model first, or copy tokenizer_data.npy "
            "into the models/ folder."
        )

    # ── Load Keras model ──────────────────────────────────────────────────────

    model = _keras_load(str(config.MODEL_PATH))
    logger.info("Model loaded ← %s", config.MODEL_PATH)

    # ── Load tokenizer data ───────────────────────────────────────────────────

    raw = np.load(str(config.TOKENIZER_PATH), allow_pickle=True).item()

    if not isinstance(raw, dict):
        raise ValueError(
            f"Tokenizer file is corrupt (expected dict, got {type(raw).__name__}).\n"
            "Fix: delete models/tokenizer_data.npy and retrain."
        )

    # Schema validation — all required keys must be present
    missing = _REQUIRED_KEYS - set(raw.keys())
    if missing:
        raise ValueError(
            f"Tokenizer file is missing keys: {missing}\n"
            "Fix: delete both model files and retrain."
        )

    # Schema version warning (non-fatal for now)
    saved_version = raw.get("schema_version", 0)
    if saved_version != _SCHEMA_VERSION:
        logger.warning(
            "Tokenizer schema version mismatch: file=%d, expected=%d. "
            "Consider retraining.",
            saved_version, _SCHEMA_VERSION,
        )

    # ── Reconstruct Tokenizer ─────────────────────────────────────────────────

    tokenizer = Tokenizer(oov_token=raw["oov_token"])

    # Assign both dicts directly — no O(n) rebuild needed
    tokenizer.word_index = {str(k): int(v) for k, v in raw["word_index"].items()}

    # Cast index_word keys to int explicitly — numpy/JSON round-trips can
    # store them as strings, which breaks tokenizer.index_word.get(int(idx))
    tokenizer.index_word = {int(k): str(v) for k, v in raw["index_word"].items()}

    vocab_size = int(raw["vocab_size"])
    max_len    = int(raw["max_len"])

    logger.info(
        "Tokenizer loaded ← %s  (vocab=%d, max_len=%d)",
        config.TOKENIZER_PATH, vocab_size, max_len,
    )
    return model, tokenizer, max_len, vocab_size