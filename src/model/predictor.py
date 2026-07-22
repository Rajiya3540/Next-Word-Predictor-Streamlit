"""
src/model/predictor.py
======================
Next-word prediction: preprocessing, inference, top-k with confidence.

Public API
----------
preprocess_seed     : apply same cleaning as training to a seed phrase
predict_next_words  : return top-k {"word", "confidence"} dicts
"""

from __future__ import annotations

import logging
import re
import string

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences

import config

logger = logging.getLogger(__name__)


# ── Seed Preprocessing ────────────────────────────────────────────────────────

def preprocess_seed(text: str) -> str:
    """
    Apply the same cleaning as training to a seed phrase.

    Must match clean_text() in preprocessor.py exactly so that
    inference is consistent with the training distribution.

    Differences from clean_text():
      - No Gutenberg boilerplate removal (user input never contains it)
      - No chapter heading removal (same reason)
      - Otherwise identical: lowercase → strip punctuation → collapse spaces

    Parameters
    ----------
    text : str  Raw seed text from user input.

    Returns
    -------
    str  Cleaned lowercase text, or "" if input is blank.
    """
    if not text or not text.strip():
        return ""

    text = text.lower()

    # Keep exactly the same characters allowed during training
    allowed = set(string.ascii_lowercase + string.digits + " '")
    text = "".join(ch if ch in allowed else " " for ch in text)

    return re.sub(r"\s+", " ", text).strip()


# ── Prediction ────────────────────────────────────────────────────────────────

def predict_next_words(
    seed_text : str,
    model     : Sequential,
    tokenizer : Tokenizer,
    max_len   : int,
    top_k     : int = config.TOP_K,
    oov_token : str = config.OOV_TOKEN,
) -> list[dict[str, object]]:
    """
    Predict the top-k most likely next words for a seed phrase.

    Bug fixes applied
    -----------------
    BUG 1 — O(n^2) reverse vocab rebuild on every call
        BEFORE: {v: k for k, v in tokenizer.word_index.items()}
                Iterates entire vocab on every prediction call.
                At 10K vocab x 200 keypresses = 2M wasted iterations.
        AFTER:  tokenizer.index_word.get(int(idx), "")
                tokenizer.index_word is pre-built by Keras — O(1) per lookup.

    BUG 2 — Dead code with undefined variable
        BEFORE: return results on line N; then a second `results` block
                on line N+1 referencing `index_to_word` (never defined)
                → NameError if ever reached.
        AFTER:  Dead block removed. Single clean loop before return.

    BUG 3 — Unknown words silently dropped
        BEFORE: Words missing from vocab disappeared from token_list,
                changing the sequence length and corrupting padding.
        AFTER:  Tokenizer was fitted with oov_token so unknown words
                map to the OOV index. Padding length stays correct.
                OOV is filtered from OUTPUT predictions, not INPUT tokens.

    BUG 4 — Padding token (index 0) in output
        BEFORE: Index 0 mapped to "" via index_word and was included
                in predictions as a blank entry.
        AFTER:  Explicit `if idx == 0: continue` guard.

    BUG 5 — Seed not preprocessed before tokenization
        BEFORE: Raw "Alice Was" sent to tokenizer trained on "alice was".
                Mismatch caused vocabulary lookup failures for every word.
        AFTER:  preprocess_seed() called first — matches training exactly.

    Parameters
    ----------
    seed_text : str         Raw user input (one or more words).
    model     : Sequential  Trained Keras model. Returns [] if None.
    tokenizer : Tokenizer   Fitted tokenizer from training or loaded from disk.
    max_len   : int         Padded sequence length the model expects.
    top_k     : int         Number of predictions to return.
    oov_token : str         OOV placeholder — filtered from results.

    Returns
    -------
    list[dict]
        Each entry: {"word": str, "confidence": float (0.0 to 1.0)}
        Sorted by confidence descending.
        Returns [] when: model is None, seed is empty, or no known
        vocabulary tokens exist in the cleaned seed.
    """
    # ── Guard clauses ─────────────────────────────────────────────────────────

    if model is None:
        logger.warning("predict_next_words: model is None")
        return []

    if not tokenizer.word_index:
        logger.warning("predict_next_words: tokenizer is empty")
        return []

    # ── Preprocess (BUG 5 FIX) ────────────────────────────────────────────────

    cleaned = preprocess_seed(seed_text)
    if not cleaned:
        return []

    # ── Tokenize (BUG 3 FIX) ──────────────────────────────────────────────────

    # Unknown words map to oov_token index — sequence length stays correct
    token_list = tokenizer.texts_to_sequences([cleaned])[0]

    if not token_list:
        logger.debug("Seed '%s' produced no tokens", seed_text)
        return []

    # ── Pad to model input length ─────────────────────────────────────────────

    padded = pad_sequences(
        [token_list],
        maxlen  = max_len - 1,
        padding = "pre",        # left-pad — matches how training sequences were padded
    )

    # ── Inference ─────────────────────────────────────────────────────────────

    # probs shape: (vocab_size,) — softmax probability over all words
    probs = model.predict(padded, verbose=0)[0]

    # Fetch extra candidates to have room to filter OOV + padding
    n_fetch     = min(top_k + 10, len(probs))
    top_indices = np.argsort(probs)[-n_fetch:][::-1]   # highest prob first

    # ── Build results (BUG 1 + 2 + 4 FIX) ────────────────────────────────────

    results: list[dict[str, object]] = []

    for idx in top_indices:

        if idx == 0:                                  # BUG 4 FIX: skip padding
            continue

        word = tokenizer.index_word.get(int(idx), "") # BUG 1 FIX: O(1) lookup

        if not word or word == oov_token:             # skip OOV in output
            continue

        results.append({
            "word"       : word,
            "confidence" : round(float(probs[idx]), 4),
        })

        if len(results) == top_k:
            break
    # BUG 2 FIX: no dead code here — single return below

    logger.debug(
        "predict('%s') → %s",
        seed_text,
        [(r["word"], f"{r['confidence']:.1%}") for r in results],
    )
    return results
