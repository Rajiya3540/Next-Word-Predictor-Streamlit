"""
src/model/predictor.py
======================

Next-word prediction module.

Responsibilities:
- Clean and preprocess seed text
- Convert text into tokenizer sequences
- Run model inference
- Return top-k next-word predictions
- Remove invalid predictions such as padding/OOV
- Reduce extremely low-confidence predictions

Public API
----------
preprocess_seed()
predict_next_words()
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


# ============================================================================
# SEED PREPROCESSING
# ============================================================================

def preprocess_seed(text: str) -> str:
    """
    Apply the same basic cleaning used during training.

    Steps:
    1. Convert to lowercase
    2. Keep letters, digits, spaces and apostrophes
    3. Replace unwanted characters with spaces
    4. Collapse multiple spaces
    """

    if not text or not text.strip():
        return ""

    # Lowercase
    text = text.lower()

    # Allowed characters
    allowed = set(
        string.ascii_lowercase
        + string.digits
        + " '"
    )

    # Remove unwanted characters
    text = "".join(
        ch if ch in allowed else " "
        for ch in text
    )

    # Remove multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ============================================================================
# NEXT WORD PREDICTION
# ============================================================================

def predict_next_words(
    seed_text: str,
    model: Sequential,
    tokenizer: Tokenizer,
    max_len: int,
    top_k: int = config.TOP_K,
    oov_token: str = config.OOV_TOKEN,
) -> list[dict[str, object]]:
    """
    Predict the most likely next words.

    Returns:
    [
        {
            "word": "alice",
            "confidence": 0.45
        }
    ]
    """

    # ------------------------------------------------------------------------
    # GUARD CONDITIONS
    # ------------------------------------------------------------------------

    if model is None:
        logger.warning(
            "predict_next_words: model is None"
        )
        return []

    if tokenizer is None:
        logger.warning(
            "predict_next_words: tokenizer is None"
        )
        return []

    if not tokenizer.word_index:
        logger.warning(
            "predict_next_words: tokenizer is empty"
        )
        return []

    if max_len <= 1:
        logger.warning(
            "predict_next_words: invalid max_len=%s",
            max_len
        )
        return []

    # ------------------------------------------------------------------------
    # PREPROCESS INPUT
    # ------------------------------------------------------------------------

    cleaned = preprocess_seed(seed_text)

    if not cleaned:
        return []

    # ------------------------------------------------------------------------
    # TOKENIZE
    # ------------------------------------------------------------------------

    token_list = tokenizer.texts_to_sequences(
        [cleaned]
    )[0]

    if not token_list:
        logger.debug(
            "Seed '%s' produced no tokens",
            seed_text
        )
        return []

    # ------------------------------------------------------------------------
    # PAD INPUT
    # ------------------------------------------------------------------------

    padded = pad_sequences(
        [token_list],
        maxlen=max_len - 1,
        padding="pre",
        truncating="pre",
    )

    # ------------------------------------------------------------------------
    # MODEL PREDICTION
    # ------------------------------------------------------------------------

    try:

        probs = model.predict(
            padded,
            verbose=0
        )[0].astype(np.float64)

    except Exception as exc:

        logger.exception(
            "Prediction failed: %s",
            exc
        )

        return []

    # ------------------------------------------------------------------------
    # CLEAN INVALID PREDICTIONS
    # ------------------------------------------------------------------------

    # Padding index
    if len(probs) > 0:
        probs[0] = 0.0

    # OOV token
    oov_index = tokenizer.word_index.get(
        oov_token
    )

    if (
        oov_index is not None
        and oov_index < len(probs)
    ):
        probs[oov_index] = 0.0

    # Remove extremely low probability predictions
    probs[probs < 0.001] = 0.0

    # ------------------------------------------------------------------------
    # NORMALIZE PROBABILITIES
    # ------------------------------------------------------------------------

    total_probability = probs.sum()

    if total_probability <= 0:
        return []

    probs = probs / total_probability

    # ------------------------------------------------------------------------
    # GET TOP CANDIDATES
    # ------------------------------------------------------------------------

    n_fetch = min(
        top_k + 10,
        len(probs)
    )

    top_indices = np.argsort(
        probs
    )[-n_fetch:][::-1]

    # ------------------------------------------------------------------------
    # BUILD RESULTS
    # ------------------------------------------------------------------------

    results: list[dict[str, object]] = []

    for idx in top_indices:

        idx = int(idx)

        # Skip padding
        if idx == 0:
            continue

        # Get word
        word = tokenizer.index_word.get(
            idx,
            ""
        )

        # Skip invalid words
        if not word:
            continue

        # Skip OOV
        if word == oov_token:
            continue

        confidence = float(
            probs[idx]
        )

        # Skip zero probability
        if confidence <= 0:
            continue

        results.append(
            {
                "word": word,
                "confidence": round(
                    confidence,
                    4
                ),
            }
        )

        # Stop after top-k
        if len(results) >= top_k:
            break

    # ------------------------------------------------------------------------
    # LOG
    # ------------------------------------------------------------------------

    logger.debug(
        "Prediction for '%s': %s",
        seed_text,
        [
            (
                item["word"],
                f"{item['confidence']:.1%}"
            )
            for item in results
        ],
    )

    return results