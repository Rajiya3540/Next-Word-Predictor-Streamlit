"""
src/data/preprocessor.py
========================
Complete data pipeline: clean → tokenize → sequences → dataset.

Public API
----------
clean_text         : clean raw corpus text
build_tokenizer    : fit a Keras Tokenizer on cleaned text
generate_sequences : prefix n-gram sequences from a token list
prepare_dataset    : pad + X/y split (CORRECT: y = padded[:, -1])
build_dataset      : full pipeline from corpus file to (X, y, tokenizer, ...)
"""

from __future__ import annotations

import logging
import re
import string
from pathlib import Path

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences, to_categorical

import config

logger = logging.getLogger(__name__)


# ── Step 1: Text Cleaning ─────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Clean raw corpus text for NLP training.

    Pipeline
    --------
    1. Lowercase          → 'The'/'THE'/'the' become one token
    2. Gutenberg strip    → remove header/footer boilerplate (~500 lines)
    3. Chapter headings   → 'chapter i', 'chapter 3' removed
    4. Punctuation strip  → keep only letters, digits, spaces, apostrophes
                            Apostrophes kept so "don't" stays one token
    5. Whitespace collapse → multiple spaces/newlines → single space

    Parameters
    ----------
    text : str  Raw corpus string (any encoding already decoded).

    Returns
    -------
    str  Clean lowercase text, ready for tokenization.
    """
    text = text.lower()

    # Remove Project Gutenberg header/footer between *** markers
    start = re.search(r"\*\*\* start of (the|this) project gutenberg", text)
    end   = re.search(r"\*\*\* end of (the|this) project gutenberg",   text)
    if start:
        text = text[start.end():]
    if end:
        text = text[: end.start()]

    # Remove chapter/section headings
    text = re.sub(r"\bchapter\s+[ivxlcdm\d]+\b", "", text)

    # Keep only safe characters; replace everything else with space
    allowed = set(string.ascii_lowercase + string.digits + " \n'")
    text = "".join(ch if ch in allowed else " " for ch in text)

    # Collapse all whitespace to a single space
    return re.sub(r"\s+", " ", text).strip()


# ── Step 2: Tokenization ──────────────────────────────────────────────────────

def build_tokenizer(
    text: str,
    max_vocab: int = config.MAX_VOCAB_SIZE,
    oov_token: str = config.OOV_TOKEN,
) -> Tokenizer:
    """
    Fit and return a Keras Tokenizer on cleaned text.

    Design choices
    --------------
    num_words=max_vocab  : caps vocab to most frequent N words; rare words
                           that appear once hurt generalisation
    oov_token            : unknown words at inference get a valid index
                           instead of being silently dropped
    filters=''           : text is already cleaned; don't double-process
    lower=False          : already lowercased in clean_text()

    Parameters
    ----------
    text      : str  Already-cleaned corpus.
    max_vocab : int  Vocabulary size cap (default from config).
    oov_token : str  Placeholder for unknown tokens (default '<OOV>').

    Returns
    -------
    Tokenizer  Fitted Keras Tokenizer.
    """
    tokenizer = Tokenizer(
        num_words = max_vocab,
        oov_token = oov_token,
        filters   = "",
        lower     = False,
    )
    tokenizer.fit_on_texts([text])

    effective_vocab = min(len(tokenizer.word_index) + 1, max_vocab + 1)
    logger.info(
        "Tokenizer fitted | unique_words=%d | effective_vocab=%d",
        len(tokenizer.word_index),
        effective_vocab,
    )
    return tokenizer


# ── Step 3: Sequence Generation ───────────────────────────────────────────────

def generate_sequences(
    token_list : list[int],
    max_seq_len: int = config.MAX_SEQ_LEN,
    min_seq_len: int = config.MIN_SEQ_LEN,
) -> list[list[int]]:
    """
    Generate prefix n-gram sequences from a token list.

    FIX — original bug (fixed 4-token window)
    -----------------------------------------
    BEFORE (wrong):
        seq = token_list[i-3 : i+1]   # always length 4
        # max_len was always 4; padding did nothing;
        # model never saw context longer than 3 words

    AFTER (correct prefix n-gram):
        seq = token_list[max(0, i - max_seq_len + 1) : i + 1]
        # lengths grow from min_seq_len up to max_seq_len
        # padding now does real work; model learns variable context

    Example for [1, 2, 3, 4] with max_seq_len=10:
        [1, 2], [1, 2, 3], [1, 2, 3, 4]   ← lengths 2, 3, 4

    Parameters
    ----------
    token_list  : list[int]  Tokenized corpus.
    max_seq_len : int        Longest prefix to generate.
    min_seq_len : int        Shortest valid sequence (1 input + 1 label).

    Returns
    -------
    list[list[int]]  Variable-length sequences, each ending in the label token.
    """
    sequences: list[list[int]] = []

    for i in range(1, len(token_list)):
        seq = token_list[max(0, i - max_seq_len + 1) : i + 1]

        if len(seq) >= min_seq_len:
            sequences.append(seq)

        # Memory safety cap for chat-mode retraining
        if len(sequences) >= config.MAX_SEQUENCES:
            logger.warning(
                "MAX_SEQUENCES cap (%d) reached — truncating.", config.MAX_SEQUENCES
            )
            break

    logger.info(
        "Sequences generated: count=%d | min_len=%d | max_len=%d",
        len(sequences),
        min(len(s) for s in sequences) if sequences else 0,
        max(len(s) for s in sequences) if sequences else 0,
    )
    return sequences


# ── Step 4: Prepare X / y ─────────────────────────────────────────────────────

def prepare_dataset(
    sequences : list[list[int]],
    vocab_size : int,
    max_len    : int | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Pad sequences and produce X (inputs) and y (one-hot labels).

    FIX — critical label bug
    ------------------------
    BEFORE (wrong): y = sequences[:, :-1]
        → y is a 2-D matrix identical to X; model trains on wrong targets

    AFTER (correct): y_raw = padded[:, -1]
        → y_raw is a 1-D vector of next-word indices — correct labels

    Padding strategy: 'pre' (zeros on the LEFT)
        RNNs read left-to-right. Pre-padding right-aligns the real tokens
        so the hidden state at the final time step holds all the context.

    Parameters
    ----------
    sequences  : list[list[int]]  Variable-length prefix sequences.
    vocab_size : int              Total vocabulary size (output layer size).
    max_len    : int | None       Pad to this length; computed if None.

    Returns
    -------
    X       : np.ndarray  shape (n, max_len-1)          — input token indices
    y       : np.ndarray  shape (n, vocab_size)          — one-hot next-word labels
    max_len : int                                         — padded length (save for inference)
    """
    if max_len is None:
        max_len = max(len(s) for s in sequences)

    padded = pad_sequences(sequences, maxlen=max_len, padding="pre")

    X     = padded[:, :-1]   # all columns except last → input tokens
    y_raw = padded[:,  -1]   # ONLY the last column   → next-word index (1-D)
    y     = to_categorical(y_raw, num_classes=vocab_size)

    logger.info(
        "Dataset prepared | X=%s | y=%s | max_len=%d",
        X.shape, y.shape, max_len,
    )
    return X, y, max_len


# ── Full Pipeline ─────────────────────────────────────────────────────────────

def build_dataset(
    corpus_path: Path = config.CORPUS_PATH,
) -> tuple[np.ndarray, np.ndarray, Tokenizer, int, int]:
    """
    Full pipeline: read corpus → clean → tokenize → sequences → (X, y).

    This is the single function called by the training pipeline and the
    Streamlit app. It ties all four steps above into one call.

    Parameters
    ----------
    corpus_path : Path  Raw text file (default: data/alice.txt).

    Returns
    -------
    X          : np.ndarray  Input sequences
    y          : np.ndarray  One-hot labels
    tokenizer  : Tokenizer   Fitted Keras tokenizer (for inference)
    vocab_size : int         Effective vocabulary size
    max_len    : int         Padded sequence length  (for inference)

    Raises
    ------
    FileNotFoundError  Corpus not found — run data/download_corpus.py
    ValueError         Corpus too short to produce any sequences
    """
    if not corpus_path.exists():
        raise FileNotFoundError(
            f"Corpus not found: '{corpus_path}'\n"
            "Fix: run  python data/download_corpus.py"
        )

    logger.info("Loading corpus: %s", corpus_path)
    raw_text = corpus_path.read_text(encoding="utf-8", errors="ignore")
    logger.info("Raw corpus: %d characters", len(raw_text))

    # 1. Clean
    cleaned = clean_text(raw_text)
    logger.info("Cleaned corpus: %d characters", len(cleaned))

    # 2. Tokenize
    tokenizer  = build_tokenizer(cleaned)
    vocab_size = min(len(tokenizer.word_index) + 1, config.MAX_VOCAB_SIZE + 1)

    # 3. Encode text to integers
    token_list = tokenizer.texts_to_sequences([cleaned])[0]
    logger.info("Token list: %d tokens", len(token_list))

    # 4. Generate sequences
    sequences = generate_sequences(token_list)
    if not sequences:
        raise ValueError(
            "No training sequences generated. "
            "The corpus may be too short (minimum ~500 words recommended)."
        )

    # 5. Prepare X, y
    X, y, max_len = prepare_dataset(sequences, vocab_size)

    logger.info(
        "build_dataset complete | samples=%d | vocab=%d | max_len=%d",
        X.shape[0], vocab_size, max_len,
    )
    return X, y, tokenizer, vocab_size, max_len
