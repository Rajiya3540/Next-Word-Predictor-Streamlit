"""
src/model/architecture.py
=========================
Bidirectional LSTM model definition.

Architecture
------------
  Embedding(vocab_size, 128)
      ↓
  Bidirectional(LSTM(256, return_sequences=True))
      ↓
  Dropout(0.3)
      ↓
  LSTM(128)
      ↓
  Dropout(0.3)
      ↓
  Dense(vocab_size, activation='softmax')

Design decisions are documented inline in build_model().
"""

from __future__ import annotations

import logging

import tensorflow as tf
from tensorflow.keras.layers import (
    Bidirectional,
    Dense,
    Dropout,
    Embedding,
    LSTM,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

import config

logger = logging.getLogger(__name__)


def build_model(vocab_size: int, input_len: int) -> Sequential:
    """
    Build and compile the Bidirectional LSTM next-word prediction model.

    Architecture decisions
    ----------------------
    Embedding(vocab_size, 128)
        128 dimensions adequately separates 10K word types in vector space.
        64 (original) was too small; 256 adds parameters without much gain
        at this vocabulary size.

    Bidirectional(LSTM(256, return_sequences=True))
        Bidirectional passes the sequence in both directions during training,
        producing richer gradient signal. At inference we only give left
        context (future words are unknown), so predictions are still causal.
        return_sequences=True is required when stacking LSTM layers.

    Dropout(0.3)
        Increased from 0.2. Small corpora overfit quickly; 0.3 provides
        stronger regularisation without killing learning.

    LSTM(128)
        Second LSTM compresses the 512-dim bidirectional output into a
        128-dim context vector, forcing the model to generalise.

    Dropout(0.3)
        Same rationale as above.

    Dense(vocab_size, activation='softmax')
        Output: probability distribution over all vocab words.
        The predicted next word is argmax (or top-k) of this vector.

    Loss: categorical_crossentropy
        FIX: original used sparse_categorical_crossentropy, which expects
        INTEGER labels. Because prepare_dataset() returns one-hot y via
        to_categorical(), the correct loss is categorical_crossentropy.
        Using sparse_ with one-hot y produces a meaningless loss value —
        the model appears to train but learns nothing useful.

    Note: Embedding input_length omitted (deprecated in Keras 3.x).
          Shape is inferred from the data automatically.

    Parameters
    ----------
    vocab_size : int  Number of unique tokens (= output layer width).
    input_len  : int  Number of input time steps (= max_len - 1).

    Returns
    -------
    Sequential  Compiled Keras model ready for training.
    """
    model = Sequential(
        name="LSTM_NextWordPredictor",
        layers=[
            Embedding(
                input_dim  = vocab_size,
                output_dim = config.EMBEDDING_DIM,
                # input_length intentionally omitted — deprecated in Keras 3.x
                # Keras infers the shape from the first batch automatically
                name       = "embedding",
            ),
            Bidirectional(
                LSTM(config.LSTM_UNITS_1, return_sequences=True),
                name = "bilstm",
            ),
            Dropout(config.DROPOUT_RATE, name="dropout_1"),
            LSTM(config.LSTM_UNITS_2, name="lstm_2"),
            Dropout(config.DROPOUT_RATE, name="dropout_2"),
            Dense(vocab_size, activation="softmax", name="output"),
        ],
    )

    model.compile(
        optimizer = Adam(learning_rate=config.LEARNING_RATE),
        loss      = "categorical_crossentropy",   # FIX: not sparse_
        metrics   = ["accuracy"],
    )

    # Build the model explicitly so count_params() works before first fit()
    model.build(input_shape=(None, input_len))

    logger.info(
        "Model built | params=%d | vocab=%d | input_len=%d",
        model.count_params(), vocab_size, input_len,
    )
    return model
