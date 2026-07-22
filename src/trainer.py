"""
src/model/trainer.py
====================
Training pipeline: split → callbacks → fit → evaluate.

Public API
----------
split_dataset          : 80/10/10 train/val/test split
get_callbacks          : EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
train                  : build model, run fit(), return (model, history)
evaluate               : compute all metrics on three splits
run_training_pipeline  : full pipeline — corpus → trained model + metrics
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer

import config
from src.model.architecture import build_model

logger = logging.getLogger(__name__)


# ── 1. Dataset Split ──────────────────────────────────────────────────────────

def split_dataset(
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray]:
    """
    Split (X, y) into 80% train / 10% val / 10% test.

    Why a proper split matters
    --------------------------
    Without validation data, EarlyStopping cannot detect overfitting.
    Without a held-out test set, reported accuracy is always on training
    data — an optimistic, unreliable figure.

    Strategy: two-stage split
        Stage 1: 80% train, 20% temp
        Stage 2: temp → 50/50 → 10% val, 10% test

    Parameters
    ----------
    X : np.ndarray  Input sequences.
    y : np.ndarray  One-hot label matrix.

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test
    """
    val_test = config.VAL_RATIO + config.TEST_RATIO   # 0.20

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y,
        test_size    = val_test,
        random_state = 42,
        shuffle      = True,
    )

    # Split temp 50/50 → val and test
    test_of_tmp = config.TEST_RATIO / val_test        # 0.50

    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp,
        test_size    = test_of_tmp,
        random_state = 42,
        shuffle      = True,
    )

    logger.info(
        "Split | train=%d | val=%d | test=%d",
        len(X_train), len(X_val), len(X_test),
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# ── 2. Callbacks ──────────────────────────────────────────────────────────────

def get_callbacks(model_path: Path = config.MODEL_PATH) -> list:
    """
    Return the three standard training callbacks.

    EarlyStopping(patience=12, restore_best_weights=True)
        Stops training when val_loss does not improve for 12 epochs.
        Restores the weights from the best epoch automatically.
        Replaces the original approach of always running 100–400 epochs,
        which wasted compute and caused overfitting.

    ModelCheckpoint(save_best_only=True)
        Saves the model to disk ONLY when val_loss improves.
        config.MODEL_PATH always contains the best model, never the last.

    ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6)
        Halves the learning rate when val_loss plateaus for 5 epochs.
        Adam's default lr=0.001 may overshoot fine minima late in training;
        the scheduler lets the optimiser fine-tune more carefully.

    Parameters
    ----------
    model_path : Path  Where to checkpoint the best model.

    Returns
    -------
    list  Three Keras callback objects.
    """
    return [
        EarlyStopping(
            monitor              = "val_loss",
            patience             = config.PATIENCE,
            restore_best_weights = True,
            verbose              = 1,
        ),
        ModelCheckpoint(
            filepath       = str(model_path),
            monitor        = "val_loss",
            save_best_only = True,
            verbose        = 1,
        ),
        ReduceLROnPlateau(
            monitor  = "val_loss",
            factor   = config.LR_FACTOR,
            patience = config.LR_PATIENCE,
            min_lr   = config.MIN_LR,
            verbose  = 1,
        ),
    ]


# ── 3. Train ──────────────────────────────────────────────────────────────────

def train(
    X_train : np.ndarray,
    y_train : np.ndarray,
    X_val   : np.ndarray,
    y_val   : np.ndarray,
    vocab_size: int,
    input_len : int,
) -> tuple[Sequential, object]:
    """
    Build, train, and return the model and Keras History object.

    validation_data=(X_val, y_val) is passed to model.fit() so that:
      - EarlyStopping monitors real generalisation (not training loss)
      - ReduceLROnPlateau reacts to val_loss plateau
      - History records both train and val curves for plotting

    Parameters
    ----------
    X_train, y_train : np.ndarray  Training split (80%).
    X_val,   y_val   : np.ndarray  Validation split (10%).
    vocab_size        : int         Vocabulary size (output layer width).
    input_len         : int         Padded sequence length minus 1.

    Returns
    -------
    model   : Sequential     Best model (restored by EarlyStopping).
    history : keras.History  Training history for plotting curves.
    """
    model = build_model(vocab_size=vocab_size, input_len=input_len)

    logger.info(
        "Training start | epochs=%d | batch=%d | train=%d | val=%d",
        config.MAX_EPOCHS, config.BATCH_SIZE, len(X_train), len(X_val),
    )

    history = model.fit(
        X_train, y_train,
        validation_data = (X_val, y_val),
        epochs          = config.MAX_EPOCHS,
        batch_size      = config.BATCH_SIZE,
        callbacks       = get_callbacks(),
        verbose         = 1,
    )

    epochs_run = len(history.history["loss"])
    best_val   = min(history.history["val_loss"])
    logger.info(
        "Training done | epochs_run=%d | best_val_loss=%.4f",
        epochs_run, best_val,
    )
    return model, history


# ── 4. Evaluate ───────────────────────────────────────────────────────────────

def _top_k_accuracy(
    model : Sequential,
    X     : np.ndarray,
    y     : np.ndarray,
    k     : int = 3,
) -> float:
    """Return fraction of samples where true label is in model's top-k."""
    y_pred  = model.predict(X, batch_size=256, verbose=0)
    y_true  = np.argmax(y, axis=1)
    correct = sum(
        int(y_true[i] in np.argsort(y_pred[i])[-k:])
        for i in range(len(y_true))
    )
    return correct / len(y_true)


def evaluate(
    model  : Sequential,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val  : np.ndarray,
    y_val  : np.ndarray,
    X_test : np.ndarray,
    y_test : np.ndarray,
) -> dict[str, float]:
    """
    Evaluate model on all three splits and return a metrics dict.

    Metrics
    -------
    train_acc  / val_acc  / test_acc   — top-1 accuracy on each split
    train_loss / val_loss / test_loss  — cross-entropy on each split
    perplexity                          — e^(test_loss); lower is better
    top3_acc                            — fraction correct in top-3 predictions

    Perplexity interpretation
    -------------------------
    A random model guessing uniformly over vocab_size words has
    perplexity = vocab_size (e.g. 5000 for a 5K-word vocabulary).
    A well-trained model typically reaches perplexity < 100 on its domain.

    Parameters
    ----------
    model              : Sequential  Trained Keras model.
    X_train … y_test   : np.ndarray  Three dataset splits.

    Returns
    -------
    dict[str, float]  All metrics, ready to display in Streamlit.
    """
    logger.info("Evaluating on all splits …")

    def _eval(X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        loss, acc = model.evaluate(X, y, verbose=0, batch_size=256)
        return float(loss), float(acc)

    tr_loss, tr_acc = _eval(X_train, y_train)
    vl_loss, vl_acc = _eval(X_val,   y_val)
    te_loss, te_acc = _eval(X_test,  y_test)
    top3            = _top_k_accuracy(model, X_test, y_test, k=3)
    perplexity      = float(np.exp(te_loss))

    metrics = {
        "train_acc"  : tr_acc,
        "val_acc"    : vl_acc,
        "test_acc"   : te_acc,
        "train_loss" : tr_loss,
        "val_loss"   : vl_loss,
        "test_loss"  : te_loss,
        "top3_acc"   : top3,
        "perplexity" : perplexity,
    }

    logger.info(
        "Metrics | train=%.4f | val=%.4f | test=%.4f | top3=%.4f | ppl=%.2f",
        tr_acc, vl_acc, te_acc, top3, perplexity,
    )
    return metrics


# ── 5. Full Pipeline ──────────────────────────────────────────────────────────

def run_training_pipeline(
    corpus_path: Path | None = None,
) -> dict:
    """
    Full training pipeline: corpus → trained model + metrics.

    Calls in order:
      1. build_dataset()    → X, y, tokenizer, vocab_size, max_len
      2. split_dataset()    → train / val / test splits
      3. train()            → model, history
      4. evaluate()         → metrics dict

    The ModelCheckpoint callback inside train() saves the best model to
    config.MODEL_PATH during training. The tokenizer is returned here
    and saved by persistence.save() (Phase 5) after this function returns.

    Parameters
    ----------
    corpus_path : Path | None  Override default corpus (config.CORPUS_PATH).

    Returns
    -------
    dict with keys:
        model, tokenizer, vocab_size, max_len,
        history, metrics,
        X_train, y_train, X_val, y_val, X_test, y_test
    """
    from src.data import build_dataset   # import here to avoid circular imports

    path = corpus_path or config.CORPUS_PATH

    # Step 1 — build dataset
    X, y, tokenizer, vocab_size, max_len = build_dataset(path)

    # Step 2 — split
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)

    # Step 3 — train
    model, history = train(
        X_train, y_train,
        X_val,   y_val,
        vocab_size = vocab_size,
        input_len  = X_train.shape[1],
    )

    # Step 4 — evaluate
    metrics = evaluate(model, X_train, y_train, X_val, y_val, X_test, y_test)

    # Step 5 — persist (model already checkpointed; this saves the tokenizer too)
    from src.model.persistence import save as _save
    _save(model, tokenizer, max_len, vocab_size)

    return {
        "model"      : model,
        "tokenizer"  : tokenizer,
        "vocab_size" : vocab_size,
        "max_len"    : max_len,
        "history"    : history,
        "metrics"    : metrics,
        "X_train"    : X_train,
        "y_train"    : y_train,
        "X_val"      : X_val,
        "y_val"      : y_val,
        "X_test"     : X_test,
        "y_test"     : y_test,
    }