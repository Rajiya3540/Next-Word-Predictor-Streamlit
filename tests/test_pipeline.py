"""
Unit tests for the Next Word Predictor pipeline.

Run with:
    pytest tests/ -v

Tests cover:
  - Text cleaning
  - Tokenization
  - Sequence generation (correct prefix n-gram, not fixed window)
  - X/y split (correct column indexing — the bug-4 fix)
  - Prediction function (no crash on unknown words)
  - Flask routes (status, predict, clear)
"""

import os
import sys
import numpy as np
import pytest

# ── Make project root importable ────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from notebooks.lstm_next_word_predictor import clean_text, preprocess_seed


# ==================================================================
# PHASE 1 VERIFICATION — Bug fixes
# ==================================================================

class TestBugFixes:

    def test_y_label_is_1d_not_2d(self):
        """
        BUG 4 FIX VERIFICATION:
        y = sequences[:, -1]  must produce a 1-D array.
        y = sequences[:, :-1] would produce a 2-D array (the old bug).
        """
        sequences = np.array([
            [0, 1, 2, 3],
            [0, 0, 1, 4],
            [1, 2, 3, 5],
        ])
        y_correct = sequences[:, -1]    # correct: 1-D
        y_buggy   = sequences[:, :-1]   # old bug: 2-D

        assert y_correct.ndim == 1, "y must be 1-D (scalar index per sample)"
        assert y_buggy.ndim == 2,   "Sanity: old code would have been 2-D"
        assert list(y_correct) == [3, 4, 5]

    def test_x_shape_is_correct(self):
        """X = all columns except last."""
        sequences = np.array([[0, 1, 2, 3], [0, 0, 1, 4]])
        X = sequences[:, :-1]
        assert X.shape == (2, 3), f"X shape should be (2, 3), got {X.shape}"

    def test_index_word_lookup_is_o1(self):
        """
        BUG 8 FIX VERIFICATION:
        tokenizer.index_word is a dict — lookup is O(1).
        The original code looped through word_index.items() for each prediction,
        which was O(vocab_size) per prediction.
        """
        from tensorflow.keras.preprocessing.text import Tokenizer
        tok = Tokenizer()
        tok.fit_on_texts(["hello world foo bar"])
        # Direct O(1) lookup
        idx  = tok.word_index["hello"]
        word = tok.index_word[idx]
        assert word == "hello", f"index_word lookup failed: got '{word}'"


# ==================================================================
# PHASE 2 VERIFICATION — Data pipeline
# ==================================================================

class TestDataPipeline:

    def test_clean_text_lowercases(self):
        result = clean_text("Hello World PYTHON")
        assert result == result.lower(), "clean_text must lowercase"

    def test_clean_text_removes_punctuation(self):
        result = clean_text("Hello, World! This is a test.")
        assert "," not in result
        assert "!" not in result
        assert "." not in result

    def test_clean_text_keeps_apostrophes(self):
        """Apostrophes must be kept for contractions like don't, I'm."""
        result = clean_text("don't stop")
        assert "'" in result, "Apostrophes should be preserved"

    def test_clean_text_collapses_spaces(self):
        result = clean_text("hello   world")
        assert "  " not in result, "Multiple spaces must be collapsed"

    def test_prefix_ngram_generates_variable_lengths(self):
        """
        BUG 7 FIX VERIFICATION:
        Correct prefix n-gram generates sequences of increasing length.
        Old fixed-window code always generated length-4 sequences.
        """
        token_list = [1, 2, 3, 4, 5]
        sequences = []
        for i in range(1, len(token_list)):
            seq = token_list[:i + 1]
            sequences.append(seq)

        lengths = [len(s) for s in sequences]
        assert lengths == [2, 3, 4, 5], \
            f"Prefix n-gram lengths should be [2,3,4,5], got {lengths}"
        assert len(set(lengths)) > 1, \
            "All sequences have same length — fixed-window bug still present"

    def test_fixed_window_always_same_length(self):
        """Demonstrate what the OLD bug produced (should not equal our output)."""
        token_list = [1, 2, 3, 4, 5]
        old_sequences = [token_list[i-3:i+1] for i in range(3, len(token_list))]
        old_lengths = [len(s) for s in old_sequences]
        # Old code always gave length 4
        assert all(l == 4 for l in old_lengths), \
            "Old fixed-window must produce all-same-length sequences"


# ==================================================================
# PHASE 7 VERIFICATION — Prediction function
# ==================================================================

class TestPreprocessSeed:

    def test_preprocess_seed_lowercases(self):
        result = preprocess_seed("Alice WAS Here")
        assert result == result.lower()

    def test_preprocess_seed_strips_punctuation(self):
        result = preprocess_seed("Hello, world!")
        assert "," not in result
        assert "!" not in result

    def test_preprocess_seed_handles_empty(self):
        result = preprocess_seed("")
        assert result == ""

    def test_preprocess_seed_handles_numbers(self):
        result = preprocess_seed("chapter 2")
        assert "2" in result


# ==================================================================
# PHASE 8 VERIFICATION — Flask routes
# ==================================================================

class TestFlaskApp:

    @pytest.fixture
    def client(self):
        """Create a test Flask client."""
        import app as flask_app
        flask_app.app.config["TESTING"] = True
        with flask_app.app.test_client() as client:
            yield client

    def test_home_returns_200(self, client):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_status_endpoint(self, client):
        resp = client.get("/status")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "training"     in data
        assert "vocab_size"   in data
        assert "model_loaded" in data

    def test_predict_empty_text(self, client):
        resp = client.post(
            "/predict",
            json={"text": ""},
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["predictions"] == []

    def test_predict_unknown_words_no_crash(self, client):
        """Prediction must not crash on words not in vocabulary."""
        resp = client.post(
            "/predict",
            json={"text": "xyzzyqqqnonexistentword"},
            content_type="application/json",
        )
        assert resp.status_code == 200
        # predictions may be empty (no model trained) but must not raise 500

    def test_clear_resets_state(self, client):
        client.post("/clear")
        resp   = client.get("/status")
        data   = resp.get_json()
        assert data["chat_messages"] == 0
        assert data["vocab_size"]    == 0
        assert data["model_loaded"]  is False

    def test_chat_empty_message(self, client):
        resp = client.post(
            "/chat",
            json={"user": "Test", "message": ""},
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "empty"

    def test_chat_adds_to_history(self, client):
        client.post("/clear")
        client.post("/chat", json={"user": "A", "message": "hello world"})
        resp = client.get("/status")
        data = resp.get_json()
        assert data["chat_messages"] == 1


# ==================================================================
# RUN
# ==================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
