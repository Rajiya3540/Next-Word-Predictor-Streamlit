"""
streamlit_app.py
================
Next Word Predictor — Professional Streamlit Application

Entry point:
    streamlit run streamlit_app.py

Architecture
------------
Session state   → model, tokenizer, chat history, metrics (isolated per session)
_init_state()   → sets all defaults once per session
_try_load()     → auto-loads saved model on startup
_sidebar()      → status, corpus training, chat controls
_tab_predict()  → live next-word prediction with confidence bars
_tab_chat()     → chat-based corpus building + auto-retrain
_tab_info()     → architecture, evaluation metrics, training curves
main()          → wires everything together
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Ensure project root is importable regardless of working directory
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
import config

logging.basicConfig(level=logging.WARNING)

# ─────────────────────────────────────────────────────────────────────────────
# Page config  (must be the very first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Next Word Predictor — LSTM",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "Next Word Predictor · Bidirectional LSTM · TensorFlow & Streamlit"},
)

# ─────────────────────────────────────────────────────────────────────────────
# Session State
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULTS: dict = {
    "model"                  : None,
    "tokenizer"              : None,
    "max_len"                : 0,
    "vocab_size"             : 0,
    "model_loaded"           : False,
    "chat_history"           : [],      # list[{"role": str, "content": str}]
    "messages_at_last_train" : 0,
    "metrics"                : None,    # dict from evaluate()
    "history"                : None,    # Keras History object
    "seed"                   : "",      # controlled live-prediction input
    "error"                  : None,    # last error message to display
}


def _init_state() -> None:
    """Set session state defaults once per session."""
    for key, value in _DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ─────────────────────────────────────────────────────────────────────────────
# Model Loading
# ─────────────────────────────────────────────────────────────────────────────

def _try_load() -> None:
    """Auto-load saved model on startup if available and not yet loaded."""
    if st.session_state.model_loaded:
        return

    from src.model import model_exists, load_model

    if not model_exists():
        return

    try:
        model, tok, max_len, vocab_size = load_model()
        st.session_state.update({
            "model"        : model,
            "tokenizer"    : tok,
            "max_len"      : max_len,
            "vocab_size"   : vocab_size,
            "model_loaded" : True,
            "error"        : None,
        })
    except Exception as exc:
        st.session_state.error = f"Auto-load failed: {exc}"


# ─────────────────────────────────────────────────────────────────────────────
# Training helpers
# ─────────────────────────────────────────────────────────────────────────────

def _apply_result(result: dict) -> None:
    """Write training pipeline output into session state."""
    st.session_state.update({
        "model"        : result["model"],
        "tokenizer"    : result["tokenizer"],
        "max_len"      : result["max_len"],
        "vocab_size"   : result["vocab_size"],
        "model_loaded" : True,
        "metrics"      : result["metrics"],
        "history"      : result["history"],
        "messages_at_last_train": len(st.session_state.chat_history),
        "error"        : None,
    })


def _train_corpus() -> None:
    """Train on the configured corpus file (full pipeline)."""
    from src.model import run_training_pipeline
    try:
        _apply_result(run_training_pipeline(config.CORPUS_PATH))
    except Exception as exc:
        st.session_state.error = str(exc)


def _train_chat() -> None:
    """Train on current chat history corpus."""
    import os, tempfile
    from src.data import build_dataset
    from src.model import split_dataset, train, evaluate, save_model

    messages = [m["content"] for m in st.session_state.chat_history
                if m["role"] == "user"]

    if len(messages) < 3:
        st.session_state.error = "Need at least 3 chat messages to train."
        return

    corpus = " ".join(messages)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt",
                                     delete=False, encoding="utf-8") as f:
        f.write(corpus)
        tmp = f.name

    try:
        X, y, tok, vocab_size, max_len = build_dataset(Path(tmp))
        X_tr, X_v, X_te, y_tr, y_v, y_te = split_dataset(X, y)
        model, history = train(X_tr, y_tr, X_v, y_v,
                               vocab_size=vocab_size, input_len=X_tr.shape[1])
        metrics = evaluate(model, X_tr, y_tr, X_v, y_v, X_te, y_te)
        save_model(model, tok, max_len, vocab_size)
        _apply_result({
            "model": model, "tokenizer": tok, "max_len": max_len,
            "vocab_size": vocab_size, "metrics": metrics, "history": history,
        })
    except Exception as exc:
        st.session_state.error = str(exc)
    finally:
        os.unlink(tmp)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

def _sidebar() -> None:
    with st.sidebar:
        st.markdown("## 🧠 Next Word Predictor")
        st.caption("Bidirectional LSTM · TensorFlow · Streamlit")
        st.divider()

        # ── Model status ──────────────────────────────────────────────────────
        st.markdown("### Model Status")
        if st.session_state.model_loaded:
            st.success("✅ Model ready")
            c1, c2 = st.columns(2)
            c1.metric("Vocabulary",  f"{st.session_state.vocab_size:,}")
            c2.metric("Max length",  str(st.session_state.max_len))

            if config.MODEL_PATH.exists():
                from src.utils import human_readable_size
                st.caption(f"📦 {human_readable_size(config.MODEL_PATH.stat().st_size)}")
        else:
            st.warning("⚠️ No model loaded")
            st.caption(
                "Copy your trained `.keras` + `.npy` files into `models/`, "
                "or train a fresh model below."
            )

        st.divider()

        # ── Corpus training ───────────────────────────────────────────────────
        st.markdown("### Train from Corpus")
        corpus_ok = config.CORPUS_PATH.exists()

        if corpus_ok:
            kb = config.CORPUS_PATH.stat().st_size / 1024
            st.caption(f"📄 `{config.CORPUS_PATH.name}` — {kb:.0f} KB")
        else:
            st.caption(f"📄 No corpus at `{config.CORPUS_PATH.name}`")
            st.code("python data/download_corpus.py", language="bash")

        if st.button("🚀 Train on Corpus", disabled=not corpus_ok,
                     use_container_width=True,
                     type="primary" if corpus_ok else "secondary"):
            with st.spinner("Training… (may take several minutes)"):
                _train_corpus()
            st.rerun()

        st.divider()

        # ── Chat training controls ────────────────────────────────────────────
        n_msgs      = len(st.session_state.chat_history)
        since_train = n_msgs - st.session_state.messages_at_last_train

        if n_msgs > 0:
            st.markdown("### Chat Training")
            progress = min(since_train / max(config.RETRAIN_EVERY, 1), 1.0)
            st.progress(progress,
                        text=f"{since_train} / {config.RETRAIN_EVERY} new messages")

        if st.button("⚡ Retrain on Chat", disabled=n_msgs < 3,
                     use_container_width=True):
            with st.spinner("Retraining on chat corpus…"):
                _train_chat()
            st.rerun()

        if st.button("🗑 Clear Chat", use_container_width=True,
                     disabled=n_msgs == 0):
            st.session_state.chat_history           = []
            st.session_state.messages_at_last_train = 0
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Tab 1 — Live Prediction
# ─────────────────────────────────────────────────────────────────────────────

def _tab_predict() -> None:
    st.subheader("🔮 Live Next-Word Prediction")

    if not st.session_state.model_loaded:
        st.info(
            "**No model loaded yet.** Choose one of these options:\n\n"
            "1. Copy `lstm_best_model.keras` + `tokenizer_data.npy` into the `models/` folder.\n"
            "2. Use the sidebar → **Train on Corpus** (requires `data/alice.txt`).\n"
            "3. Use the **Chat & Train** tab to build a small corpus from conversation."
        )
        return

    # ── Controlled text input ─────────────────────────────────────────────────
    seed = st.text_input(
        "Seed text:",
        value    = st.session_state.seed,
        placeholder = "e.g.  alice was beginning to",
        help     = "Type one or more words — predictions update automatically.",
    )
    st.session_state.seed = seed   # keep state in sync

    if not seed.strip():
        st.caption("Start typing above to see predictions.")
        return

    # ── Predict ───────────────────────────────────────────────────────────────
    from src.model import predict_next_words

    predictions = predict_next_words(
        seed_text = seed,
        model     = st.session_state.model,
        tokenizer = st.session_state.tokenizer,
        max_len   = st.session_state.max_len,
        top_k     = config.TOP_K,
    )

    if not predictions:
        st.warning(
            "No predictions for this seed. "
            "Try different words or retrain the model on a larger corpus."
        )
        return

    # ── Confidence bars ───────────────────────────────────────────────────────
    st.markdown("**Top predictions:**")
    for pred in predictions:
        c1, c2, c3 = st.columns([1.5, 6, 1])
        c1.markdown(f"**`{pred['word']}`**")
        c2.progress(float(pred["confidence"]))
        c3.caption(f"{pred['confidence']:.1%}")

    # ── Click-to-append buttons ───────────────────────────────────────────────
    st.markdown("**Click to append:**")
    btn_cols = st.columns(len(predictions))
    for i, (col, pred) in enumerate(zip(btn_cols, predictions)):
        if col.button(
            f"{pred['word']}  {pred['confidence']:.0%}",
            key=f"append_{i}_{pred['word']}",
        ):
            st.session_state.seed = seed.strip() + " " + pred["word"]
            st.rerun()

    # ── Multi-word sentence generation ────────────────────────────────────────
    st.divider()
    st.markdown("**Auto-generate a continuation:**")

    gc1, gc2, gc3 = st.columns([4, 1, 1])
    n_words = gc2.number_input("Words", min_value=1, max_value=20, value=5, step=1)

    if gc3.button("Generate ▶", use_container_width=True):
        generated = seed.strip()
        current   = generated
        for _ in range(int(n_words)):
            preds = predict_next_words(
                current,
                st.session_state.model,
                st.session_state.tokenizer,
                st.session_state.max_len,
                top_k=1,
            )
            if not preds:
                break
            current   = current.strip() + " " + preds[0]["word"]
            generated = current

        gc1.text_area("Generated:", value=generated, height=75, key="gen_out")


# ─────────────────────────────────────────────────────────────────────────────
# Tab 2 — Chat & Train
# ─────────────────────────────────────────────────────────────────────────────

def _tab_chat() -> None:
    st.subheader("💬 Chat & Train")
    st.caption(
        f"Type messages to build a training corpus. "
        f"The model retrains automatically every **{config.RETRAIN_EVERY} messages**, "
        f"or use **⚡ Retrain on Chat** in the sidebar anytime."
    )

    # ── Message history ───────────────────────────────────────────────────────
    with st.container(height=380):
        if not st.session_state.chat_history:
            st.caption("No messages yet — start chatting below.")
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

    # ── Chat input ────────────────────────────────────────────────────────────
    if user_input := st.chat_input("Type to add to the training corpus…"):
        st.session_state.chat_history.append(
            {"role": "user", "content": user_input}
        )

        # Live prediction response (if model is loaded)
        if st.session_state.model_loaded:
            from src.model import predict_next_words
            preds = predict_next_words(
                user_input,
                st.session_state.model,
                st.session_state.tokenizer,
                st.session_state.max_len,
                top_k=3,
            )
            if preds:
                reply = "Next words: " + " · ".join(
                    f"**{p['word']}** ({p['confidence']:.0%})" for p in preds
                )
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": reply}
                )

        # Auto-retrain at threshold
        since = (len(st.session_state.chat_history)
                 - st.session_state.messages_at_last_train)
        if since >= config.RETRAIN_EVERY:
            with st.spinner(f"Auto-retraining after {config.RETRAIN_EVERY} messages…"):
                _train_chat()

        st.rerun()

    # ── Corpus stats ──────────────────────────────────────────────────────────
    user_msgs = [m["content"] for m in st.session_state.chat_history
                 if m["role"] == "user"]
    if user_msgs:
        words = sum(len(m.split()) for m in user_msgs)
        st.caption(f"Corpus: **{len(user_msgs)} messages** · **{words:,} words**")


# ─────────────────────────────────────────────────────────────────────────────
# Tab 3 — Model Info
# ─────────────────────────────────────────────────────────────────────────────

def _tab_info() -> None:
    st.subheader("📊 Model Information")

    if not st.session_state.model_loaded:
        st.info("No model loaded yet.")
        return

    # ── Architecture ──────────────────────────────────────────────────────────
    with st.expander("🏗 Architecture", expanded=True):
        col_arch, col_hyp = st.columns(2)

        col_arch.markdown("**Layers**")
        col_arch.markdown(f"""
| Layer | Config |
|---|---|
| Embedding | `{st.session_state.vocab_size:,} × {config.EMBEDDING_DIM}` |
| Bidirectional LSTM | `{config.LSTM_UNITS_1} units × 2` |
| Dropout | `{config.DROPOUT_RATE}` |
| LSTM | `{config.LSTM_UNITS_2} units` |
| Dropout | `{config.DROPOUT_RATE}` |
| Dense (softmax) | `{st.session_state.vocab_size:,} units` |
""")
        col_hyp.markdown("**Hyperparameters**")
        col_hyp.markdown(f"""
| Parameter | Value |
|---|---|
| Vocabulary size | `{st.session_state.vocab_size:,}` |
| Sequence length | `{st.session_state.max_len}` |
| Batch size | `{config.BATCH_SIZE}` |
| Learning rate | `{config.LEARNING_RATE}` |
| ES patience | `{config.PATIENCE}` |
| Max epochs | `{config.MAX_EPOCHS}` |
""")

    # ── Evaluation metrics ────────────────────────────────────────────────────
    if st.session_state.metrics:
        from src.utils import format_metric_table
        fmt = format_metric_table(st.session_state.metrics)

        st.markdown("### 📈 Evaluation Results")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Train Acc",   fmt.get("train_acc",  "—"))
        c2.metric("Val Acc",     fmt.get("val_acc",    "—"))
        c3.metric("Test Acc",    fmt.get("test_acc",   "—"))
        c4.metric("Top-3 Acc",   fmt.get("top3_acc",  "—"))

        c5, c6, c7, _ = st.columns(4)
        c5.metric("Train Loss",  fmt.get("train_loss", "—"))
        c6.metric("Val Loss",    fmt.get("val_loss",   "—"))
        c7.metric("Perplexity",  fmt.get("perplexity", "—"))
    else:
        st.info("Metrics will appear here after training.")

    # ── Training curves ───────────────────────────────────────────────────────
    curves_path  = config.RESULTS_DIR / "training_curves.png"
    summary_path = config.RESULTS_DIR / "accuracy_summary.png"

    if st.session_state.history is not None:
        st.markdown("### 📉 Training Curves")
        if not curves_path.exists():
            from src.utils import plot_training_curves
            plot_training_curves(st.session_state.history, config.RESULTS_DIR)
        if curves_path.exists():
            st.image(str(curves_path), use_container_width=True)
        if summary_path.exists():
            st.image(str(summary_path), use_container_width=True)

    elif curves_path.exists():
        st.markdown("### 📉 Training Curves")
        st.image(str(curves_path), use_container_width=True)
        if summary_path.exists():
            st.image(str(summary_path), use_container_width=True)
    else:
        st.info(
            "Training curves appear here after training. "
            "If you trained in Jupyter, copy `results/training_curves.png` here."
        )

    # ── Files on disk ─────────────────────────────────────────────────────────
    with st.expander("💾 Files on Disk"):
        from src.utils import human_readable_size
        for path in [config.MODEL_PATH, config.TOKENIZER_PATH]:
            if path.exists():
                st.success(f"✅ `{path.name}` — {human_readable_size(path.stat().st_size)}")
            else:
                st.error(f"❌ `{path.name}` — not found")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    _init_state()
    _try_load()
    _sidebar()

    st.title("🧠 Next Word Predictor")
    st.caption(
        "Bidirectional LSTM · Predicts the most likely next word for any seed phrase · "
        f"[GitHub](https://github.com)"
    )

    # Show any persistent error
    if st.session_state.error:
        st.error(f"⚠️ {st.session_state.error}")
        if st.button("Dismiss"):
            st.session_state.error = None
            st.rerun()

    tab1, tab2, tab3 = st.tabs([
        "🔮 Live Prediction",
        "💬 Chat & Train",
        "📊 Model Info",
    ])
    with tab1: _tab_predict()
    with tab2: _tab_chat()
    with tab3: _tab_info()


if __name__ == "__main__":
    main()
