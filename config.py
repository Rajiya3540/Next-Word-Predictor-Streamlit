"""
config.py
=========
Central configuration for the Next Word Predictor project.

All constants, paths, and hyperparameters live here.
No other module should hardcode any value that belongs in this file.
"""

from pathlib import Path

# ── Root paths ────────────────────────────────────────────────────────────────

ROOT_DIR    = Path(__file__).parent
DATA_DIR    = ROOT_DIR / "data"
MODELS_DIR  = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"

# ── Model persistence ─────────────────────────────────────────────────────────

MODEL_PATH     = MODELS_DIR / "lstm_best_model.keras"
TOKENIZER_PATH = MODELS_DIR / "tokenizer_data.npy"

# ── Corpus ────────────────────────────────────────────────────────────────────

CORPUS_PATH = DATA_DIR / "all_books.txt"

# ── Vocabulary ────────────────────────────────────────────────────────────────

MAX_VOCAB_SIZE = 10_000      # cap vocabulary to the most frequent N words
OOV_TOKEN      = "<OOV>"    # token assigned to unknown words at inference

# ── Sequence generation ───────────────────────────────────────────────────────

MAX_SEQ_LEN = 30             # longest prefix to generate (memory + speed tradeoff)
MIN_SEQ_LEN = 2              # shortest valid sequence (1 input token + 1 label)

# ── Model architecture ────────────────────────────────────────────────────────

EMBEDDING_DIM = 128          # word vector dimensions; 128 fits vocab ≤ 10K well
LSTM_UNITS_1  = 256          # first BiLSTM layer (×2 after bidirectional merge)
LSTM_UNITS_2  = 128          # second LSTM; compresses context to 128-dim vector
DROPOUT_RATE  = 0.3          # higher than default 0.2; reduces overfitting on small corpora

# ── Training ──────────────────────────────────────────────────────────────────

BATCH_SIZE    = 64           # smooth gradient updates; 128 overshoots on small corpora
MAX_EPOCHS    = 150          # EarlyStopping will terminate before this in practice
PATIENCE      = 12           # epochs to wait for val_loss improvement before stopping
LEARNING_RATE = 1e-3         # Adam default; ReduceLROnPlateau will halve this when needed
LR_FACTOR     = 0.5          # ReduceLROnPlateau reduction factor
LR_PATIENCE   = 5            # epochs before learning rate is reduced
MIN_LR        = 1e-6         # floor for learning rate scheduler

# ── Dataset splits ────────────────────────────────────────────────────────────

TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
TEST_RATIO  = 0.10           # must sum to 1.0

# ── Streamlit app behaviour ───────────────────────────────────────────────────

RETRAIN_EVERY = 20           # trigger retrain after this many new chat messages
MAX_SEQUENCES = 50_000       # hard cap on sequence count to prevent OOM in chat mode
TOP_K         = 5            # default number of next-word predictions to show

# ── Ensure runtime directories exist ─────────────────────────────────────────

for _dir in (DATA_DIR, MODELS_DIR, RESULTS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)
