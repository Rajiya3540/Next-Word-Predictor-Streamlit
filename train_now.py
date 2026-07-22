"""
train_now.py - Lightweight training script for laptops
Run: python train_now.py
"""
import os, sys, warnings, re, string
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

from pathlib import Path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

print("=" * 50)
print("  Next Word Predictor - Training")
print("=" * 50)

CORPUS = ROOT / "data" / "all_books.txt"
MODEL_OUT  = ROOT / "models" / "lstm_best_model.keras"
TOK_OUT    = ROOT / "models" / "tokenizer_data.npy"
VOCAB_SIZE = 3000
MAX_SEQ    = 20
EMBED_DIM  = 64
LSTM_U1    = 128
LSTM_U2    = 64
DROPOUT    = 0.2
BATCH      = 32
EPOCHS     = 50
PATIENCE   = 8
OOV        = "<OOV>"

(ROOT / "models").mkdir(exist_ok=True)

if not CORPUS.exists():
    print("ERROR: data/alice.txt not found!")
    print("Run: python data/download_corpus.py")
    sys.exit(1)

print(f"\n[1/6] Corpus: {CORPUS.name} ({CORPUS.stat().st_size//1024} KB)")

# Clean text
print("[2/6] Cleaning text...")
raw  = CORPUS.read_text(encoding="utf-8", errors="ignore")
text = raw.lower()
s = re.search(r"\*\*\* start of (the|this) project gutenberg", text)
e = re.search(r"\*\*\* end of (the|this) project gutenberg", text)
if s: text = text[s.end():]
if e: text = text[:e.start()]
allowed = set(string.ascii_lowercase + string.digits + " \n'")
text = "".join(ch if ch in allowed else " " for ch in text)
text = re.sub(r"\s+", " ", text).strip()
print(f"         {len(text):,} characters")

# Tokenize
print("[3/6] Tokenizing...")
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences, to_categorical

tok = Tokenizer(num_words=VOCAB_SIZE, oov_token=OOV, filters="", lower=False)
tok.fit_on_texts([text])
vocab  = min(len(tok.word_index) + 1, VOCAB_SIZE + 1)
tokens = tok.texts_to_sequences([text])[0]
print(f"         vocab={vocab}, tokens={len(tokens):,}")

# Sequences
print("[4/6] Generating sequences...")
seqs = []
for i in range(1, len(tokens)):
    seq = tokens[max(0, i - MAX_SEQ + 1): i + 1]
    if len(seq) >= 2:
        seqs.append(seq)
    if len(seqs) >= 30000:
        break

max_len = max(len(s) for s in seqs)
padded  = pad_sequences(seqs, maxlen=max_len, padding="pre")
X       = padded[:, :-1]
y       = to_categorical(padded[:, -1], num_classes=vocab)
print(f"         {len(seqs):,} sequences, shape={X.shape}")

# Split
print("[5/6] Splitting data...")
from sklearn.model_selection import train_test_split
X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.2, random_state=42)
X_v,  X_te,  y_v,  y_te  = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42)
print(f"         train={len(X_tr)}, val={len(X_v)}, test={len(X_te)}")

# Build model
print("[6/6] Building model...")
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
tf.get_logger().setLevel("ERROR")

model = Sequential([
    Embedding(vocab, EMBED_DIM),
    Bidirectional(LSTM(LSTM_U1, return_sequences=True)),
    Dropout(DROPOUT),
    LSTM(LSTM_U2),
    Dropout(DROPOUT),
    Dense(vocab, activation="softmax"),
])

# Build before compile to avoid count_params error
model.build(input_shape=(None, X_tr.shape[1]))
model.compile(
    optimizer = "adam",
    loss      = "categorical_crossentropy",
    metrics   = ["accuracy"],
)
print(f"         Parameters: {model.count_params():,}")

print()
print("Training started! Watch progress below...")
print("(EarlyStopping will stop when val_loss stops improving)")
print("-" * 50)

callbacks = [
    EarlyStopping(
        monitor="val_loss", patience=PATIENCE,
        restore_best_weights=True, verbose=1,
    ),
    ModelCheckpoint(
        str(MODEL_OUT), monitor="val_loss",
        save_best_only=True, verbose=0,
    ),
    ReduceLROnPlateau(
        monitor="val_loss", factor=0.5,
        patience=4, min_lr=1e-6, verbose=0,
    ),
]

model.fit(
    X_tr, y_tr,
    validation_data = (X_v, y_v),
    epochs          = EPOCHS,
    batch_size      = BATCH,
    callbacks       = callbacks,
    verbose         = 1,
)

# Evaluate
_, test_acc = model.evaluate(X_te, y_te, verbose=0)
val_acc     = max(model.history.history["val_accuracy"])
epochs_run  = len(model.history.history["loss"])

# Save tokenizer
np.save(str(TOK_OUT), {
    "schema_version" : 1,
    "word_index"     : dict(tok.word_index),
    "index_word"     : dict(tok.index_word),
    "max_len"        : int(max_len),
    "vocab_size"     : int(vocab),
    "oov_token"      : OOV,
}, allow_pickle=True)

print()
print("=" * 50)
print("  TRAINING COMPLETE!")
print(f"  Epochs run    : {epochs_run}")
print(f"  Val  accuracy : {val_acc*100:.1f}%")
print(f"  Test accuracy : {test_acc*100:.1f}%")
print(f"  Model saved   : models/lstm_best_model.keras")
print(f"  Tokenizer     : models/tokenizer_data.npy")
print()
print("  Now run:")
print("  streamlit run streamlit_app.py")
print("=" * 50)
