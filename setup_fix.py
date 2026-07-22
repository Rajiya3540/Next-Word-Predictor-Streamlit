"""
setup_fix.py
============
Run this script ONCE from your project root to fix everything.

    python setup_fix.py

It will:
  1. Detect where all files currently are
  2. Move them to the correct locations
  3. Create all __init__.py files with correct content
  4. Verify every import works
  5. Tell you exactly what to do next
"""

import os
import sys
import shutil
from pathlib import Path

ROOT = Path(__file__).parent
print("=" * 60)
print("  Next Word Predictor — Auto Fix Script")
print(f"  Project root: {ROOT}")
print("=" * 60)

# ── Step 1: Create folder structure ──────────────────────────────
print("\n[1/5] Creating folder structure...")
folders = [
    ROOT / "src" / "data",
    ROOT / "src" / "model",
    ROOT / "src" / "utils",
    ROOT / "data",
    ROOT / "models",
    ROOT / "results",
    ROOT / "tests",
]
for folder in folders:
    folder.mkdir(parents=True, exist_ok=True)
    print(f"  ✅ {folder.relative_to(ROOT)}")

# ── Step 2: File mapping — where each file SHOULD be ─────────────
print("\n[2/5] Moving files to correct locations...")

# Maps: possible source names → destination path
FILE_MAP = {
    # Data module
    "preprocessor.py"     : ROOT / "src" / "data" / "preprocessor.py",
    # Model module
    "architecture.py"     : ROOT / "src" / "model" / "architecture.py",
    "trainer.py"          : ROOT / "src" / "model" / "trainer.py",
    "predictor.py"        : ROOT / "src" / "model" / "predictor.py",
    "persistence.py"      : ROOT / "src" / "model" / "persistence.py",
    # Utils module
    "helpers.py"          : ROOT / "src" / "utils" / "helpers.py",
    # Tests
    "test_pipeline.py"    : ROOT / "tests" / "test_pipeline.py",
    # Data
    "download_corpus.py"  : ROOT / "data" / "download_corpus.py",
}

for filename, dest in FILE_MAP.items():
    src_in_root = ROOT / filename
    if src_in_root.exists() and not dest.exists():
        shutil.move(str(src_in_root), str(dest))
        print(f"  ✅ Moved {filename} → {dest.relative_to(ROOT)}")
    elif dest.exists():
        print(f"  ✅ Already in place: {dest.relative_to(ROOT)}")
    else:
        print(f"  ⚠️  Not found: {filename} (may need to download)")

# Handle data_init.py → src/data/__init__.py
data_init_src = ROOT / "data_init.py"
data_init_dst = ROOT / "src" / "data" / "__init__.py"
if data_init_src.exists() and not data_init_dst.exists():
    shutil.move(str(data_init_src), str(data_init_dst))
    print(f"  ✅ Moved data_init.py → src/data/__init__.py")

# ── Step 3: Write all __init__.py files ──────────────────────────
print("\n[3/5] Writing __init__.py files...")

# src/__init__.py (empty)
(ROOT / "src" / "__init__.py").write_text("", encoding="utf-8")
print("  ✅ src/__init__.py")

# src/data/__init__.py
data_init = ROOT / "src" / "data" / "__init__.py"
if not data_init.exists() or data_init.stat().st_size < 50:
    data_init.write_text(
        'from src.data.preprocessor import (\n'
        '    build_dataset, build_tokenizer, clean_text,\n'
        '    generate_sequences, prepare_dataset,\n'
        ')\n'
        '__all__ = ["build_dataset","build_tokenizer","clean_text",'
        '"generate_sequences","prepare_dataset"]\n',
        encoding="utf-8"
    )
    print("  ✅ src/data/__init__.py (written)")
else:
    print("  ✅ src/data/__init__.py (exists)")

# src/model/__init__.py
model_init = ROOT / "src" / "model" / "__init__.py"
model_init_content = '''\
from src.model.architecture import build_model
from src.model.persistence import (
    load as load_model,
    model_exists,
    save as save_model,
)
from src.model.predictor import predict_next_words, preprocess_seed
from src.model.trainer import (
    evaluate,
    get_callbacks,
    run_training_pipeline,
    split_dataset,
    train,
)

__all__ = [
    "build_model",
    "evaluate",
    "get_callbacks",
    "load_model",
    "model_exists",
    "predict_next_words",
    "preprocess_seed",
    "run_training_pipeline",
    "save_model",
    "split_dataset",
    "train",
]
'''
model_init.write_text(model_init_content, encoding="utf-8")
print("  ✅ src/model/__init__.py (written fresh)")

# src/utils/__init__.py
utils_init = ROOT / "src" / "utils" / "__init__.py"
if not utils_init.exists() or utils_init.stat().st_size < 50:
    utils_init.write_text(
        'from src.utils.helpers import (\n'
        '    compute_perplexity, format_metric_table,\n'
        '    human_readable_size, plot_training_curves,\n'
        ')\n'
        '__all__ = ["compute_perplexity","format_metric_table",'
        '"human_readable_size","plot_training_curves"]\n',
        encoding="utf-8"
    )
    print("  ✅ src/utils/__init__.py (written)")
else:
    print("  ✅ src/utils/__init__.py (exists)")

# tests/__init__.py
(ROOT / "tests" / "__init__.py").write_text("", encoding="utf-8")
print("  ✅ tests/__init__.py")

# ── Step 4: Delete junk files ─────────────────────────────────────
print("\n[4/5] Cleaning up unwanted files...")
junk = [
    ROOT / "index.html",
    ROOT / "lstm_next_word_predictor.py",
    ROOT / "config.cpython-312.pyc",
    ROOT / "data_init.py",
    ROOT / "model_init_p4.py",
    ROOT / "model_init_p5p6.py",
    ROOT / "utils_init.py",
]
for f in junk:
    if f.exists():
        f.unlink()
        print(f"  🗑  Deleted {f.name}")

# Root __init__.py (0 bytes) — delete it
root_init = ROOT / "__init__.py"
if root_init.exists() and root_init.stat().st_size == 0:
    root_init.unlink()
    print("  🗑  Deleted root __init__.py (was 0 bytes)")

# Rename gitignore.txt → .gitignore
gitignore_txt = ROOT / "gitignore.txt"
gitignore     = ROOT / ".gitignore"
if gitignore_txt.exists() and not gitignore.exists():
    gitignore_txt.rename(gitignore)
    print("  ✅ Renamed gitignore.txt → .gitignore")

# Copy model files from parent folder if available
parent = ROOT.parent
for old_name, new_name in [
    ("lstm_next_word_model.keras", "lstm_best_model.keras"),
    ("tokenizer_data.npy",         "tokenizer_data.npy"),
]:
    src_path  = parent / old_name
    dest_path = ROOT / "models" / new_name
    if src_path.exists() and not dest_path.exists():
        shutil.copy2(str(src_path), str(dest_path))
        print(f"  ✅ Copied {old_name} → models/{new_name}")

# ── Step 5: Verify imports ────────────────────────────────────────
print("\n[5/5] Verifying imports...")
sys.path.insert(0, str(ROOT))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings
warnings.filterwarnings("ignore")

errors = []

def try_import(label, code):
    try:
        exec(code, {"__builtins__": __builtins__})
        print(f"  ✅ {label}")
    except Exception as e:
        print(f"  ❌ {label}")
        print(f"       Error: {e}")
        errors.append((label, str(e)))

try_import("config",    "import config")
try_import("src.data",  "from src.data import build_dataset")
try_import("src.model", "from src.model import model_exists, load_model, predict_next_words")
try_import("src.utils", "from src.utils import compute_perplexity, format_metric_table")

# ── Final report ──────────────────────────────────────────────────
print("\n" + "=" * 60)

# Check streamlit_app.py size
app_path = ROOT / "streamlit_app.py"
if app_path.exists():
    size = app_path.stat().st_size
    if size < 8000:
        print(f"  ⚠️  streamlit_app.py is {size} bytes — this is the OLD Phase 1 stub!")
        print("      Download 'streamlit_app_CORRECT.py' from Claude outputs")
        print("      Rename it to 'streamlit_app.py' and replace the old one.")
    else:
        print(f"  ✅ streamlit_app.py is {size} bytes — correct Phase 7 version")
else:
    print("  ❌ streamlit_app.py not found!")

print()
if not errors:
    print("  ✅ ALL IMPORTS WORKING")
    print()
    if app_path.exists() and app_path.stat().st_size > 8000:
        print("  🚀 Ready to run:")
        print("     streamlit run streamlit_app.py")
    else:
        print("  ⚠️  Replace streamlit_app.py first, then run:")
        print("     streamlit run streamlit_app.py")
else:
    print(f"  ❌ {len(errors)} import(s) failed:")
    for label, err in errors:
        print(f"     {label}: {err}")
    print()
    print("  Check that these files exist inside src/ subfolders:")
    print("    src/data/preprocessor.py")
    print("    src/model/architecture.py")
    print("    src/model/trainer.py")
    print("    src/model/predictor.py")
    print("    src/model/persistence.py")
    print("    src/utils/helpers.py")

print("=" * 60)
