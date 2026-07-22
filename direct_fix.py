"""
direct_fix.py
=============
Run this from your project root. It directly writes all correct
files to the right locations - no manual copying needed.

    python direct_fix.py
"""
import sys, os, shutil
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

print("Writing all correct files directly...\n")

# ── 1. src/model/__init__.py ─────────────────────────────────────────────────
(ROOT / "src" / "model" / "__init__.py").write_text('''\
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
''', encoding="utf-8")
print("  Written: src/model/__init__.py")

# ── 2. src/data/__init__.py ──────────────────────────────────────────────────
(ROOT / "src" / "data" / "__init__.py").write_text('''\
from src.data.preprocessor import (
    build_dataset,
    build_tokenizer,
    clean_text,
    generate_sequences,
    prepare_dataset,
)
__all__ = [
    "build_dataset", "build_tokenizer", "clean_text",
    "generate_sequences", "prepare_dataset",
]
''', encoding="utf-8")
print("  Written: src/data/__init__.py")

# ── 3. src/utils/__init__.py ─────────────────────────────────────────────────
(ROOT / "src" / "utils" / "__init__.py").write_text('''\
from src.utils.helpers import (
    compute_perplexity,
    format_metric_table,
    human_readable_size,
    plot_training_curves,
)
__all__ = [
    "compute_perplexity", "format_metric_table",
    "human_readable_size", "plot_training_curves",
]
''', encoding="utf-8")
print("  Written: src/utils/__init__.py")

# ── 4. src/__init__.py ───────────────────────────────────────────────────────
(ROOT / "src" / "__init__.py").write_text("", encoding="utf-8")
print("  Written: src/__init__.py")

# ── 5. tests/__init__.py ─────────────────────────────────────────────────────
(ROOT / "tests" / "__init__.py").write_text("", encoding="utf-8")
print("  Written: tests/__init__.py")

# ── 6. Delete all __pycache__ folders (stale .pyc files cause import errors) ─
print("\n  Clearing Python cache (.pyc files)...")
for cache_dir in ROOT.rglob("__pycache__"):
    shutil.rmtree(cache_dir, ignore_errors=True)
    print(f"    Deleted: {cache_dir.relative_to(ROOT)}")

# ── 7. Check which src files exist and their sizes ───────────────────────────
print("\n  Checking src module files:")
files_to_check = {
    "src/model/trainer.py"      : 8000,   # must be > 8 KB
    "src/model/architecture.py" : 2000,
    "src/model/predictor.py"    : 4000,
    "src/model/persistence.py"  : 4000,
    "src/data/preprocessor.py"  : 5000,
    "src/utils/helpers.py"      : 4000,
}

missing = []
too_small = []
for rel_path, min_size in files_to_check.items():
    full = ROOT / rel_path
    if not full.exists():
        print(f"    ❌ MISSING : {rel_path}")
        missing.append(rel_path)
    elif full.stat().st_size < min_size:
        size = full.stat().st_size
        print(f"    ⚠️  TOO SMALL: {rel_path} ({size} bytes — likely stub!)")
        too_small.append(rel_path)
    else:
        size = full.stat().st_size
        print(f"    ✅ OK       : {rel_path} ({size:,} bytes)")

# ── 8. Check for run_training_pipeline in trainer.py ─────────────────────────
trainer_path = ROOT / "src" / "model" / "trainer.py"
if trainer_path.exists():
    content = trainer_path.read_text(encoding="utf-8", errors="ignore")
    if "def run_training_pipeline" in content:
        print("\n  ✅ run_training_pipeline found in trainer.py")
    else:
        print("\n  ❌ run_training_pipeline MISSING from trainer.py!")
        too_small.append("src/model/trainer.py")

# ── 9. Test imports ──────────────────────────────────────────────────────────
print("\n  Testing imports...")
import warnings; warnings.filterwarnings("ignore")

results = {}
test_cases = [
    ("config",    "import config"),
    ("src.data",  "from src.data import build_dataset, clean_text"),
    ("src.model", "from src.model import model_exists, load_model, predict_next_words, run_training_pipeline"),
    ("src.utils", "from src.utils import compute_perplexity, format_metric_table"),
]

all_ok = True
for label, code in test_cases:
    try:
        exec(code)
        print(f"    ✅ {label}")
    except Exception as e:
        print(f"    ❌ {label}: {e}")
        results[label] = str(e)
        all_ok = False

# ── 10. Final report ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
if missing or too_small:
    print("  ⚠️  SOME FILES NEED TO BE REPLACED:")
    for f in set(missing + too_small):
        correct_name = "CORRECT_" + Path(f).name
        print(f"    {f}")
        print(f"    → Download '{correct_name}' from Claude")
        print(f"    → Copy it to: {f}")
    print()
    print("  After replacing, run this script again.")
elif all_ok:
    app_path = ROOT / "streamlit_app.py"
    app_size = app_path.stat().st_size if app_path.exists() else 0
    print("  ✅ ALL IMPORTS WORKING!")
    if app_size < 8000:
        print()
        print("  ⚠️  streamlit_app.py is too small (stub version)")
        print("     Download CORRECT_streamlit_app.py from Claude")
        print("     Rename it to streamlit_app.py and replace")
    else:
        print()
        print("  🚀 Ready! Run:")
        print("     streamlit run streamlit_app.py")
else:
    print("  ❌ Import errors remain. Check files above.")
print("=" * 60)
