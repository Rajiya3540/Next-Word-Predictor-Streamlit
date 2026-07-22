@echo off
echo ================================================
echo   Next Word Predictor - Structure Fix Script
echo ================================================
echo.

REM ── Step 1: Create all required subfolders ───────────────────────────────
echo [1/6] Creating folder structure...
if not exist "src\data"  mkdir "src\data"
if not exist "src\model" mkdir "src\model"
if not exist "src\utils" mkdir "src\utils"
if not exist "data"      mkdir "data"
if not exist "models"    mkdir "models"
if not exist "results"   mkdir "results"
if not exist "tests"     mkdir "tests"
echo       Done.

REM ── Step 2: Move files into correct src subfolders ───────────────────────
echo [2/6] Moving files to correct locations...

if exist "preprocessor.py"    move /Y "preprocessor.py"    "src\data\preprocessor.py"
if exist "architecture.py"    move /Y "architecture.py"    "src\model\architecture.py"
if exist "trainer.py"         move /Y "trainer.py"         "src\model\trainer.py"
if exist "predictor.py"       move /Y "predictor.py"       "src\model\predictor.py"
if exist "persistence.py"     move /Y "persistence.py"     "src\model\persistence.py"
if exist "helpers.py"         move /Y "helpers.py"         "src\utils\helpers.py"
if exist "test_pipeline.py"   move /Y "test_pipeline.py"   "tests\test_pipeline.py"
if exist "download_corpus.py" move /Y "download_corpus.py" "data\download_corpus.py"
if exist "data_init.py"       move /Y "data_init.py"       "src\data\__init__.py"
if exist "model_init_p4.py"   move /Y "model_init_p4.py"   "src\model\__init__.py"
if exist "model_init_p5p6.py" move /Y "model_init_p5p6.py" "src\model\__init__.py"
if exist "utils_init.py"      move /Y "utils_init.py"      "src\utils\__init__.py"
echo       Done.

REM ── Step 3: Create __init__.py files ─────────────────────────────────────
echo [3/6] Creating __init__.py files...

REM src\__init__.py
type nul > "src\__init__.py"

REM src\data\__init__.py (if not already moved)
if not exist "src\data\__init__.py" (
    echo from src.data.preprocessor import build_dataset, build_tokenizer, clean_text, generate_sequences, prepare_dataset > "src\data\__init__.py"
)

REM src\utils\__init__.py (if not already moved)
if not exist "src\utils\__init__.py" (
    echo from src.utils.helpers import compute_perplexity, format_metric_table, human_readable_size, plot_training_curves > "src\utils\__init__.py"
)

REM tests\__init__.py
type nul > "tests\__init__.py"

echo       Done.

REM ── Step 4: Create src\model\__init__.py if missing ──────────────────────
echo [4/6] Checking src\model\__init__.py...
if not exist "src\model\__init__.py" (
    (
        echo from src.model.architecture import build_model
        echo from src.model.persistence import load as load_model, model_exists, save as save_model
        echo from src.model.predictor import predict_next_words, preprocess_seed
        echo from src.model.trainer import evaluate, get_callbacks, run_training_pipeline, split_dataset, train
        echo __all__ = ["build_model","evaluate","get_callbacks","load_model","model_exists","predict_next_words","preprocess_seed","run_training_pipeline","save_model","split_dataset","train"]
    ) > "src\model\__init__.py"
)
echo       Done.

REM ── Step 5: Delete unwanted files ────────────────────────────────────────
echo [5/6] Removing old/unwanted files...
if exist "index.html"                 del /Q "index.html"
if exist "lstm_next_word_predictor.py" del /Q "lstm_next_word_predictor.py"
if exist "config.cpython-312.pyc"     del /Q "config.cpython-312.pyc"
if exist "gitignore.txt" (
    if not exist ".gitignore" rename "gitignore.txt" ".gitignore"
)
REM Delete root-level __init__.py only if it's 0 bytes
for %%F in ("__init__.py") do (
    if %%~zF==0 del /Q "__init__.py"
)
echo       Done.

REM ── Step 6: Copy model files if they exist in parent folder ──────────────
echo [6/6] Checking model files...
if exist "..\lstm_next_word_model.keras" (
    if not exist "models\lstm_best_model.keras" (
        copy "..\lstm_next_word_model.keras" "models\lstm_best_model.keras"
        echo       Copied lstm_next_word_model.keras to models\lstm_best_model.keras
    )
)
if exist "..\tokenizer_data.npy" (
    if not exist "models\tokenizer_data.npy" (
        copy "..\tokenizer_data.npy" "models\tokenizer_data.npy"
        echo       Copied tokenizer_data.npy to models\
    )
)
echo       Done.

echo.
echo ================================================
echo   Structure fixed!
echo ================================================
echo.
echo IMPORTANT: streamlit_app.py must be 10+ KB
echo Current size:
for %%F in ("streamlit_app.py") do echo   streamlit_app.py = %%~zF bytes
echo.
echo If it shows less than 8000 bytes, download the
echo correct version from Claude outputs and replace it.
echo.
echo Then run:  streamlit run streamlit_app.py
echo ================================================
pause
