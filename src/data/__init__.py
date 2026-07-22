"""
src/data
========
Data pipeline package.

Exports the public API so callers can write:
    from src.data import build_dataset, clean_text
instead of the full module path.
"""

from src.data.preprocessor import (
    build_dataset,
    build_tokenizer,
    clean_text,
    generate_sequences,
    prepare_dataset,
)

__all__ = [
    "build_dataset",
    "build_tokenizer",
    "clean_text",
    "generate_sequences",
    "prepare_dataset",
]
