"""
data/download_corpus.py
=======================
Download public-domain books from Project Gutenberg.

Usage
-----
    python data/download_corpus.py

Output
------
    data/alice.txt        — Alice in Wonderland (~26K words)
    data/sherlock.txt     — Sherlock Holmes (~107K words)
    data/frankenstein.txt — Frankenstein (~78K words)
    data/all_books.txt    — Combined corpus (~211K words)  ← recommended

To use a specific corpus, set CORPUS_PATH in config.py:
    CORPUS_PATH = DATA_DIR / "all_books.txt"
"""

from __future__ import annotations

import re
import sys
import urllib.request
from pathlib import Path

# Allow running from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

BOOKS: dict[str, str] = {
    "alice.txt"       : "https://www.gutenberg.org/files/11/11-0.txt",
    "sherlock.txt"    : "https://www.gutenberg.org/files/1661/1661-0.txt",
    "frankenstein.txt": "https://www.gutenberg.org/files/84/84-0.txt",
}


def _strip_gutenberg(text: str) -> str:
    """Remove Project Gutenberg header/footer boilerplate."""
    start = re.search(r"\*\*\* start of (the|this) project gutenberg", text, re.I)
    end   = re.search(r"\*\*\* end of (the|this) project gutenberg",   text, re.I)
    if start:
        text = text[start.end():]
    if end:
        text = text[: end.start()]
    return text.strip()


def main() -> None:
    parts: list[str] = []

    for filename, url in BOOKS.items():
        dest = config.DATA_DIR / filename

        if dest.exists():
            print(f"  ✅  {filename} already exists — skipping.")
        else:
            print(f"  ⬇   Downloading {filename} ...", end=" ", flush=True)
            try:
                urllib.request.urlretrieve(url, dest)
                print("done.")
            except Exception as exc:
                print(f"FAILED ({exc})")
                continue

        text = dest.read_text(encoding="utf-8", errors="ignore")
        parts.append(_strip_gutenberg(text))

    if not parts:
        print("\n❌ No books downloaded. Check your internet connection.")
        sys.exit(1)

    combined_path = config.DATA_DIR / "all_books.txt"
    combined_path.write_text("\n\n".join(parts), encoding="utf-8")

    total_chars = sum(len(p) for p in parts)
    print(f"\n✅ Combined corpus saved → {combined_path}")
    print(f"   Total characters : {total_chars:,}")
    print(f"   Approx. words    : {total_chars // 5:,}")
    print("\nTo use the combined corpus, set in config.py:")
    print('   CORPUS_PATH = DATA_DIR / "all_books.txt"')


if __name__ == "__main__":
    main()
