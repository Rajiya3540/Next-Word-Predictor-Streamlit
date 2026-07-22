"""
conftest.py
===========
Pytest configuration — adds the project root to sys.path so that
every test file can do  `import config`  and  `from src.x import y`
without needing its own sys.path manipulation.

This file is automatically discovered by pytest before any test runs.
"""

import sys
from pathlib import Path

# Insert project root (the directory containing this file) at the front
# of sys.path so our packages are importable in any pytest invocation.
sys.path.insert(0, str(Path(__file__).parent))
