"""Root conftest.py — ensures the project root is on sys.path for all tests.

This allows test files to import top-level modules (e.g. `from main import app`)
without needing to install the package or manipulate paths in each test file.
"""

import sys
from pathlib import Path

# Add project root to sys.path so `main`, `app.*`, and `scripts.*` are importable
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
