"""
Shared fixtures for the test suite.

Both api.py and app/main.py import from `common.constants`, so tests need the
project root on sys.path regardless of where pytest is invoked from.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
