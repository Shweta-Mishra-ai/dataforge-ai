"""conftest.py — ensures project root is in sys.path for all tests."""
import sys
import os

# Add project root to path so `from core.xxx import ...` works
sys.path.insert(0, os.path.dirname(__file__))
