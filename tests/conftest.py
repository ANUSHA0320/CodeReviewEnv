"""
tests/conftest.py
=================
Pytest configuration – ensures project root is on sys.path so all
package imports resolve correctly when running from any directory.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
