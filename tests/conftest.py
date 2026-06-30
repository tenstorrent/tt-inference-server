"""Configure sys.path so that the orchestrator package is importable."""
import sys
import os

# Add the repo root (parent of tests/) to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
