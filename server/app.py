"""
server/app.py — OpenEnv multi-mode deployment entry point.
This re-exports the FastAPI app from the root server.py so the
validator can find it at server.app:app.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server import app  # noqa: F401 — re-export

__all__ = ["app"]
