"""
server/app.py — OpenEnv multi-mode deployment entry point.
Loads the FastAPI app directly from the root server.py file
without importing the server package (to avoid circular imports).
"""
import os
import importlib.util
import uvicorn

# Load root-level server.py directly by file path
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "_root_server", os.path.join(_root, "server.py")
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

# Re-export the FastAPI app so uvicorn can find it as server.app:app
app = _mod.app


def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()

__all__ = ["app", "main"]
