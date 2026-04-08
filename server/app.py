"""
server/app.py — OpenEnv multi-mode deployment entry point.
"""

import os
import importlib.util
import uvicorn
import threading

# Load root-level server.py
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "_root_server", os.path.join(_root, "server.py")
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

app = _mod.app


# 🔥 RUN INFERENCE IN BACKGROUND
def run_inference():
    try:
        import inference
    except Exception as e:
        print(f"[ERROR] Failed to run inference: {e}", flush=True)


def main():
    # 🔥 start inference in background thread
    threading.Thread(target=run_inference).start()

    # start server
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()


__all__ = ["app", "main"]
