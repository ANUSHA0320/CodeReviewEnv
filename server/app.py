"""
server/app.py
=============
Server entry point for CodeReviewEnv-v0.
Runs the FastAPI REST API via uvicorn on port 7860.

Usage
-----
    python -m server.app
    # or via script entry point:
    code-review-server

Environment Variables
---------------------
    HOST  : bind host  (default: 0.0.0.0)
    PORT  : bind port  (default: 7860)
    RELOAD: enable hot-reload for development (default: false)
"""

from __future__ import annotations

import os
import sys

# Ensure repo root is on the path when invoked directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> None:
    """Entry point used by pyproject.toml [project.scripts]."""
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "7860"))
    reload = os.getenv("RELOAD", "false").lower() == "true"

    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
