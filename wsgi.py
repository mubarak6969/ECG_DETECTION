"""Production entrypoint. Preferred (used by the Dockerfile and most
PaaS "start command" fields):

    waitress-serve --host=0.0.0.0 --port=$PORT wsgi:app

`PORT` is the standard env var injected by Render/Railway/Heroku-style
platforms; running this file directly (`python wsgi.py`) is an
equivalent fallback for platforms that only support a plain Python
start command, and reads the same env var.
"""
import os

from app.main import app

if __name__ == "__main__":
    from waitress import serve
    port = int(os.environ.get("PORT", 8000))
    serve(app, host="0.0.0.0", port=port)
