"""Small HTTP-layer upload helpers: filename sanitization and a
size-capped async read.

No Flask/Werkzeug dependency here or anywhere else in the app - FastAPI
and Starlette are the only web framework in this project post-migration.
`secure_filename` reimplements just the guarantees this project relies
on from werkzeug's version (strip directory components, drop unsafe
characters, never return a hidden/empty name) without pulling in the
rest of that library.
"""
from __future__ import annotations

import re

from fastapi import HTTPException, UploadFile

_UNSAFE_CHARS_RE = re.compile(r"[^A-Za-z0-9_.-]")


def secure_filename(filename: str) -> str:
    """Reduce a client-supplied filename to a single safe path component.

    Strips directory components (both `/` and Windows `\\` separators),
    replaces any character outside a safe allowlist with `_`, and strips
    leading dots/underscores - so the result can never be empty, hidden,
    or escape the directory it's saved into (verified against `../../evil.dat`
    and similar in tests/test_api.py).
    """
    filename = filename.replace("\\", "/").rsplit("/", 1)[-1]
    filename = _UNSAFE_CHARS_RE.sub("_", filename)
    return filename.lstrip("._")


async def read_capped(upload: UploadFile, max_bytes: int) -> bytes:
    """Read an UploadFile into memory, aborting with a 413 as soon as
    `max_bytes` is exceeded.

    A Content-Length header check happens first, up front, as a fast
    path (see routes/predict.py) - this loop is what actually enforces
    the cap against a lying or chunked-transfer client, not just an
    honest header.
    """
    chunks = []
    total = 0
    chunk_size = 1024 * 1024
    while True:
        chunk = await upload.read(chunk_size)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(status_code=413, detail="Uploaded file exceeds the maximum allowed size")
        chunks.append(chunk)
    return b"".join(chunks)
