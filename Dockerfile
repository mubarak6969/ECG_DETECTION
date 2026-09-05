# ECG Intelligence - production image. The PTB-XL dataset is never baked
# into the image; only a trained model (artifacts/models/rcnn_model.h5)
# and the bundled demo record are needed at runtime.
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ecg/ ecg/
COPY app/ app/
COPY scripts/ scripts/
COPY data/demo/ data/demo/
COPY wsgi.py .

# Small, non-sensitive evaluation/calibration JSON reports (no model
# weights, no dataset) - lets the deployed UI show real metrics without
# requiring a volume mount. .dockerignore filters this to *.json only
# (model weights and PTB-XL data are excluded from the build context).
# REQUIRES scripts/evaluate_model.py to have been run locally first, same
# as the model file below - see README "Deployment".
COPY artifacts/reports/ artifacts/reports/

# Mount or bake a trained model at this path before running:
#   artifacts/models/rcnn_model.h5
# (produced by scripts/prepare_data.py + scripts/train_model.py against a
# local PTB-XL copy - see README, it is intentionally not part of the image)
ENV ECG_MODEL_PATH=/app/artifacts/models/rcnn_model.h5

# PORT is the standard env var injected by Render/Railway/Heroku-style
# platforms; 8000 is only the local-run default when it's unset.
ENV PORT=8000
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s \
    CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\",8000)}/health')" || exit 1

CMD waitress-serve --host=0.0.0.0 --port=${PORT:-8000} wsgi:app
