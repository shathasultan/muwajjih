# ---------------------------------------------------------------------------
# Stage 1: builder -- installs the training extras and trains the model
# FROM SOURCE. No binary artifact is downloaded or copied in from the host:
# the deterministic generator (train/generate_dataset.py, fixed seed) builds
# the dataset, then train/train_model.py fits the pipeline, entirely inside
# this build. Re-running `docker build` reproduces the same model, which is
# the whole point of pinning the seed.
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS builder

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src
COPY train ./train

RUN pip install --no-cache-dir -e ".[train]"

RUN python -m train.generate_dataset && python -m train.train_model

# ---------------------------------------------------------------------------
# Stage 2: runtime -- only what's needed to serve the API. No pandas, no
# training code, no dev/test dependencies: a smaller image and a smaller
# attack surface than the builder stage.
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

WORKDIR /app

RUN useradd --create-home --uid 1000 appuser

COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir .

# Only the trained artifact + its metrics report cross the stage boundary --
# not the raw generated CSVs, which stay in the builder stage.
COPY --from=builder /app/models ./models

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/v1/ready', timeout=3)" || exit 1

# Exec form, so uvicorn is PID 1 and receives SIGTERM directly from
# `docker stop`. The shell form would put /bin/sh at PID 1, which does not
# forward signals -- the container would ignore SIGTERM and be SIGKILLed ten
# seconds later, dropping every in-flight request. scripts/smoke.sh asserts
# the clean exit code this produces.
#
# --timeout-graceful-shutdown gives in-flight requests a bounded window to
# finish before the worker is torn down; it is deliberately shorter than
# compose's stop_grace_period so uvicorn, not the daemon, decides how the
# shutdown ends.
CMD ["uvicorn", "intent_service.api.main:create_app", \
     "--factory", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--timeout-graceful-shutdown", "15", \
     "--no-server-header"]
