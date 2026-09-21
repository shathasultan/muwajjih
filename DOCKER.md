# Running the service with Docker (macOS)

## 1. Install Docker Desktop

1. Go to <https://www.docker.com/products/docker-desktop/> and download
   **Docker Desktop for Mac**.
2. Pick the right build for your chip:
   -  menu → **About This Mac**.
   - "Apple M1/M2/M3/M4" → download **Apple Silicon**.
   - "Intel" → download **Intel Chip**.
3. Open the downloaded `.dmg` and drag **Docker** into **Applications**.
4. Launch Docker from Applications. Approve the privileged-helper prompt when
   macOS asks (Docker needs it to manage networking).
5. Wait until the whale icon in the menu bar stops animating — that means the
   engine is running.

Verify from Terminal:

```bash
docker --version
docker info
```

`docker info` printing server details (not an error) means you are ready.

## 2. Build the image

From the project root:

```bash
docker build -t intent-service:local .
```

The build trains the model from source inside the container — the first build
takes a few minutes because it installs scikit-learn and fits the pipeline.
Later builds reuse cached layers and are much faster.

## 3. Run it

The service requires an API key, so generate one first:

```bash
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
```

Then run, passing that key in:

```bash
docker run -p 8000:8000 -e INTENT_API_KEYS="PASTE-YOUR-KEY-HERE" intent-service:local
```

Or with Compose, which reads `.env` automatically:

```bash
cp .env.example .env     # then edit .env and set INTENT_API_KEYS
docker compose up --build
```

## 4. Check it works

In a second Terminal tab:

```bash
# Readiness needs no key
curl http://127.0.0.1:8000/v1/ready

# Prediction needs the key
curl -X POST http://127.0.0.1:8000/v1/predict \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: PASTE-YOUR-KEY-HERE' \
  -d '{"text":"وصلني الجهاز مكسور ومب شغال"}'
```

Expected:

```json
{"ready":true,"model_loaded":true,"model_version":"v1.0.0"}
{"intent":"complaint","confidence":0.23,"low_confidence":false,"model_version":"v1.0.0"}
```

Interactive docs: open <http://127.0.0.1:8000/docs> in a browser.

## 5. Stop it

```bash
# If started with `docker run`: press Ctrl+C in that tab, then
docker ps            # find the container id
docker rm -f <id>

# If started with Compose:
docker compose down
```

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Cannot connect to the Docker daemon` | Docker Desktop is not running | Launch Docker from Applications and wait for the whale icon to settle |
| Container exits immediately, logs say `INTENT_API_KEYS is empty` | Secure-by-default guard | Pass `-e INTENT_API_KEYS="..."` or set it in `.env` |
| `401 missing or invalid API key` | Key not sent or mistyped | Send the `X-API-Key` header with the exact key |
| `429 rate limit exceeded` | More than 60 requests in 60s | Wait for the window, or raise `INTENT_RATE_LIMIT_REQUESTS` |
| `port is already allocated` | Something else uses 8000 | Run with `-p 8001:8000` and call port 8001 |
| Build fails pulling `python:3.12-slim` | No network or a blocked registry | Check connectivity; on a restricted network, use a registry mirror |
