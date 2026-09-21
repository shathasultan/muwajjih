# Intent Service — Arabic Customer-Message Classifier

A production-shaped ML service: a text classifier trained from scratch on a
reproducible synthetic dataset, wrapped in a clean-architecture package,
served over a REST API, containerised, and gated by automated tests and CI.

Built as the capstone for **SDA-AIE-113 — Software Engineering Practices for
AI Systems**.

---

## What it does

Classifies an Arabic customer-support message into one of six intents:

| Intent | Meaning |
|---|---|
| `complaint` | شكوى عن منتج تالف أو خدمة سيئة |
| `price_inquiry` | سؤال عن السعر أو الخصم أو التقسيط |
| `support_request` | طلب مساعدة تقنية في استخدام المنتج |
| `praise` | ثناء على المنتج أو الخدمة |
| `order_status` | سؤال عن حالة الطلب أو موعد التوصيل |
| `return_refund` | طلب إرجاع أو استبدال أو استرداد مبلغ |

```bash
curl -X POST http://127.0.0.1:8000/v1/predict \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: your-key' \
  -d '{"text":"وصلني الجهاز مكسور ومب شغال"}'
```
```json
{"intent":"complaint","confidence":0.31,"low_confidence":false,
 "model_version":"v1.0.0","trace_id":"4398c99b-54ef-4735-9217-2614b20dae8a"}
```

---

## Quick start

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,train]"

python -m train.generate_dataset   # deterministic, seed=42
python -m train.train_model        # writes models/intent_model.joblib

cp .env.example .env               # then set INTENT_API_KEYS in .env
pytest                             # 71 tests
uvicorn intent_service.api.main:create_app --factory --reload
```

With Docker (full macOS walkthrough in [DOCKER.md](DOCKER.md)):

```bash
docker build -t intent-service:local .
docker run -p 8000:8000 -e INTENT_API_KEYS="your-key" intent-service:local
# or, reading .env automatically:
docker compose up --build
```

---

## How this project meets each programme objective

| # | Objective | Where it lives |
|---|---|---|
| 1 | RESTful APIs serving ML models as reliable services | `src/intent_service/api/` — four endpoints, pydantic-validated contracts, correct status codes (200/401/413/422/429/500/503), API-key auth, rate limiting, graceful degradation when the artifact is missing |
| 2 | Containerised AI services with Docker and Compose | `Dockerfile` (two-stage: trains in the builder, ships only runtime deps), `docker-compose.yml` (healthcheck, restart policy, env-based config) |
| 3 | Automated unit, integration, and model-behaviour tests | `tests/unit/` (no model loaded), `tests/integration/` (real API via TestClient), `tests/behavioural/` (model quality gates) |
| 4 | CI/CD pipelines that verify and deploy AI code changes | `.github/workflows/ci.yml` — three dependent stages: quality → train & test → build image & smoke-test a live container |
| 5 | Clean architecture and configuration management | `domain/` → `service/` → `adapters/` → `api/`, dependencies pointing inward only; one typed `Settings` object (`config.py`) |
| 6 | Code quality via reviews, linters, static analysis | `ruff` (lint + format), `mypy --strict`, `.pre-commit-config.yaml`, `.github/pull_request_template.md` with an architecture-specific review checklist |
| 7 | A containerised model service capstone | This repository |

---

## Architecture

Dependencies point inward only. Nothing in `domain/` or `service/` knows that
scikit-learn, FastAPI, or a filesystem exist.

```
api/main.py ───────────── composition root: the only place concretes are wired
   │
   ├── config.py ───────── Settings (pydantic-settings, INTENT_ prefix)
   ├── api/schemas.py ──── HTTP wire contracts (separate from domain entities)
   ├── adapters/ ───────── SklearnIntentModel — the only module importing sklearn/joblib
   │       implements ↓
   ├── service/
   │     ├── interfaces.py  IntentModel protocol (predict, model_version, labels)
   │     └── classifier.py  IntentClassifier — orchestration only
   └── domain/
         └── entities.py    CustomerMessage, IntentPrediction, Intent
```

| Layer | May import | Responsibility |
|---|---|---|
| `domain` | stdlib, pydantic | The vocabulary: what a message and a prediction are |
| `service` | `domain` | Orchestrating the use case against a port |
| `adapters` | anything | Translating a port into a concrete ML library |
| `config` / `api` | anything | Configuration, wiring, HTTP |

### Design decisions

**The model is built from source, never downloaded.** `train/generate_dataset.py`
is seeded (`SEED = 42`) and its output is sorted before writing, so it produces
byte-identical CSVs on every run — verified by regenerating and diffing
checksums. The Docker build and the CI pipeline both regenerate the dataset and
retrain rather than consuming a committed binary, so "the model" is always
reproducible from code.

**Feature extraction lives inside the artifact.** The TF-IDF vectoriser and the
classifier are fit together as one sklearn `Pipeline` and pickled as a unit, so
the serving adapter never re-implements preprocessing. This is the structural
defence against training/serving skew: there is no second copy of the feature
logic that could drift.

**The artifact validates itself at load time.** `SklearnIntentModel.load()`
compares `pipeline.classes_` against the artifact's `labels` list and raises on
mismatch, turning a silent label-mapping bug into an immediate startup failure.

**Failures are not swallowed.** `IntentClassifier` has no exception handling at
all — a broken model raises, and the API layer decides the response (500 with a
logged traceback, and an incremented error counter). Returning a plausible-looking
default intent on failure would hide outages from every downstream consumer.

**A missing artifact degrades, it does not crash-loop.** If the model file is
absent at startup the service still boots: `/v1/health` keeps answering 200
(the process is alive), while `/v1/ready` reports 503 with
`model_loaded: false` and `/v1/predict` returns 503. An orchestrator sees a clear unhealthy
signal instead of a container restarting with no explanation.

**API schemas are separate from domain entities.** `api/schemas.py` holds the
wire contracts; `domain/entities.py` holds the business vocabulary. They look
similar today, but keeping them apart means a future HTTP concern (versioning,
an envelope, pagination) never leaks inward.

---

## API

All routes are versioned under `/v1`, so a breaking change can ship as
`/v2` while existing clients keep working.

| Method | Path | Auth | Purpose |
|---|---|---|---|
| `POST` | `/v1/predict` | API key | Classify one message |
| `GET` | `/v1/health` | none | **Liveness** — no I/O, no model access, answers instantly |
| `GET` | `/v1/ready` | none | **Readiness** — `503` until the model is loaded *and* warmed up |
| `GET` | `/v1/model-info` | API key | Model version and the label set it was trained on |
| `GET` | `/v1/metrics` | API key | Request counts, per-intent breakdown, error count |
| `GET` | `/docs` | none | Auto-generated OpenAPI documentation |

**Liveness and readiness are deliberately separate.** `/v1/health` answers
"is the process alive" and touches nothing — restarting on its failure is
correct only when the process itself is broken. `/v1/ready` answers "can this
instance serve traffic", and stays `503` until a warm-up prediction has
actually returned, so a load balancer never hands a real user the slow first
request.

Every prediction response carries a unique `trace_id`, also written to the
logs, so one request can be found among thousands.

Validation is enforced by pydantic at the boundary: empty text, missing fields,
and text over 2000 characters all return `422` before the model is ever called —
and those requests are deliberately *not* counted as served predictions in
`/metrics`.

---

## Configuration

`INTENT_API_KEYS` is **required** -- the service refuses to start without it
unless authentication is explicitly disabled. Everything else is optional and
defaults to a safe value. Override via environment variables (prefix
`INTENT_`) or a `.env` file.

| Variable | Default | Purpose |
|---|---|---|
| `INTENT_API_KEYS` | *(none)* | **Required.** Comma-separated accepted API keys |
| `INTENT_REQUIRE_API_KEY` | `true` | Set `false` for local development only |
| `INTENT_RATE_LIMIT_REQUESTS` | `60` | Requests allowed per window, per caller |
| `INTENT_RATE_LIMIT_WINDOW_SECONDS` | `60` | Length of the rate-limit window |
| `INTENT_CORS_ORIGINS` | *(empty)* | Empty denies all cross-origin requests |
| `INTENT_MAX_REQUEST_BYTES` | `16384` | Bodies larger than this get `413` |
| `INTENT_ENABLE_DOCS` | `false` | Set `true` to expose the interactive `/docs` |
| `INTENT_TRUST_PROXY_HEADERS` | `false` | Read `X-Forwarded-For` for rate-limit identity — only behind a trusted proxy |
| `INTENT_MODEL_PATH` | `models/intent_model.joblib` | Where the artifact is read from |
| `INTENT_CONFIDENCE_FLOOR` | `0.15` | Below this, responses set `low_confidence: true` |
| `INTENT_LOG_LEVEL` | `INFO` | Log verbosity |
| `INTENT_API_TITLE` | `Intent Classification Service` | Title shown in `/docs` |

The service **refuses to start** if `INTENT_REQUIRE_API_KEY` is true while
`INTENT_API_KEYS` is empty — a misconfiguration fails loudly at boot instead of
silently rejecting every request.

---

## Testing strategy

Three layers, each answering a different question:

| Suite | Question it answers | Needs the real model? |
|---|---|---|
| `tests/unit/` | Is the logic correct in isolation? | No — uses a `ConstantModel` double |
| `tests/integration/` | Does the HTTP contract hold end to end? | Yes |
| `tests/behavioural/` | Is the model actually any good? | Yes |

The behavioural suite is the one that treats the model as the thing under test:

- **Accuracy gate** — at least 90% on the held-out test split, so a training
  change that quietly degrades quality fails the build.
- **Generalisation probes** — six hand-written sentences that appear in *no*
  template in `train/generate_dataset.py`, with different products and phrasing.
  Passing these means the model generalised rather than memorised.
- **Determinism** — identical input must give identical output.
- **Bounded confidence** — adversarial inputs (emoji, digits, 500-character
  strings, mixed English/Arabic) must still yield a confidence in `[0, 1]`.

```
$ pytest
71 passed
```

---

## Security

Full policy in [SECURITY.md](SECURITY.md). Every control below is covered by a
test and was verified against a running service.

| Control | Behaviour |
|---|---|
| API key (`X-API-Key`) | Required on `/predict`, `/model-info`, `/metrics` → `401` without it |
| Constant-time comparison | `secrets.compare_digest` — no timing oracle on the key |
| Rate limiting | 60 req/60s per caller → `429` with `Retry-After` |
| Body size cap | 16 KiB → `413`, checked before the body is read |
| CORS | **Denied by default**; opened only via `INTENT_CORS_ORIGINS` |
| Security headers | `nosniff`, `DENY`, `no-store`, CSP, Permissions-Policy — on every response including errors |
| Non-root container | `appuser` (uid 1000), `no-new-privileges`, read-only filesystem |
| Fail-fast config | Refuses to boot if auth is required but no keys are set |

Three decisions worth defending:

- **`/health` is open and exempt from rate limiting.** Liveness probes carry no
  credentials, and a probe must never be throttled into a false "unhealthy".
  It exposes only whether the artifact loaded.
- **Authentication runs before inference.** An unauthenticated caller cannot
  spend model compute — asserted by `test_auth_runs_before_the_model`.
- **Rate limiting runs before authentication.** Counting rejected requests too
  is what stops an attacker flooding the service with cheap `401`s.

Generate a key with:

```bash
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
```

---

## Honest limitations

These are real and worth stating rather than hiding:

1. **The dataset is synthetic and lexically separable.** Test accuracy is 100%
   across a balanced 45-per-class split, which reflects the data being
   template-generated, not a model that would hold up on real customer
   messages. The generalisation probes in the behavioural
   suite are the meaningful signal here, not the headline accuracy. Replacing
   `train/generate_dataset.py` with real labelled messages is the single highest-value
   next step.
2. **Every class is balanced, but only because the generator was tuned to make
   it so.** Each label reaches its full quota of unique examples, and the
   generator now prints a warning naming any label that exhausts its template
   combinations early -- silence there would hide a skewed split behind a
   healthy-looking accuracy number.
3. **`LinearSVC` has no calibrated probabilities.** The `confidence` field is a
   softmax over decision-function margins — useful for ranking and for the
   `low_confidence` flag, but it is not a true probability and should not be read
   as one.
4. **`/metrics` is an in-process counter.** It resets on restart and is not
   aggregated across replicas. A real deployment would export to Prometheus.
5. **API keys are static shared secrets.** There is no rotation, no expiry and
   no per-key scope. A production system should move to short-lived tokens
   (OAuth2 / JWT) issued per client.
6. **The body-size limit relies on `Content-Length`.** Requests without it are
   rejected with `411` rather than being streamed and measured, so a proxy-level
   limit is still the right outer defence.
7. **Rate-limit identity falls back to the socket IP.** Behind a reverse proxy
   every unauthenticated caller would share one bucket, so set
   `INTENT_TRUST_PROXY_HEADERS=true` *only* when a trusted proxy overwrites
   `X-Forwarded-For` -- the header is caller-controlled otherwise.

---

## Project layout

```
intent-service/
├── Dockerfile                  # two-stage: trains in builder, ships runtime only
├── docker-compose.yml
├── pyproject.toml              # deps, ruff, mypy, pytest config
├── .pre-commit-config.yaml
├── .github/
│   ├── workflows/ci.yml        # quality → train & test → image smoke test
│   └── pull_request_template.md
├── train/
│   ├── generate_dataset.py     # deterministic synthetic data (seed=42)
│   └── train_model.py          # TF-IDF + LinearSVC → versioned joblib artifact
├── src/intent_service/
│   ├── config.py
│   ├── domain/entities.py
│   ├── service/{interfaces,classifier}.py
│   ├── adapters/sklearn_model.py
│   └── api/{main,schemas,metrics}.py
└── tests/{unit,integration,behavioural}/
```

---

## Licence

Proprietary — all rights reserved. See [LICENSE](LICENSE). Instructors and
authorised examiners of SDA-AIE-113 may access and run this code for
assessment purposes only.
