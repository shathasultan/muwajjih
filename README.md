# Muwajjih — Arabic customer-message triage

A production-shaped ML service that makes an **operational decision**, not a
prediction. It reads an Arabic customer message and decides one of three
things: route it automatically, send it to a human, or return it to the sender.

This project was completed as part of the **SDA-AIE-113 — Software Engineering
Practices for AI Systems** training program at **SDAIA Academy**, under the
supervision of **Abdullah Khalid AlShahrani**.

The portfolio demonstrates the practical application of software engineering
practices for AI systems — building a production-style AI/ML service through
clean architecture, a well-defined API contract, containerization, a layered
automated testing suite, a CI/CD pipeline with branch protection, and safe
configuration, secrets, and logging management.

Official SDAIA Academy GitHub: https://github.com/SDAIAAcademy

> ### Track B — own idea
>
> Customer-message triage, following the same decision shape as *Muwajjih*
> in the ready list (route to the right department + priority) applied to a
> customer-support queue rather than a municipal one. It meets each Track B
> condition:
>
> | Condition | How |
> |---|---|
> | A clear decision between 2–3 options | Exactly three, mutually exclusive: `auto_route` / `human_review` / `reject` — not open-ended generation |
> | Tabular data or short text only | Arabic text, 1–2 000 characters. No images, audio or video |
> | A lightweight model is enough | scikit-learn: TF-IDF → LogisticRegression, 0.61 MB, trains in 3.3 s |
> | At least one deterministic behavioural test | An urgency term (`حريق`, `تسرب غاز`) must **always** raise priority to `urgent` and can never produce a rejection — pinned in [`tests/behavioural/test_directional.py`](tests/behavioural/test_directional.py) across every term in the list |

```bash
curl -X POST http://127.0.0.1:8000/v1/predict \
  -H 'Content-Type: application/json' -H 'X-API-Key: your-key' \
  -d '{"text":"وصلني الجهاز مكسور ومب شغال"}'
```

```json
{
  "data": {
    "action": "auto_route",
    "department": "quality_assurance",
    "priority": "normal",
    "intent": "complaint",
    "confidence": 0.974,
    "urgency_signals": [],
    "reason": "confident_classification: confidence 0.974 >= auto_route_floor 0.600",
    "model_version": "v1.0.0",
    "cached": false
  },
  "error": null,
  "meta": { "trace_id": "4398c99b-54ef-4735-9217-2614b20dae8a" }
}
```

---

## A verified run

Every response below is copied verbatim from a real run of this service, in
the container built from this repository, on 2026-09-22. Nothing is
illustrative. Reproduce it with `make compose-up` and the commands in the next
section.

**1 — Readiness.** Redis reaches `Healthy` before the API is started at all;
`/v1/ready` then reports which cache backend actually attached, rather than
which one was configured.

```json
{"data":{"ready":true,"model_loaded":true,"model_version":"v1.0.0",
         "cache_backend":"redis","cache_healthy":true},
 "error":null,"meta":{"trace_id":"1fcde103-71e8-454a-8eca-36fa9e92bc66"}}
```

**2 — A confident message is acted on automatically.** Note that the answer is
a decision with a written justification, not a label.

```
POST /v1/predict  {"text":"وين طلبي؟ صار له اسبوع وما وصل"}
```
```json
{"data":{"action":"auto_route","department":"logistics","priority":"normal",
         "intent":"order_status","confidence":0.9566265812119552,
         "urgency_signals":[],
         "reason":"confident_classification: confidence 0.957 >= auto_route_floor 0.600",
         "model_version":"v1.0.0","cached":false},
 "error":null,"meta":{"trace_id":"107f3eec-e272-46c4-adbe-69581a4cec0e"}}
```

**3 — The safety rule overrides a model that is wrong.** This is the case the
whole design exists for. The classifier reads a gas-leak report as `praise`,
at 0.370 confidence — below the 0.600 automation floor, so on the model's word
alone this message would have gone to a human queue as ordinary feedback.

```
POST /v1/predict  {"text":"في تسرب غاز من السخان والرائحة قوية"}
```
```json
{"data":{"action":"auto_route","department":"safety","priority":"urgent",
         "intent":"praise","confidence":0.3699270872536515,
         "urgency_signals":["تسرب","غاز"],
         "reason":"safety_escalation: message contains urgency terms (تسرب, غاز); routed to safety at urgent priority, overriding both the model's department (customer_relations) and its confidence (0.370)",
         "model_version":"v1.0.0","cached":true},
 "error":null,"meta":{"trace_id":"fb90902c-dcf1-41c8-80e1-849cc4b9eca7"}}
```

Three things in that one response:

- `priority: urgent` and `action: auto_route` — escalated despite the model
- `department: safety` — and routed to the team that handles emergencies, not
  to `customer_relations`, which is where the model's own answer pointed
- `reason` names the department it overrode and the confidence it ignored, so
  an operator finding a `praise` message in the safety queue reads an
  explanation rather than a bug

`cached: true` is the extension working: this exact text had been sent before,
so the decision came from Redis instead of a second inference — measured at
186× faster in BENCHMARKS.md.

**4 — A malformed request gets the same envelope.** `data` is null, `error` is
populated, the trace id is still in the same place, and the offending key is
named.

```
POST /v1/predict  {"txt":"حقل غلط"}        → 422
```
```json
{"data":null,
 "error":{"code":"validation_error","message":"the request body failed validation",
          "fields":["body.text","body.txt"]},
 "meta":{"trace_id":"522a8e2c-fced-40ff-8ea1-5a927a3375d0"}}
```

**5 — An unauthenticated request is refused.**

```
POST /v1/predict  (no X-API-Key)           → 401
```

---

## Run it in under 10 minutes

You need **Python 3.12+**, and **Docker** for the container path.

### Path A — Docker Compose (the service plus its Redis)

```bash
git clone https://github.com/shathasultan/muwajjih.git
cd muwajjih

cp .env.example .env
# Edit .env and set INTENT_API_KEYS to any value. The service refuses to
# start without one, so there is no accidental open deployment.

make compose-up      # waits on REAL health, not just "container created"
```

Then:

```bash
curl -s http://127.0.0.1:8000/v1/ready | jq
# {"data":{"ready":true,"model_loaded":true,"cache_backend":"redis", ...}}

KEY=$(grep INTENT_API_KEYS .env | cut -d= -f2)

# A valid request
curl -s -X POST http://127.0.0.1:8000/v1/predict \
  -H 'Content-Type: application/json' -H "X-API-Key: $KEY" \
  -d '{"text":"وين طلبي؟ صار له اسبوع وما وصل"}' | jq

# An emergency — escalated regardless of what the model thinks
curl -s -X POST http://127.0.0.1:8000/v1/predict \
  -H 'Content-Type: application/json' -H "X-API-Key: $KEY" \
  -d '{"text":"في تسرب غاز من السخان والرائحة قوية"}' | jq '.data.priority'
# "urgent"

# A malformed request — same envelope, trace id still present
curl -s -X POST http://127.0.0.1:8000/v1/predict \
  -H 'Content-Type: application/json' -H "X-API-Key: $KEY" \
  -d '{"txt":"unknown field"}' | jq

make compose-down
```

There is **no `latest` tag**. Images are published to GHCR addressed only by
commit SHA:

```bash
docker pull ghcr.io/shathasultan/muwajjih:<commit-sha>
```

### Path B — local Python

```bash
python3.12 -m venv .venv && source .venv/bin/activate
make install        # editable install + dev/train extras + pre-commit hooks
make train          # generate the seeded dataset, then fit the model (~4s)
make gate           # lint, types, architecture contract, unit + integration

INTENT_API_KEYS=local-dev-key \
  uvicorn intent_service.api.main:create_app --factory --reload
```

### Every command

| Command | What it does |
|---|---|
| `make install` | Editable install with dev + train extras, installs pre-commit |
| `make train` | Regenerate the seeded dataset and retrain (~3.4 s) |
| `make lint` | Ruff, format check, strict mypy, **and the import-linter contract** |
| `make gate` | **Fast gate** — lint + unit + integration, fails over 60 s |
| `make test` | Full pyramid with the branch-coverage gate |
| `make image` | Build the image, fail if it exceeds 500 MB (currently **435 MB**) |
| `make smoke` | Run the built image and exercise it over real HTTP |
| `make compose-up` / `make compose-down` | The stack, gated on real health |
| `make all` | Everything CI runs, in CI's order |

---

## What it decides

Six intents, each owned by a department:

| Intent | Department | Meaning |
|---|---|---|
| `complaint` | `quality_assurance` | شكوى عن منتج تالف أو خدمة سيئة |
| `price_inquiry` | `sales` | سؤال عن السعر أو الخصم أو التقسيط |
| `support_request` | `technical_support` | طلب مساعدة تقنية |
| `praise` | `customer_relations` | ثناء على المنتج أو الخدمة |
| `order_status` | `logistics` | سؤال عن حالة الطلب أو التوصيل |
| `return_refund` | `returns` | طلب إرجاع أو استبدال أو استرداد |

A seventh department, `safety`, exists but no intent maps to it. It is
reachable only through the escalation rule below — the model cannot route
anything there on its own, so a classifier error can never bury a real
emergency under ordinary traffic.

And one of three actions:

| Action | When | What the caller does |
|---|---|---|
| `auto_route` | confidence ≥ `0.60`, **or** an urgency term is present | Send it to `department` at `priority` |
| `human_review` | `0.25` ≤ confidence < `0.60` | An operator confirms the department first |
| `reject` | confidence < `0.25` | Return to the sender for clarification |

Those two thresholds were **measured, not chosen**: real messages score
0.87–0.99 and gibberish scores 0.20–0.34, so `0.60` and `0.25` sit inside the
empty gap between them. See [BENCHMARKS.md](BENCHMARKS.md).

### The safety rule

A message containing a term from a small, hand-curated list (`حريق`, `تسرب`,
`غاز`, `دخان`, `انفجار`, `اصابة`, …) is escalated to `urgent`, auto-routed,
and sent to the `safety` department — **overriding both the model's
confidence and its department**. A false escalation costs one wasted human
minute; a missed gas leak does not compare.

The department override is the part that was learned the hard way. An earlier
version escalated the priority but still took the department from the model,
which is fine until the model is wrong about an emergency — and that is
precisely when it is most likely to be. Observed in a live run:

```
"في تسرب غاز من السخان والرائحة قوية"
  intent:     praise          ← the model is badly wrong
  confidence: 0.370           ← and knows it is unsure
  priority:   urgent          ✅ escalated correctly
  department: customer_relations   ❌ urgently, to the wrong team
```

A message the classifier cannot read is exactly where its opinion is worth
least. So when the policy overrides the model, it now overrides it
completely. `tests/behavioural/test_directional.py` pins this against the real
model, including a case with praise-shaped wording wrapped around an
emergency.

The list is deliberately hand-written rather than learned, so it is reviewable
by a non-engineer and does not change silently when the model is retrained. The
scan runs on normalised text, so diacritics (`حَريق`), tatweel (`حــريق`) and
punctuation cannot defeat it — `tests/behavioural/test_invariance.py` pins
that, and `test_directional.py` asserts that **every** term in the list
actually escalates, so adding one is self-verifying.

---

## API

All routes are under `/v1`, so a breaking change can ship as `/v2` without
moving clients. Every response uses the same envelope.

| Route | Auth | Purpose |
|---|---|---|
| `POST /v1/predict` | API key | Decide on one message |
| `GET /v1/health` | none | **Liveness** — no I/O, no model access, always fast |
| `GET /v1/ready` | none | **Readiness** — 503 until the model is loaded *and* warmed |
| `GET /v1/policy` | API key | The live thresholds, departments and urgency terms |
| `GET /v1/metrics` | API key | Request, decision, error and cache-hit counters |

**Liveness vs readiness** is a real split, not two names for one check.
`/health` answers instantly even while the model is loading or after it has
failed to load — restarting on a failed liveness probe is only correct when the
*process* is broken. `/ready` returns 503 until a throwaway warm-up prediction
has actually returned, so a load balancer never hands a user the slow first
request.

### The envelope

Success and failure have the same shape:

```json
{"data": {...}, "error": null, "meta": {"trace_id": "..."}}
{"data": null, "error": {"code": "validation_error", "message": "...", "fields": ["body.txt"]}, "meta": {"trace_id": "..."}}
```

`error.code` is stable and machine-readable; `error.message` is for humans and
may be reworded. The trace id is also returned as the `X-Trace-Id` header and
written to every log line for that request.

| Status | `error.code` | Cause |
|---|---|---|
| 401 | `unauthorized` | Missing or invalid API key |
| 411 | `length_required` | Body without `Content-Length` |
| 413 | `payload_too_large` | Body over `INTENT_MAX_REQUEST_BYTES` |
| 422 | `validation_error` | Unknown field, empty text, or text over 2 000 chars |
| 429 | `rate_limited` | Over the per-caller budget (`Retry-After` set) |
| 500 | `internal_error` | Unhandled failure — details stay in the logs |
| 503 | `service_unavailable` | Model not loaded |

---

## Architecture

```
src/intent_service/
├── domain/      entities + policy   ← stdlib & pydantic only. No I/O.
├── service/     use-case + Protocol ports
├── adapters/    sklearn, redis      ← the only files that import them
└── api/         FastAPI, auth, middleware, envelope
config.py        one typed, fail-fast Settings
```

Dependencies point inward only. The service layer depends on `Protocol` ports
(`IntentModel`, `DecisionCache`), never on concrete adapters — which is why the
unit suite tests it with a six-line fake and no model artifact, and why
swapping sklearn for a hosted API means writing one adapter and changing one
line in the composition root.

**The contract is enforced by a tool, not by reviewer memory.**
`.importlinter` declares four contracts — the layering, the domain's purity,
the service's independence from adapters, and the confinement of sklearn/redis
to the adapter layer. `make lint` and CI both run `lint-imports`, so a
violation fails the pull request that introduces it.

`api/main.py` is the composition root and the only place concrete
implementations are named. There is deliberately no module-level
`app = create_app()`: building the app reads configuration, so a module-level
instance would do real work at import time and importing the module would fail
whenever the environment is incomplete. `create_app` is a factory, launched
with `--factory`.

---

## Tests

```bash
make gate    # 6.1 s — lint, types, contract, unit + integration
make test    # 11.3 s — everything, with the branch-coverage gate
```

| Layer | Tests | Asks |
|---|---|---|
| **Unit** | 72 | Is each piece correct in isolation? No model, no HTTP. |
| **Integration** | 35 | Is the wiring right? Real model, real middleware, via TestClient. |
| **Behavioural** | 94 | Does the *system* behave? Against the real trained artifact. |

The behavioural layer is four files, each pinning a different kind of claim:

- **`test_invariance.py`** — meaning-preserving edits (whitespace,
  punctuation, diacritics, tatweel) must not change the decision.
- **`test_directional.py`** — adding an urgency term can only raise priority,
  never lower it, and can never turn an actionable message into a rejection.
  Also that higher confidence never *reduces* automation.
- **`test_golden.py`** — 18 fixed messages, with their approved decisions
  recorded in `golden_decisions.json`.
- **`test_model_quality.py`** — six hand-written messages that appear nowhere
  in the generator, so a pass means generalisation rather than memorisation.

**Branch** coverage is **95.5%** against an 80% requirement; the domain and
service layers are at 100%. Branch rather than line, because a policy made of
`if`/`elif` bands can hit every line while never exercising the `human_review`
or `reject` paths.

### If a golden test fails

It is a **behaviour change, not a broken test**. Read the diff, decide whether
the new behaviour is better, and fix the code if it is not. If it genuinely is
better:

```bash
python -m scripts.regenerate_golden \
  --reviewed-by "Your Name" --reason "why this behaviour should change"
```

The generator refuses to run without both flags, prints a full diff of every
decision it would change, requires interactive confirmation, and writes the
reviewer and reason into the file — so `git blame` names a person and an
unexplained regeneration is visible in review rather than buried in a green
tick.

---

## CI/CD

Six stages, in `.github/workflows/ci.yml`:

1. **Fast gate** — lint, format, strict mypy, import-linter, unit + integration.
   Budget-checked at 60 s, with `timeout-minutes: 2` as a backstop.
2. **Secret scan** — gitleaks over the **full history** (`fetch-depth: 0`). A
   key removed in a later commit is still a leaked key.
3. **Dependency audit** — `pip-audit --strict`.
4. **Train & test** — trains from source, re-verifies that the seeded generator
   is byte-identical across runs, then all three layers plus the coverage gate.
5. **Image** — builds, fails over 500 MB, runs `scripts/smoke.sh` against the
   real container, and brings the compose stack up to confirm Redis attaches.
6. **Publish** — GHCR, **only** on push to `main`, tagged by commit SHA.

### Branch protection on `main`

Configure under *Settings → Branches → Add rule* for `main`:

- ☑ Require a pull request before merging — **1 approval**
- ☑ Require status checks to pass: `Fast gate (< 60s)`,
  `Secret scan (full history)`, `Dependency audit`,
  `Train & full test pyramid`, `Build image, check size & smoke test`
- ☑ Require branches to be up to date before merging
- ☑ Require conversation resolution before merging
- ☐ Allow force pushes — **disabled**
- ☐ Allow deletions — **disabled**

---

## Configuration

Every variable is read in exactly one place (`src/intent_service/config.py`),
typed, and validated at startup. A misconfigured service **refuses to boot**
with a message naming the variable, rather than booting and failing every
request. Two guards in particular:

- `INTENT_REQUIRE_API_KEY=true` with an empty `INTENT_API_KEYS` is refused —
  otherwise every call 401s and the outage looks like a code bug.
- `INTENT_REJECT_FLOOR` above `INTENT_AUTO_ROUTE_FLOOR` is refused — the
  `human_review` band would be empty and nothing would ever reach a person,
  a safety regression no status code would reveal.

See [.env.example](.env.example) for the full annotated list. Defaults are the
safe ones: auth required, CORS closed, docs off, proxy headers untrusted.

**Logs** are JSON on stdout, correlated by `trace_id`. The formatter copies
only allowlisted fields out of each record, so message text is never logged at
any level — a caller quotes their trace id in a support ticket instead of
pasting their message into one.

---

## Known limitations

Stated rather than hidden — each is a real constraint of this build.

- **The rate limiter is in-process.** Each replica enforces its own budget, so
  N replicas allow N× the configured rate. A shared limiter belongs in the
  Redis that is already in the stack; it is not implemented.
- **The training data is synthetic.** It is generated, seeded, and
  reproducible, but it is not real customer traffic. The test-split accuracy of
  1.000 measures the generator's consistency, not the model's generalisation —
  which is exactly why the hand-written probes in `test_model_quality.py`
  exist and why BENCHMARKS.md reports both.
- **Urgency detection is exact-token matching** on a curated list. It will not
  catch a paraphrase ("الدنيا اشتعلت"). That is the deliberate trade for a
  safety rule a non-engineer can audit and that cannot shift under a retrain.
- **API keys are compared against a plaintext env list.** Fine for this scope;
  a real deployment wants hashed keys with rotation and per-key scopes.
- **`docs/` is off by default.** Set `INTENT_ENABLE_DOCS=true` to browse the
  OpenAPI schema locally.

---

## Further reading

- [DECISIONS.md](DECISIONS.md) — five engineering decisions, each with the
  alternative rejected and what it cost
- [BENCHMARKS.md](BENCHMARKS.md) — real measurements: 435 MB image, 186×
  cache speed-up, 14 s fast gate, 96% branch coverage, confidence separation
- [SECURITY.md](SECURITY.md) — threat model and the controls in place
- [DOCKER.md](DOCKER.md) — container internals

## Acknowledgement

This project was completed as part of the **SDA-AIE-113 — Software Engineering
Practices for AI Systems** training program at **SDAIA Academy**, under the
supervision of **Abdullah Khalid AlShahrani**.

Official SDAIA Academy GitHub: https://github.com/SDAIAAcademy

## Licence

**Proprietary — all rights reserved.** See [LICENSE](LICENSE).

This repository is coursework, and the licence is written for that: no general
permission to use, copy, modify or redistribute is granted, and the code is
public so that it can be read and assessed, not reused.

The licence does carry one explicit grant. Instructors and authorised
examiners of SDA-AIE-113 have a limited, non-transferable right to access,
run and evaluate this software for the purpose of assessment — so everything
in this README can be executed by a reviewer without asking permission
first.
