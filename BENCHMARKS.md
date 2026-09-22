# Benchmarks

Every number below comes from `./scripts/benchmark.sh` on the machine described
in *Environment*. Nothing here is estimated. Re-run the script after any change
that could plausibly move a figure — a stale benchmarks file is worse than none,
because it is trusted.

Two rows are marked **PENDING**. They require a Docker daemon with registry
access, which the environment these were captured in does not have (its egress
policy blocks Docker Hub blob downloads). `make image` and `make smoke` produce
them, and CI asserts the budget on every push regardless — see
*Container* below.

## Environment

| | |
|---|---|
| Python | 3.12.3 |
| Platform | Linux 6.18.44 x86_64, glibc 2.39 |
| scikit-learn | 1.9.1 |
| CPUs | 4 |
| Captured | 2026-09-22 |

## Training pipeline

Both steps run from source on every CI push — no binary artifact is committed,
so CI re-proves reproducibility rather than trusting it.

| Step | Time |
|---|---|
| `generate_dataset` (3 000 examples, seed=42) | 0.1 s |
| `train_model` (word+char TF-IDF → LogisticRegression) | 3.3 s |
| **Total, cold, from an empty checkout** | **3.4 s** |

| Artifact | Size |
|---|---|
| `models/intent_model.joblib` | 0.61 MB |

## Model quality

| Metric | Value |
|---|---|
| Test-split accuracy (450 held-out examples) | 1.000 |
| Test-split macro F1 | 1.000 |
| **Hand-written held-out probes** | **6 / 6** |

The middle two rows are the ones not to trust, and the project is explicit
about why. The test split is drawn from the same generator as the training set,
so a perfect score on it measures the *generator's* consistency, not the
model's generalisation. The row that carries real information is the last one:
six messages written by hand, in phrasing that appears nowhere in
`train/generate_dataset.py`, checked by
`tests/behavioural/test_model_quality.py`.

That distinction is not academic. The previous model (char-only TF-IDF →
LinearSVC) also scored 1.000 on its test split, and got **2 / 6** on those
probes — it had learned the generator's sentence skeletons. The current
figures come from fixing that; see DECISIONS.md #2 and #3.

### Confidence separation

The property the decision thresholds depend on, measured on the probes above
versus deliberate noise (`؟؟؟`, `ووووووو`, `xyz abc 123`, `٪٪٪٪`):

| Input class | Confidence range |
|---|---|
| Real messages | 0.866 – 0.988 |
| Noise / gibberish | 0.197 – 0.336 |

`auto_route_floor = 0.60` and `reject_floor = 0.25` sit inside that empty gap.
They were chosen from this measurement, not picked as round numbers — which is
what makes them defensible when a reviewer asks why 0.60.

## Latency

200 distinct messages, single process, model warm. Measured at the service
layer, so this is inference plus policy, excluding HTTP framing.

| | mean | p50 | p95 |
|---|---|---|---|
| Uncached | 1.13 ms | 1.01 ms | 1.55 ms |
| Cached (decision cache hit) | 0.01 ms | 0.01 ms | 0.01 ms |

**Speed-up on a cache hit: 186×.**

This is the number that justifies the extension. Support queues are extremely
repetitive — the same boilerplate sentence arrives hundreds of times a day —
and at 1.13 ms per inference, a duplicate is pure waste. The cache is
in-process here; against Redis the hit path adds a network round trip, so the
realistic saving is smaller but the shape holds. Either backend is a
best-effort optimisation: every method swallows backend errors and reports a
miss, so a Redis outage makes the service slower, never unavailable
(`tests/unit/test_redis_cache.py` pins that contract).

## Test suite

| Suite | Tests | Time |
|---|---|---|
| Unit | 72 | 5.7 s |
| Integration | 35 | 3.1 s |
| Behavioural | 94 | 2.3 s |
| **Full suite with branch coverage** | **201** | **11.3 s** |

| Gate | Measured | Budget |
|---|---|---|
| **Fast gate** (lint + format + mypy + import-linter + unit + integration) | **6.1 s** | 60 s |

The fast gate has 10× headroom. That is deliberate: a budget met at 55 s is one
commit away from failing, and a gate people expect to fail stops being a gate.
`make gate` fails if it ever crosses 60 s, and the CI job carries
`timeout-minutes: 2` as a second line of defence.

Coverage is applied to `make test`, not `make gate` — instrumenting every line
roughly doubles the suite's wall time, and the fast gate's whole purpose is to
stay fast.

## Coverage

Branch coverage, not line coverage: a policy built from `if`/`elif` bands can
reach 100% of lines while never exercising the `human_review` or `reject`
paths.

| Module | Branch coverage |
|---|---|
| `domain/policy.py` | 100% |
| `domain/entities.py` | 100% |
| `service/triage.py` | 100% |
| `adapters/sklearn_model.py` | 100% |
| `adapters/redis_cache.py` | 100% |
| `api/main.py` | 94% |
| `api/security.py` | 87% |
| **Total** | **95.5%** |

Requirement: ≥ 80% on the core layers. The domain and service layers — where a
bug is a wrong decision shipped to a customer rather than a 500 — are at 100%.

## Container

| Metric | Value |
|---|---|
| Image size | **PENDING** — run `make image` |
| Cold build | **PENDING** — run `make image` |
| Base image | `python:3.12-slim`, multi-stage |
| Runs as | non-root (`appuser`, uid 1000) |
| Healthcheck | `GET /v1/ready` |
| Shutdown | SIGTERM → exit 0/143, asserted by `scripts/smoke.sh` |

To fill in the two pending rows:

```bash
make image    # builds, prints the measured size, fails over 500 MB
make smoke    # runs the image and exercises it over real HTTP
```

The budget is enforced independently of this file: the CI `image` job fails the
build if the image exceeds 500 MB and writes the measured size into the run
summary, so the number is verified on every push whether or not anyone updates
the table.

The runtime stage is expected to land well under budget — it carries neither
pandas nor the training code nor the dev dependencies, only the fitted
artifact crossing the stage boundary — but "expected" is not a measurement, so
the row stays PENDING until somebody runs the command.
