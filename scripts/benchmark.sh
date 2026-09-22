#!/usr/bin/env bash
# Produce the numbers in BENCHMARKS.md. Every figure there comes from this
# script -- nothing is estimated, and re-running it is how the table is kept
# honest after a change.
#
# Usage: ./scripts/benchmark.sh
set -uo pipefail
cd "$(dirname "$0")/.."

echo "=== environment ==="
python -c "import platform,sys; print('python', sys.version.split()[0]); print('platform', platform.platform())"
python -c "import sklearn; print('scikit-learn', sklearn.__version__)"
nproc 2>/dev/null | sed 's/^/cpus /' || true

echo; echo "=== dataset generation ==="
start=$(date +%s.%N)
python -m train.generate_dataset >/dev/null
printf "generate_dataset: %.1f s\n" "$(echo "$(date +%s.%N) - $start" | bc)"

echo; echo "=== training ==="
start=$(date +%s.%N)
python -m train.train_model >/dev/null
printf "train_model: %.1f s\n" "$(echo "$(date +%s.%N) - $start" | bc)"
python -c "import json;m=json.load(open('models/metrics.json'));print('test_accuracy', m['test_accuracy']);print('test_macro_f1', m['test_macro_f1'])"
ls -l models/intent_model.joblib | awk '{printf \
  "artifact size: %.2f MB\n", $5/1000000}'

echo; echo "=== test suites ==="
for suite in unit integration behavioural; do
  start=$(date +%s.%N)
  pytest "tests/$suite" -q --no-header >/dev/null 2>&1
  printf "%-13s %.1f s\n" "$suite:" "$(echo "$(date +%s.%N) - $start" | bc)"
done
start=$(date +%s.%N)
pytest tests -q --no-header --cov >/dev/null 2>&1
printf "%-13s %.1f s\n" "full+cov:" "$(echo "$(date +%s.%N) - $start" | bc)"

echo; echo "=== fast gate (budget: 60s) ==="
start=$(date +%s.%N)
ruff check src tests train scripts >/dev/null \
  && ruff format --check src tests train scripts >/dev/null \
  && mypy src >/dev/null \
  && lint-imports >/dev/null \
  && pytest tests/unit tests/integration -q --no-header >/dev/null 2>&1
printf "fast gate: %.1f s\n" "$(echo "$(date +%s.%N) - $start" | bc)"

echo; echo "=== inference latency & cache effect ==="
python - <<'PY'
import statistics, time
from intent_service.adapters.sklearn_model import SklearnIntentModel
from intent_service.adapters.redis_cache import InMemoryDecisionCache
from intent_service.domain.entities import CustomerMessage
from intent_service.domain.policy import PolicyThresholds
from intent_service.service.triage import TriageService

model = SklearnIntentModel.load("models/intent_model.joblib")
thresholds = PolicyThresholds()
texts = [f"وين طلبي رقم {i}؟ صار له اسبوع" for i in range(200)]

cold = TriageService(model=model, thresholds=thresholds, cache=None)
cold.triage(CustomerMessage(text="تهيئة"))  # warm-up

def bench(service, texts):
    samples = []
    for t in texts:
        s = time.perf_counter()
        service.triage(CustomerMessage(text=t))
        samples.append((time.perf_counter() - s) * 1000)
    return samples

uncached = bench(cold, texts)
warm = TriageService(model=model, thresholds=thresholds, cache=InMemoryDecisionCache(4096))
bench(warm, texts)              # populate
cached = bench(warm, texts)     # all hits

for name, s in (("uncached", uncached), ("cached", cached)):
    s_sorted = sorted(s)
    print(f"{name:9s} mean {statistics.mean(s):6.2f} ms   "
          f"p50 {s_sorted[len(s)//2]:6.2f} ms   "
          f"p95 {s_sorted[int(len(s)*0.95)]:6.2f} ms")
print(f"speed-up: {statistics.mean(uncached)/statistics.mean(cached):.1f}x")
PY

echo; echo "=== container (needs a working docker daemon) ==="
if docker info >/dev/null 2>&1; then
  start=$(date +%s)
  docker build -t intent-service:bench . >/dev/null 2>&1 \
    && echo "cold build: $(( $(date +%s) - start )) s" \
    || { echo "build FAILED"; exit 1; }
  start=$(date +%s)
  docker build -t intent-service:bench . >/dev/null 2>&1
  echo "cached build: $(( $(date +%s) - start )) s"
  docker image inspect intent-service:bench --format '{{.Size}}' \
    | awk '{printf "image size: %.1f MB (budget 500 MB)\n", $1/1000000}'
  start=$(date +%s.%N)
  ./scripts/smoke.sh intent-service:bench 8099 bench-key >/dev/null 2>&1 \
    && printf "smoke test: %.1f s\n" "$(echo "$(date +%s.%N) - $start" | bc)" \
    || echo "smoke FAILED"
else
  echo "SKIPPED: no docker daemon reachable"
fi
