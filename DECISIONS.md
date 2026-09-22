# Engineering decisions

Five decisions that shaped this service, each with the alternative that was
rejected and the reason. A decision without a stated cost is a preference, so
every entry names what it gave up.

---

## 1. The decision lives in the domain, not in the model

**Decision.** The model returns a label and a calibrated probability. It does
not decide anything. A pure function in `domain/policy.py` turns that output
into one of three actions — `auto_route`, `human_review`, `reject` — plus a
department, a priority, and a written reason.

**Alternative rejected.** Let the classifier's label *be* the answer, and have
the API return it. This is what the service did before: `POST /v1/predict`
answered `{"intent": "complaint", "confidence": 0.31}` and left the caller to
decide what that meant.

**Why.** A confidence number is not an operational instruction. Every consumer
would have had to invent its own threshold, and those thresholds would have
drifted apart silently — one team automating at 0.3, another at 0.7, with no
way to audit either. Worse, safety rules would have had nowhere to live: the
requirement that a message mentioning a gas leak is escalated *regardless of
what the model thinks* cannot be expressed as a confidence threshold at all.

Putting the policy in the domain also made the behavioural suite possible.
`decide()` is total, deterministic, and free of I/O, so invariance and
directional tests can call it thousands of times without a model, a socket, or
a clock. `tests/unit/test_policy.py` pins both threshold boundaries from both
sides — the kind of off-by-one that is invisible in production until an auditor
asks why a 0.60 message went to a human.

**Cost.** Two places to look when a decision surprises you, and a policy that
can drift out of step with the model it thresholds. Mitigated by
`GET /v1/policy`, which reports the live thresholds of a running instance, and
by folding both into the cache key (see #5).

---

## 2. Word + character features, because character-only over-fit the generator

**Decision.** The vectoriser is a `FeatureUnion` of word 1–2 grams and
`char_wb` 3–5 grams.

**Alternative rejected.** Character n-grams alone — the conventional choice for
Arabic, and what this project used first. Arabic morphology glues prefixes and
suffixes onto stems (`ال-`, `و-`, `-ها`, `-كم`), so sub-word features are
genuinely useful and word features alone miss a lot.

**Why the change.** The char-only model scored **1.000 on its own test split**
and **2/6 on six hand-written held-out messages**. It routed `الخدمة سيئة جدا`
(*your service is very bad*) to `praise`. It had learned the generator's
sentence skeletons rather than its vocabulary, and the test split — drawn from
the same generator — could not see that.

The fix was two-sided. The feature union restored the content words, and
`train/generate_dataset.py` was rewritten to compose messages as
`[opener] + core + [closer]`, where openers and closers are drawn from pools
**shared by every intent**. A model that tries to key on the polite framing
gets no label signal from it, so it is forced onto the content. Held-out probe
accuracy went from 2/6 to **6/6**.

**Cost.** Roughly double the feature count and a slightly larger artifact
(0.61 MB). At 1.13 ms per inference this is not worth optimising.

**The lesson worth keeping.** A test-set score computed on synthetic data
measures the generator, not the model. The hand-written probes are the real
gate, and they are hand-written precisely so no generator can flatter them.

---

## 3. LogisticRegression over LinearSVC, because the policy thresholds a number

**Decision.** The classifier is `LogisticRegression`, and the adapter returns
`predict_proba`.

**Alternative rejected.** `LinearSVC`, which was faster to fit and marginally
more accurate on this data.

**Why.** `LinearSVC` exposes only `decision_function` margins. The adapter was
squashing those through a softmax to produce a bounded number — monotonic, but
meaningless in absolute terms. Measured, it compressed every input into
0.18–0.49, with real messages and gibberish overlapping. A threshold of 0.45
against that scale is arbitrary: there is no answer to "why 0.45" beyond "it
seemed to work".

With a likelihood-fit model, `predict_proba` is a real probability. Measured on
the same inputs, real messages land at 0.87–0.99 and noise at 0.20–0.34 — an
empty gap, and `auto_route_floor = 0.60` sits inside it. That is a threshold
with an answer.

The adapter also makes one `predict_proba` call and takes the label from the
argmax, rather than calling `predict` and `predict_proba` separately. Two calls
would run feature extraction twice and could in principle disagree, handing the
policy a confidence belonging to a different label than the one returned.
`tests/unit/test_sklearn_model.py` pins that they agree.

**Cost.** Slower to fit (3.3 s vs ~1 s) and a hair less accurate on the
synthetic split. Both are irrelevant next to having interpretable thresholds.

---

## 4. One response envelope, for successes and failures alike

**Decision.** Every response — 200, 401, 422, 429, 500, 503 — has the same
three keys: `{"data": …, "error": …, "meta": {"trace_id": …}}`, with `data`
xor `error` populated. Three exception handlers enforce it, including one
registered against Starlette's base `HTTPException` so unmatched-route 404s
cannot escape it.

**Alternative rejected.** FastAPI's default: return the payload at the top
level on success, and `{"detail": …}` on failure.

**Why.** Under the default, a client must branch on the status code before it
can find anything — including the trace id, which is exactly what it needs when
something has gone wrong. Two shapes means two parsers, and the error parser is
the one that gets written last and tested least. The envelope also gives
`error.code` a stable machine-readable string separate from `error.message`,
so the prose can be improved without breaking integrations.

The trace id is generated in middleware and stashed on `request.state` before
anything can fail, so the success path, every handler, the `X-Trace-Id` header,
and every log line quote the same value.

**Cost.** One level of nesting on every response, and a hand-written envelope
rather than the framework's default. The integration suite parametrises the
same assertion across nine endpoint/status combinations to keep it honest.

---

## 5. The Redis decision cache degrades instead of failing

**Decision.** The extension is a decision cache behind a `DecisionCache`
Protocol, with a Redis adapter and an in-process fallback. Every Redis method
swallows `RedisError` and reports a miss; both socket timeouts are set to
250 ms; Redis is the compose stack's supporting service, gated on a real
healthcheck.

**Alternative rejected.** Call Redis directly from the API layer and let
failures propagate.

**Why.** Support traffic is extremely repetitive, and at 1.13 ms per inference
every duplicate is waste — measured at **186× faster** on a hit. But adding a
network dependency to a request path is normally a downgrade in availability,
and that trade is only acceptable if the dependency cannot take the service
down. Hence the contract: a Redis outage makes the service slower, never
unavailable. `tests/unit/test_redis_cache.py` asserts every method's behaviour
against a client that raises on every call. `/v1/ready` reports
`cache_healthy` but deliberately does **not** fail readiness on it — failing
readiness over a cache outage would take down a service perfectly able to
serve.

The two socket timeouts are the single most important lines in the adapter.
Without them, a hung Redis — not a dead one — would stall every prediction, and
the optional dependency would become a total outage.

The cache key is a SHA-256 of `model_version | thresholds | text`. Hashing
keeps customer text out of backend monitoring tools and bounds the key length;
folding in the version and thresholds means a retrain or a policy change
invalidates every entry automatically. Serving a decision made under
superseded rules would be a silent correctness bug, and nothing else would
catch it.

**Cost.** A second container in compose, a `redis` dependency, and a cached
payload that can outlive the schema that wrote it — handled by validating on
read and recomputing if it no longer parses, rather than 500-ing.

---

## Appendix: two rules that were never in question

**No `:latest` tag.** Images are published to GHCR addressed only by commit
SHA. `latest` is mutable: two deployments of it a week apart are different
software with no record of which is running, and a rollback has nothing to roll
back to.

**The golden file is never regenerated to make a test pass.** A red golden test
is a behaviour change, not a broken test. `scripts/regenerate_golden.py`
refuses to run without `--reviewed-by` and `--reason`, prints a full diff of
every decision it would change, and writes the reviewer and reason into the
file — so `git blame` names a person, and an unexplained regeneration is
visible in review rather than buried in a green tick.
