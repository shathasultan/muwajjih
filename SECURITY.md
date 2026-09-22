# Security Policy

## Reporting a vulnerability

Report suspected vulnerabilities privately to the repository owner. Do not
open a public issue describing an exploitable flaw.

## Controls implemented

| Control | Where | Default |
|---|---|---|
| API key authentication (`X-API-Key`) | `api/security.py` | Required on `/v1/predict`, `/v1/policy`, `/v1/metrics` |
| Constant-time key comparison | `security.is_authorised` | Always — prevents timing attacks |
| Per-caller rate limiting | `security.SlidingWindowRateLimiter` | 60 requests / 60 seconds |
| Request body size cap | `api/main.py` middleware | 16 KiB → `413` |
| Input length validation | `domain/entities.py`, `api/schemas.py` | 1–2000 characters, unknown fields rejected → `422` |
| CORS | `api/main.py` | **Deny all** unless origins configured |
| Security response headers | `security.SECURITY_HEADERS` | Applied to every response, including errors |
| Non-root container user | `Dockerfile` | `appuser` (uid 1000) |
| Secrets kept out of the repo | `.gitignore`, `.env.example` | `.env` never committed |
| Fail-fast misconfiguration | `config.Settings` validators | Refuses to start on missing keys or inverted policy thresholds |
| Secret scanning | `gitleaks`, pre-commit + CI | Full git history scanned on every push |
| Dependency vulnerability audit | `pip-audit --strict` in CI | Fails the build on a known CVE |
| No customer data in logs | `api/logging_config.py` | Field allowlist — message text is never logged |
| Cache keys are hashed | `service/triage.py` | SHA-256, so text never reaches backend tooling |

## Design decisions

**`/v1/health` and `/v1/ready` are intentionally unauthenticated and
un-rate-limited.** Orchestrator probes do not carry credentials, and
rate-limiting a probe turns a healthy instance into a falsely-unhealthy one
that gets restarted. They expose only load and warm-up state — no business
data, no model internals.

**Authentication runs before the model.** The API key dependency is evaluated
before any inference, so an unauthenticated caller can never spend compute.
This is covered by `test_auth_runs_before_the_model`.

**Authentication errors are deliberately vague.** A missing key and a wrong key
return an identical `401` body. Distinguishing them tells an attacker whether a
key exists.

**Internal errors never leak details.** A failed prediction returns a generic
`500` inside the standard envelope; the traceback goes to the logs only,
correlated by the `trace_id` that *is* returned — so a caller can be helped
without the service telling them anything about its internals.

**Message text is never written to a log line.** The JSON formatter copies only
an allowlist of fields off each record, so a future `extra={"text": ...}`
silently drops the customer data rather than shipping it to a log aggregator.
The `trace_id` handed back to the caller is what makes this workable: a user
quotes the id in a support ticket instead of pasting their message into one.

## Known limitations

These are real and deliberately documented rather than hidden:

1. **Rate limiting is in-process.** Each replica keeps its own counters, so N
   replicas allow roughly N× the configured budget. A multi-replica deployment
   needs a shared store (Redis) or a gateway-level limiter.
2. **API keys are static, shared secrets.** There is no rotation, expiry, or
   per-key scope. A production system should move to short-lived tokens
   (OAuth2 / JWT) with per-client identity.
3. **No TLS at the application layer.** The service speaks plain HTTP and
   expects to sit behind a TLS-terminating reverse proxy or load balancer.
   `Strict-Transport-Security` is therefore not set here — it belongs on the
   proxy.
4. **No audit logging.** Requests are counted, not recorded. A regulated
   deployment would need per-request audit trails with retention rules.
5. **Keys are compared against an in-memory list from the environment.** There
   is no secret manager integration (Vault, AWS Secrets Manager).

## Dependency management

Dependencies are declared with minimum-version bounds in `pyproject.toml`.
`pre-commit` and CI run `ruff` and `mypy --strict` on every change. Adding
`pip-audit` to the CI pipeline is the recommended next hardening step.
