# Security Policy

## Reporting a vulnerability

Report suspected vulnerabilities privately to the repository owner. Do not
open a public issue describing an exploitable flaw.

## Controls implemented

| Control | Where | Default |
|---|---|---|
| API key authentication (`X-API-Key`) | `api/security.py` | Required on `/predict`, `/model-info`, `/metrics` |
| Constant-time key comparison | `security.is_authorised` | Always — prevents timing attacks |
| Per-caller rate limiting | `security.SlidingWindowRateLimiter` | 60 requests / 60 seconds |
| Request body size cap | `api/main.py` middleware | 16 KiB → `413` |
| Input length validation | `domain/entities.py` | 1–2000 characters → `422` |
| CORS | `api/main.py` | **Deny all** unless origins configured |
| Security response headers | `security.SECURITY_HEADERS` | Applied to every response, including errors |
| Non-root container user | `Dockerfile` | `appuser` (uid 1000) |
| Secrets kept out of the repo | `.gitignore`, `.env.example` | `.env` never committed |
| Fail-fast misconfiguration | `config.Settings` validator | Refuses to start if auth required but no keys set |

## Design decisions

**`/health` is intentionally unauthenticated and un-rate-limited.** Liveness
probes do not carry credentials, and the endpoint exposes only whether the
model artifact loaded — no business data, no model internals.

**Authentication runs before the model.** The API key dependency is evaluated
before any inference, so an unauthenticated caller can never spend compute.
This is covered by `test_auth_runs_before_the_model`.

**Authentication errors are deliberately vague.** A missing key and a wrong key
return an identical `401` body. Distinguishing them tells an attacker whether a
key exists.

**Internal errors never leak details.** A failed prediction returns a generic
`500`; the traceback goes to the logs only.

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
