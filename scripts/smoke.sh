#!/usr/bin/env bash
# Smoke test: start the built image and prove it serves real HTTP.
#
# This is the gate that catches "works on my machine". A green unit suite
# means nothing if the image does not start, does not become ready, or 200s a
# request that should be 401. Everything here goes over the network to a
# container -- no TestClient, no in-process shortcuts.
#
# Usage: scripts/smoke.sh <image:tag> <port> <api-key>
set -euo pipefail

IMAGE="${1:?usage: smoke.sh <image:tag> <port> <api-key>}"
PORT="${2:?}"
KEY="${3:?}"
NAME="intent-smoke-$$"

cleanup() {
  docker logs "$NAME" 2>&1 | tail -30 || true
  docker rm -f "$NAME" >/dev/null 2>&1 || true
}
trap cleanup EXIT

fail() { echo "SMOKE FAIL: $*" >&2; exit 1; }

echo "--> starting $IMAGE"
docker run -d --name "$NAME" -p "${PORT}:8000" \
  -e INTENT_API_KEYS="$KEY" \
  -e INTENT_LOG_LEVEL=INFO \
  "$IMAGE" >/dev/null

echo "--> waiting for /v1/ready"
for _ in $(seq 1 45); do
  if curl -sf "http://127.0.0.1:${PORT}/v1/ready" | grep -q '"ready":true'; then
    ready=1; break
  fi
  sleep 1
done
[ "${ready:-0}" = 1 ] || fail "service never became ready"

echo "--> 1. valid request is auto-routed"
body=$(curl -sf -X POST "http://127.0.0.1:${PORT}/v1/predict" \
  -H 'Content-Type: application/json' -H "X-API-Key: ${KEY}" \
  -d '{"text":"وصلني الجهاز مكسور ومب شغال نهائيا وابي تعويض"}')
echo "$body"
echo "$body" | grep -q '"department":"quality_assurance"' || fail "wrong department"
echo "$body" | grep -q '"trace_id"' || fail "envelope is missing meta.trace_id"

echo "--> 2. an urgency term escalates priority"
body=$(curl -sf -X POST "http://127.0.0.1:${PORT}/v1/predict" \
  -H 'Content-Type: application/json' -H "X-API-Key: ${KEY}" \
  -d '{"text":"في تسرب غاز من السخان وريحة قوية في البيت"}')
echo "$body"
echo "$body" | grep -q '"priority":"urgent"' || fail "urgency was not escalated"

echo "--> 3. a malformed body is rejected inside the envelope"
code=$(curl -s -o /tmp/smoke_bad.json -w '%{http_code}' -X POST \
  "http://127.0.0.1:${PORT}/v1/predict" \
  -H 'Content-Type: application/json' -H "X-API-Key: ${KEY}" \
  -d '{"txt":"unknown field"}')
cat /tmp/smoke_bad.json; echo
[ "$code" = "422" ] || fail "expected 422 for an unknown field, got $code"
grep -q '"code":"validation_error"' /tmp/smoke_bad.json || fail "missing error.code"
grep -q '"trace_id"' /tmp/smoke_bad.json || fail "error envelope lost the trace id"

echo "--> 4. an unauthenticated request is rejected"
code=$(curl -s -o /dev/null -w '%{http_code}' -X POST \
  "http://127.0.0.1:${PORT}/v1/predict" \
  -H 'Content-Type: application/json' -d '{"text":"مرحبا"}')
[ "$code" = "401" ] || fail "expected 401 without a key, got $code"

echo "--> 5. the container runs as a non-root user"
whoami_out=$(docker exec "$NAME" id -u)
[ "$whoami_out" != "0" ] || fail "container is running as root"

echo "--> 6. SIGTERM produces a clean, prompt shutdown"
start=$(date +%s)
docker stop -t 15 "$NAME" >/dev/null
elapsed=$(( $(date +%s) - start ))
rc=$(docker inspect "$NAME" --format '{{.State.ExitCode}}')
echo "stopped in ${elapsed}s with exit code ${rc}"
# 143 = 128 + SIGTERM. A 0 is fine too; anything else means the process was
# SIGKILLed after ignoring the stop signal, which loses in-flight requests.
[ "$rc" = "0" ] || [ "$rc" = "143" ] || fail "unclean shutdown, exit code $rc"
[ "$elapsed" -lt 15 ] || fail "container ignored SIGTERM and had to be killed"

echo "SMOKE PASS"
