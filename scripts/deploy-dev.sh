#!/usr/bin/env bash
# Self-verifying dev deploy for the Squirrel LLM gateway image.
#
# Builds the gateway image from origin/dev (a clean worktree, never the dirty
# checkout), pushes it to Nexus as the mutable `dev-latest` tag plus an
# immutable `dev-<sha>` tag, triggers the inference-network Coolify app to
# re-pull + recreate the stack, polls the deployment to `finished`, and
# smoke-tests two providers so a recreate that breaks provider-key decryption
# is caught immediately.
#
# Mirrors the repo-owned `deploy:*` convention used by the back-end / front-end /
# researchly publish pipelines (build origin/<branch> -> push Nexus -> deploy ->
# verify). The compose pin lives once at `:dev-latest`; this script never edits
# the inference-network repo. Topology + secrets are documented in the
# inference-network repo: docs/coolify-source-mapping.md.
#
# Usage:  make deploy-dev   (from the llm-gateway repo root)
set -euo pipefail

REGISTRY="docker.iocloudhost.net/researchly/llm-gateway"
BUILDER="ci-remote"                       # native linux/amd64 builder on popos-sf4
APP_UUID="bdtgxpkb79qusq5f7szgl5xw"       # squirrel-llm-gateway-with-queue (inference-network compose)
COOLIFY="https://coolify.iocloudhost.net"
WORKTREE="$(mktemp -d -t llm-gateway-deploy-XXXX)"

cleanup() { git worktree remove --force "$WORKTREE" 2>/dev/null || rm -rf "$WORKTREE"; }
trap cleanup EXIT

# 1. Resolve origin/dev and check out a clean worktree of exactly that commit.
git fetch -q origin dev
SHA="$(git rev-parse --short origin/dev)"
echo "==> gateway deploy: building origin/dev = $SHA"
git worktree add -q --detach "$WORKTREE" "origin/dev"

# 2. Build native amd64 and push both the mutable and immutable tags to Nexus.
docker buildx build --builder "$BUILDER" --platform linux/amd64 \
  -t "${REGISTRY}:dev-latest" -t "${REGISTRY}:dev-${SHA}" \
  --push "$WORKTREE"
echo "==> pushed ${REGISTRY}:dev-latest and :dev-${SHA}"

# 3. Trigger the Coolify deploy (force => re-pull the mutable dev-latest tag).
CK="$(doppler run -p personal -c dev -- printenv COOLIFY_API_KEY)"
curl -fsS "${COOLIFY}/api/v1/deploy?uuid=${APP_UUID}&force=true" \
  -H "Authorization: Bearer ${CK}" >/dev/null
echo "==> Coolify deploy queued (force re-pull)"

# 4. Poll the deployment to a terminal state.
for _ in $(seq 1 40); do
  ST="$(curl -fsS "${COOLIFY}/api/v1/deployments/applications/${APP_UUID}" \
        -H "Authorization: Bearer ${CK}" \
        | python3 -c 'import sys,json;print(json.load(sys.stdin)["deployments"][0]["status"])')"
  echo "    deploy: ${ST}"
  case "$ST" in
    finished) break ;;
    failed|error) echo "!! Coolify deploy ${ST}"; exit 1 ;;
  esac
  sleep 15
done

# 5. Smoke test: a real chat call per protocol proves the image is live AND the
#    provider keys still decrypt after the Squirrel recreate. max_completion_tokens
#    must be >= 16 (the OpenAI Responses API rejects anything lower).
KEY="$(doppler run -p back-end -c staging -- printenv LLM_API_KEY)"
KEY="$KEY" python3 - "$SHA" <<'PY'
import json, os, sys, urllib.request, urllib.error
key = os.environ["KEY"]; sha = sys.argv[1]
url = "https://api.llm-gateway.iocloudhost.net/v1/chat/completions"
ok = True
for model in ("gpt-5.4", "claude-opus-4-7"):
    body = {"model": model, "messages": [{"role": "user", "content": "Reply OK"}],
            "max_completion_tokens": 16}
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=40) as r:
            d = json.load(r)
            content = (d.get("choices", [{}])[0].get("message", {}) or {}).get("content")
            print(f"  smoke {model}: {r.status} {content!r}")
            ok = ok and r.status == 200 and bool(content)
    except urllib.error.HTTPError as e:
        print(f"  smoke {model}: HTTP {e.code} {e.read().decode()[:80]}")
        ok = False
sys.exit(0 if ok else 1)
PY

echo "==> deploy-dev complete: ${REGISTRY}:dev-${SHA} live + smoke passed"
