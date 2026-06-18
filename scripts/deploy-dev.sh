#!/usr/bin/env bash
# Self-verifying dev deploy for the Squirrel LLM gateway image.
#
# Builds the gateway image from origin/dev (a clean worktree, never the dirty
# checkout), pushes it to Nexus as the mutable `dev-latest` tag plus an
# immutable `dev-<sha>` tag, pulls that immutable tag onto the deploy host,
# repoints the host-local `dev-latest` tag, triggers the inference-network
# Coolify app to recreate the stack, verifies the running container image id,
# and smoke-tests two providers so a recreate that breaks provider-key
# decryption is caught immediately.
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
DEPLOY_HOST="popos-sf3.com"
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

# 3. Bust the deploy host's mutable-tag cache before Coolify recreates the
#    compose stack. Coolify can report a successful force deploy while Docker
#    keeps the previous host-local dev-latest image; pulling the immutable tag
#    first and repointing dev-latest makes the compose image reference resolve
#    to exactly the build above.
ssh -o BatchMode=yes "root@${DEPLOY_HOST}" \
  "docker pull ${REGISTRY}:dev-${SHA} && docker tag ${REGISTRY}:dev-${SHA} ${REGISTRY}:dev-latest"
echo "==> ${DEPLOY_HOST} dev-latest retagged to ${REGISTRY}:dev-${SHA}"

# 4. Trigger the Coolify deploy (force => recreate against the host-local tag above).
CK="$(doppler run -p personal -c dev -- printenv COOLIFY_API_KEY)"
curl -fsS "${COOLIFY}/api/v1/deploy?uuid=${APP_UUID}&force=true" \
  -H "Authorization: Bearer ${CK}" >/dev/null
echo "==> Coolify deploy queued (force re-pull)"

# 5. Poll the deployment to a terminal state.
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

# 6. Verify Coolify actually recreated from the image we just built. This catches
#    stale mutable-tag reuse before provider smokes can produce a false green.
EXPECTED_IMAGE="$(ssh -o BatchMode=yes "root@${DEPLOY_HOST}" \
  "docker image inspect --format '{{.ID}}' ${REGISTRY}:dev-${SHA}")"
RUNNING_IMAGE="$(ssh -o BatchMode=yes "root@${DEPLOY_HOST}" '
container="$(docker ps --format "{{.Names}}" | grep "^llm-gateway-bdtgxpkb79qusq5f7szgl5xw" | head -1)"
test -n "$container"
docker inspect "$container" --format "{{.Image}}"
')"
if [[ "$RUNNING_IMAGE" != "$EXPECTED_IMAGE" ]]; then
  echo "!! Coolify is running ${RUNNING_IMAGE}, expected ${EXPECTED_IMAGE} (${REGISTRY}:dev-${SHA})"
  exit 1
fi
echo "==> running gateway image verified: ${REGISTRY}:dev-${SHA} (${RUNNING_IMAGE})"

# 7. Smoke test: real calls prove the image is live AND the provider keys still
#    decrypt after the Squirrel recreate. max_completion_tokens must be >= 16
#    (the OpenAI Responses API rejects anything lower). The Anthropic-compatible
#    GPT smoke covers the harness path where metadata.user_id translates to the
#    OpenAI Responses user field.
KEY="$(doppler run -p back-end -c staging -- printenv LLM_API_KEY)"
KEY="$KEY" python3 - "$SHA" <<'PY'
import json, os, sys, urllib.request, urllib.error
key = os.environ["KEY"]; sha = sys.argv[1]
chat_url = "https://api.llm-gateway.iocloudhost.net/v1/chat/completions"
messages_url = "https://api.llm-gateway.iocloudhost.net/v1/messages?beta=true"
ok = True
for model in ("gpt-5.4", "claude-opus-4-7"):
    body = {"model": model, "messages": [{"role": "user", "content": "Reply OK"}],
            "max_completion_tokens": 16}
    req = urllib.request.Request(chat_url, data=json.dumps(body).encode(),
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

body = {
    "model": "gpt-5.4",
    "messages": [{"role": "user", "content": "Reply OK"}],
    "max_tokens": 16,
    "metadata": {"user_id": "u" * 150},
}
req = urllib.request.Request(messages_url, data=json.dumps(body).encode(), headers={
    "Authorization": f"Bearer {key}",
    "Content-Type": "application/json",
    "anthropic-version": "2023-06-01",
})
try:
    with urllib.request.urlopen(req, timeout=40) as r:
        d = json.load(r)
        content = d.get("content") or []
        text = content[0].get("text") if content and isinstance(content[0], dict) else None
        print(f"  smoke gpt-5.4 messages long-user: {r.status} {text!r}")
        ok = ok and r.status == 200 and bool(text)
except urllib.error.HTTPError as e:
    print(f"  smoke gpt-5.4 messages long-user: HTTP {e.code} {e.read().decode()[:120]}")
    ok = False

sys.exit(0 if ok else 1)
PY

echo "==> deploy-dev complete: ${REGISTRY}:dev-${SHA} live + smoke passed"
