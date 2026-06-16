
default:
	echo "Please specify a target: start-backend or start-frontend or test"

test:
	cd llm_api_converter && uv sync --extra dev && uv run pytest
	$(MAKE) test-backend

test-backend:
	./backend/run-tests.sh

start-backend:
	cd backend && uv sync && uv run python -m app.main

start-frontend:
	cd frontend && npm install && npm run dev

dev:
	./start-dev.sh

# Build origin/dev -> push Nexus (dev-latest + dev-<sha>) -> Coolify redeploy -> verify.
# Self-verifying; see scripts/deploy-dev.sh and the inference-network repo's
# docs/coolify-source-mapping.md for the full topology.
deploy-dev:
	./scripts/deploy-dev.sh

.PHONY: start-backend start-frontend test test-backend dev deploy-dev
