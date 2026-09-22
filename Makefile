# One command interface for the whole project. Every target below is exactly
# what CI runs -- so "green locally, red in CI" is a bug in this file, not a
# mystery. `make help` lists them.

.DEFAULT_GOAL := help
SHELL := /bin/bash
IMAGE ?= intent-service
TAG   ?= local
PORT  ?= 8000
# A throwaway key so `make smoke` needs no .env. Never used outside the
# ephemeral smoke container.
SMOKE_KEY := smoke-test-key

.PHONY: help install train lint test gate image smoke compose-up compose-down clean all

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "};{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Install the package with dev + train extras
	pip install -e ".[dev,train]"
	pre-commit install || true

train: ## Regenerate the dataset (seeded) and retrain the model
	python -m train.generate_dataset
	python -m train.train_model

lint: ## Ruff, formatting, strict mypy, YAML, and the architecture contract
	ruff check src tests train scripts
	ruff format --check src tests train scripts
	mypy src
	lint-imports
	@# A malformed workflow file cannot be caught by the workflow itself -- the
	@# run fails at startup with zero jobs and no usable error. Parse it here.
	@python -c "import sys, yaml; [yaml.safe_load(open(f)) for f in sys.argv[1:]]; print('yaml ok')" \
		.github/workflows/ci.yml docker-compose.yml .pre-commit-config.yaml .github/dependabot.yml

gate: ## FAST gate -- lint + unit + integration, must finish in < 60s
	@start=$$(date +%s); \
	$(MAKE) --no-print-directory lint && \
	pytest tests/unit tests/integration -q --no-header; \
	rc=$$?; \
	elapsed=$$(( $$(date +%s) - start )); \
	echo "fast gate finished in $${elapsed}s (budget: 60s)"; \
	if [ $$elapsed -gt 60 ]; then echo "FAIL: fast gate exceeded its 60s budget"; exit 1; fi; \
	exit $$rc

test: ## Full suite: all three layers with the branch-coverage gate
	pytest tests -q --cov --cov-report=term-missing

image: ## Build the runtime image and assert it stays under 500 MB
	docker build -t $(IMAGE):$(TAG) .
	@bytes=$$(docker image inspect $(IMAGE):$(TAG) --format '{{.Size}}'); \
	mb=$$(( bytes / 1000000 )); \
	echo "image size: $${mb} MB (budget: 500 MB)"; \
	if [ $$mb -gt 500 ]; then echo "FAIL: image exceeds the 500 MB budget"; exit 1; fi

smoke: image ## Run the built image and exercise it over real HTTP
	@./scripts/smoke.sh $(IMAGE):$(TAG) $(PORT) $(SMOKE_KEY)

compose-up: ## Start the service and Redis, waiting on real health
	docker compose up -d --wait

compose-down: ## Stop the stack and remove its volumes
	docker compose down -v

clean: ## Remove build, cache and coverage artefacts
	rm -rf .pytest_cache .ruff_cache .mypy_cache .coverage htmlcov build dist *.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

all: lint test image smoke ## Everything CI runs, in CI's order
