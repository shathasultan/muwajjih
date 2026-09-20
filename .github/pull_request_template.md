## What changed

<!-- One or two sentences. What does this PR do, and why? -->

## Type of change

- [ ] Model change (retraining, new features, different algorithm)
- [ ] API change (endpoints, request/response contracts)
- [ ] Infrastructure (Docker, CI, configuration)
- [ ] Refactor / tests / docs only

## Review checklist

**Architecture**
- [ ] Dependencies still point inward: `domain/` imports nothing from `service/`, `adapters/` or `api/`
- [ ] `adapters/sklearn_model.py` is still the only module importing sklearn/joblib
- [ ] No model loading at import time -- loading happens in the composition root only
- [ ] Any new configuration goes through `Settings`, not a bare `os.environ[...]`

**Model changes only**
- [ ] `train/generate_dataset.py` still produces byte-identical output for the same seed
- [ ] Behavioural test thresholds reviewed -- if accuracy moved, was that intended?
- [ ] `MODEL_VERSION` bumped in `train/train_model.py`
- [ ] `models/metrics.json` reviewed for regressions

**Always**
- [ ] `ruff check` and `ruff format --check` pass
- [ ] `mypy src` passes (strict)
- [ ] New behaviour has a test at the right layer (unit / integration / behavioural)
- [ ] No secrets, API keys, or real customer data in the diff
