# Repository Guidelines

## Objective
- Optimize ML model for detecting the presence and direction of arrows

## Project Structure & Module Organization
- Core code lives in `src/arrow_classification_model/`: `train_main.py` trains the ResNet18 model; `predict_main.py` loads `arrow_classifier_resnet18_3class.pth` and reports results.
- Data layout (relative to repo root): `train/arrow/_classes.csv` with `filename,left,right` plus images; `train/none/` for negative samples; `predict/` for evaluation images grouped by class.
- Saved weights default to `arrow_classifier_resnet18_3class.pth` at the repo root. Update the path in the scripts only if you also update docs and PR notes.
- `tests/` is ready for Pytest-style tests (`test_*.py`); add fixtures under `tests/fixtures/` when needed.

## Environment & Setup
- Python 3.13; manage deps with Poetry. Install via `poetry install` (creates `.venv` if needed). Activate with `poetry shell` or prefix commands with `poetry run`.
- GPU is optional; scripts automatically use CUDA if available.

## Build, Train, and Predict Commands
- `poetry run train`: trains the 3-class model using data under `train/` and saves weights to `arrow_classifier_resnet18_3class.pth`. Adjust `train_root` in `train_main.py` only if your data lives elsewhere.
- `poetry run predict`: loads the saved weights, scans `predict/` for images, and prints a Rich table with per-image and per-class accuracy.
- For ad-hoc execution: `poetry run python src/arrow_classification_model/train_main.py` or `.../predict_main.py`.

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indentation and type hints. Keep function/variable names in `snake_case`; class names in `CapWords`.
- Prefer `pathlib.Path` for paths and keep class-to-index mappings synchronized between train and predict (`CLASS_TO_IDX` / `IDX_TO_CLASS`).
- Add concise docstrings for public functions; avoid silent failures except where explicitly guarded.

## Testing Guidelines
- Use Pytest; place tests in `tests/test_*.py`. Example: `poetry run pytest -q`.
- Mock file I/O and use tiny image fixtures to keep tests fast; avoid requiring a GPU. When adding metrics, assert on shapes and class counts rather than absolute accuracy.
- Target at least smoke coverage for CLI entry points (train/predict) using temporary directories.

## Commit & Pull Request Guidelines
- Commits are short, present-tense summaries (see history: “added none dataset”, “added dataset”). Group logical changes; avoid bundling model artifacts with code changes unless necessary.
- PRs should include: purpose and scope, data location/size notes, commands run (train/predict/tests), and any accuracy deltas. Link related issues and add screenshots or tables when output format changes.
- If you add or move model files or datasets, document the new paths in the PR description and update this guide when appropriate.
