# Contributing to DeepGesture

## Before you start

Check [open issues](../../issues) — especially ones labeled `good first issue` or `help wanted`. Comment on the issue before starting work to avoid duplicate effort.

## Setup

```bash
git clone https://github.com/saitarrun/DeepGesture
cd DeepGesture
pip install -r requirements.txt
pip install pytest pytest-cov ruff
```

**Requirements:** macOS 13+, Python 3.10–3.12, a webcam.

## Workflow

1. Fork → feature branch off `main` (`feat/short-description` or `fix/short-description`)
2. Write a failing test first if fixing a bug
3. Implement the change
4. Run tests: `pytest test_virtual_mouse.py -v`
5. Run lint: `ruff check . --select=E,W,F,I --ignore=E501`
6. Open a PR — fill out the template completely

## Code standards

- Functions do one thing. If it needs a comment to explain *what* it does, split it.
- No dead code. No commented-out blocks. Git history is the undo button.
- New features need tests. Same style as `test_virtual_mouse.py`: synthetic landmarks, no camera, no OS calls.

## ML contributions

If changing the classifier or adding gestures:

- Report accuracy on a held-out test split (not training accuracy)
- Include a confusion matrix or per-class breakdown
- Export to ONNX and verify inference latency < 5ms on M1
- Document the gesture in `GESTURES.md` (create if missing)

## Commit messages

Format: `<type>: <short imperative summary>`

Types: `feat`, `fix`, `perf`, `ml`, `test`, `docs`, `refactor`, `ci`

Examples:
```
feat: add Kalman filter for landmark temporal smoothing
fix: hysteresis dead-band not applied on fast pinch release
ml: replace MLP with transformer encoder, +3.1% accuracy on 10-gesture set
perf: skip MediaPipe detection on zero-motion frames, +18fps
```

## Reporting bugs

Use the Bug Report issue template. Include:
- macOS version + hardware (M1/M2/Intel)
- Python version
- Exact error or unexpected behavior
- Steps to reproduce (landmark recording preferred over video)

## Questions

Open a [Discussion](../../discussions) rather than an issue for design questions.
