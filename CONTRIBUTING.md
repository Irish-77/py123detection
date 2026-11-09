# Contributing

Bug reports, questions and pull requests are all welcome.

The full guide, i.e. environment setup, the checks to run, and the conventions the code follows,
lives in the documentation:

**<https://irish-77.github.io/py123detection/contributing/>**

The short version:

```bash
pip install -e ".[dev]"
ruff check .    # lint
pytest          # 504 unit tests, no data needed, about a minute
```

CI runs both on every pull request (`.github/workflows/tests.yaml`), in two jobs: one installs
only `[dev]` and runs `-m "not torch"`, which is what keeps the export side honest about needing
neither torch nor hydra; the other installs every extra and runs the whole suite. Running them
locally first is still faster than finding out from a red tick.

The 28 `integration` tests are deselected by default because they need a nuScenes tree and its 123D
conversion; run them with `pytest -m integration` if you change anything under
`src/py123detection/mmcv/export/`.
