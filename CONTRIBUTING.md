# Contributing

Bug reports, questions and pull requests are all welcome.

The full guide, i.e. environment setup, the checks to run, and the conventions the code follows,
lives in the documentation:

**<https://irish-77.github.io/py123detection/contributing/>**

The short version:

```bash
pip install -e ".[dev]"
ruff check .    # lint
pytest          # 179 unit tests, no data needed, about 10 seconds
```

There is no CI for tests or linting yet, so please run both locally before opening a pull request.
The twelve `integration` tests are deselected by default because they need a nuScenes tree and its
123D conversion; run them with `pytest -m integration` if you change anything under
`src/py123detection/mmcv_export/`.
