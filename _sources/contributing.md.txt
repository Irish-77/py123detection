# Contributing

Bug reports, questions and pull requests are all welcome. This page covers getting a working
environment, the checks to run before opening a pull request, and the conventions the existing
code follows. For where the modules live and what each one does, see
[Development](development.md).

## Setting up

```bash
git clone https://github.com/Irish-77/py123detection.git
cd py123detection
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

The export side needs only py123d and numpy: it never imports mmcv, mmdet3d or torch, and the
training-side plugin imports mmdet3d lazily, inside the one module that needs it. The whole unit
test suite therefore runs in a plain conversion environment, so **you do not need a working
mmdet3d installation to work on most of this codebase** -- which is deliberate, because a
mmcv/mmdet3d environment is pinned to Python 3.8 and sometimes painful to build.

Add the extras for the part you are touching:

| extra | pulls in | needed when |
| --- | --- | --- |
| `dev` | pytest, ruff | always -- it is what runs the checks below |
| `docs` | sphinx, furo, myst-parser, ... | editing anything under `docs/`, or a docstring you want to see rendered |
| `nuscenes` | nuscenes-devkit | working on native nuScenes token restoration |
| `images` | opencv-python | working on `image_mode=extract` |

One environment caveat worth knowing before you debug a confusing pickle failure: if your
conversion environment is on numpy 2, the pickles it writes will not load under the numpy 1 that
mmdet3d environments use. Installing `numpy<2` alongside py123d is the simple fix;
[Crossing the Python version boundary](guides/python-version-boundary.md) explains the whole
hazard and the alternative.

## Checks before a pull request

Three commands, none of which need any data on disk:

```bash
ruff check .                                # lint; the configuration lives in pyproject.toml
pytest                                      # 179 unit tests, about 10 seconds
sphinx-build -b dirhtml docs docs/_build    # only if you touched docs/ or a docstring
```

Note that `ruff format` is deliberately **not** on that list. The tree is not formatter-managed,
and running it today would reformat 14 files that have nothing to do with your change. Match the
style of the code around you instead.

Please do run these locally: there is no CI for tests or linting yet, so nothing else will catch a
regression. The only automated workflow builds the documentation.

### The integration tests

Twelve tests are marked `integration` and deselected by default (`addopts` in `pyproject.toml`),
because they need a real nuScenes tree and its 123D conversion side by side:

```bash
export NUSCENES_DATA_ROOT=/path/to/nuscenes    # holds v1.0-mini/, samples/, sweeps/
export PY123D_DATA_ROOT=/path/to/py123d        # the 123D conversion of that nuScenes
pytest -m integration
```

They export a pickle, build the reference pickle on the fly with
`tools/reference_nuscenes_converter.py` -- a port of the converter mmdetection3d and StreamPETR
ship -- and compare the two field by field. That comparison is the project's central claim, so
run it if you change anything under `src/py123detection/mmcv_export/`, and say in the pull request
whether it still holds. [Validation](validation.md) describes what it checks and the two
differences that are expected.

## Conventions

**Style.** ruff with `line-length = 120` and rules `E`, `F`, `I` -- pycodestyle errors, pyflakes,
and import sorting -- ignoring `E501` and `E741`, with `__init__.py` excluded so re-exports can be
grouped by hand. `ruff check .` is the authority; nothing else is enforced.

**Docstrings** are reST field lists, rendered by `sphinx.ext.autodoc` directly. There is no
napoleon extension, so Google- and NumPy-style sections will not render. Types come from the
annotations via `sphinx-autodoc-typehints`, so do not repeat them in the text:

```python
def export_to_mmdet3d(sources, output_path, config=None, version="py123d-v1.0") -> ExportReport:
    """Export one or more 123D sources into a single mmdetection3d info pickle.

    :param version: Stored as ``metadata['version']``, which mmdet3d reads back.
    """
```

Document a parameter when its meaning is not obvious from its name and type; the existing
docstrings skip the ones that are.

**Comments** in this codebase explain *why*, not *what* -- the constraint that forced a decision,
the reason a default is what it is, the failure that a check exists to prevent. A comment
restating the line below it is noise; a comment recording why protocol 4 is the default is the
reason the next person does not break it. Please keep writing them that way.

**Tests** are named as sentences about the behaviour they pin down --
`test_rebuild_preserves_the_shape_of_empty_arrays`, not `test_rebuild_2`. New behaviour goes in
the `tests/test_<module>.py` that matches the module it lives in.

## Adding a documentation page

Write the Markdown under `docs/` (or `docs/guides/` for a task-oriented one), then add it to a
`toctree` in `docs/index.md` -- Sphinx warns about any page that no toctree includes, and the page
is unreachable until you do.

MyST is configured so the same Markdown reads well on GitHub and on the site: heading anchors are
generated down to `h3`, so `[text](validation.md#known-differences)` resolves in both places, and
a ` ```mermaid ` fence renders as a diagram rather than a code block. See
[Building the docs](development.md#building-the-docs) for the local preview.

## Opening the pull request

Branch off `main` with a `feat/` or `fix/` prefix, and avoid naming a branch after a top-level
documentation page such as `guides` or `api` -- branch documentation previews are published
alongside those pages and would shadow them, as
[Publishing](development.md#publishing) explains.

Say what changed and why. If the change affects what ends up in an exported pickle, say so
explicitly and report the integration comparison. If a reviewer would benefit from reading the
rendered documentation, publish a preview for your branch: **Actions -> docs -> Run workflow**,
pick the branch, and link the resulting URL in the pull request.

Contributions are made under the project's Apache-2.0 license, the full text of which is in
`LICENSE` at the repository root.
