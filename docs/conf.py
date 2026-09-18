# Configuration file for the Sphinx documentation builder.
#
# Build from the repository root: sphinx-build -b html docs docs/_build/html
# See docs/development.md, and https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
import types
from html import escape

import py123detection

# -- Project information -----------------------------------------------------

project = "py123detection"
copyright = "2026"
release = py123detection.__version__
version = release

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "sphinxarg.ext",
    "sphinxcontrib.mermaid",
    "myst_parser",
]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Generate anchors for Markdown headings (levels h1-h3) so GitHub-style links like
# [Known differences](validation.md#known-differences) resolve.
myst_heading_anchors = 3
# Render ```mermaid fences as diagrams, so the same Markdown works on GitHub and here.
myst_fence_as_directive = ["mermaid"]

# -- API reference -----------------------------------------------------------

autosummary_generate = True

# The training-side plugin subclasses mmdet3d's NuScenesDataset and only defines it when mmdet3d
# imports. Mocking lets the docs build in a conversion environment and still document it.
autodoc_mock_imports = ["mmcv", "mmdet3d", "torch"]

# A mocked ``@DATASETS.register_module()`` would replace the class with a mock and hide its methods,
# so mmdet's registry gets a pass-through decorator instead (only when mmdet is not installed).
if "mmdet" not in sys.modules:
    try:
        import mmdet.datasets  # noqa: F401
    except ImportError:
        _mmdet_datasets = types.ModuleType("mmdet.datasets")
        _mmdet_datasets.DATASETS = types.SimpleNamespace(register_module=lambda *args, **kwargs: lambda cls: cls)
        sys.modules["mmdet"] = types.ModuleType("mmdet")
        sys.modules["mmdet.datasets"] = _mmdet_datasets

# Show defaults as written where the source is available.
autodoc_preserve_defaults = True

# Dataclass signatures (``ExportConfig``) and module constants (``TAXONOMIES``) still show defaults
# by repr, and a Taxonomy's generated repr runs to ~1500 characters. Shorten it for the docs only.
if py123detection.HAS_EXPORT_SIDE:
    from py123detection.taxonomy import Taxonomy

    Taxonomy.__repr__ = lambda self: f"Taxonomy(name={self.name!r})"

autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "undoc-members": True,  # dataclass fields are mostly undocumented but are the API
    "show-inheritance": True,
    # Document each name in the module that defines it, not again in every package that
    # re-exports it through __all__.
    "ignore-module-all": True,
    "exclude-members": "logger",
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "py123d": ("https://kesai.eu/py123d", None),
}

# -- Options for HTML output -------------------------------------------------

html_theme = "furo"
html_title = f"py123detection {release}"

# The docs workflow publishes the default branch at the site root and every other branch under
# /<branch>/, and passes the resulting URL in. Both are unset locally, which is fine: Sphinx
# writes relative links, so the same build works under any prefix.
html_baseurl = os.environ.get("DOCS_BASE_URL", "")

# A branch preview is published on demand, so it is a snapshot of whatever that branch was at the
# time. Name the branch and the commit in a banner, so a shared link is neither mistaken for the
# released docs nor for the branch's current state.
_preview_branch = os.environ.get("DOCS_PREVIEW_BRANCH", "")
if _preview_branch:
    _site_root = os.environ.get("DOCS_SITE_ROOT", "")
    _commit = os.environ.get("DOCS_PREVIEW_COMMIT", "")[:7]
    html_theme_options = {
        "announcement": (
            f"Preview of branch <code>{escape(_preview_branch)}</code>"
            + (f" at <code>{escape(_commit)}</code>" if _commit else "")
            + f' &mdash; <a href="{escape(_site_root)}/">go to the main docs</a>'
        ),
    }
