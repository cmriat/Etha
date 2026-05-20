"""Sphinx configuration for Etha docs."""

# -- Project information -----------------------------------------------------

project = "Etha"
author = "Etha contributors"
copyright = "2026, Etha contributors"  # noqa: A001

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_parser",
    "autoapi.extension",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinxcontrib.mermaid",
]

source_suffix = {
    ".md": "markdown",
    ".rst": "restructuredtext",
}

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- HTML output -------------------------------------------------------------

html_theme = "sphinx_book_theme"
html_title = "Etha"
html_theme_options = {
    "repository_url": "https://github.com/cmriat/Etha",
    "repository_branch": "main",
    "path_to_docs": "docs",
    "use_repository_button": True,
    "use_source_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "home_page_in_toc": True,
}

# -- MyST --------------------------------------------------------------------

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "linkify",
    "substitution",
    "tasklist",
    "attrs_inline",
    "attrs_block",
]
myst_heading_anchors = 3
myst_fence_as_directive = ["mermaid"]

# -- AutoAPI (AST-based; does not import etha) -------------------------------

autoapi_type = "python"
autoapi_dirs = ["../src/etha"]
autoapi_root = "api"
autoapi_keep_files = False
autoapi_add_toctree_entry = False
autoapi_ignore = ["*/tests/*", "*_test.py"]
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
]
autoapi_python_class_content = "both"
autoapi_member_order = "groupwise"

# -- Napoleon (Google-style docstrings) --------------------------------------

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = False
napoleon_use_rtype = True
napoleon_use_ivar = True

# -- Autodoc -----------------------------------------------------------------

autodoc_typehints = "signature"
autodoc_typehints_description_target = "documented"

# -- Intersphinx -------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable", None),
}

# -- Suppress noisy warnings while the docs are bootstrapping ----------------

suppress_warnings = ["myst.header"]
