# Configuration file for the Sphinx documentation builder.
#
# For a full list of configuration options see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from __future__ import annotations

import sys
from pathlib import Path

# Make the package importable without installation.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# ---------------------------------------------------------------------------
# Project information
# ---------------------------------------------------------------------------

project = "exodusii"
copyright = "NTESS"
author = "Timothy Jesse Fuller"

# Read the version from the installed (or editable) package.
try:
    from importlib.metadata import version as _version

    release = _version("exodusii")
except Exception:
    release = "0.1.0a0"

version = ".".join(release.split(".")[:2])

# ---------------------------------------------------------------------------
# General configuration
# ---------------------------------------------------------------------------

extensions = [
    # Core autodoc machinery
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    # NumPy-style docstring parsing
    "numpydoc",
    # Cross-references to Python builtins / NumPy / netCDF4 etc.
    "sphinx.ext.intersphinx",
    # Source-code links
    "sphinx.ext.viewcode",
    # Copy-button on code blocks
    "sphinx_copybutton",
    # Doctest validation
    "sphinx.ext.doctest",
    # Search
    "sphinx.ext.ifconfig",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The root document.
root_doc = "index"

# ---------------------------------------------------------------------------
# HTML output
# ---------------------------------------------------------------------------

html_theme = "furo"
html_static_path = ["_static"]

html_theme_options = {
    "sidebar_hide_name": False,
    "light_css_variables": {
        "color-brand-primary": "#1a6496",
        "color-brand-content": "#1a6496",
    },
    "dark_css_variables": {
        "color-brand-primary": "#4db8ff",
        "color-brand-content": "#4db8ff",
    },
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/sandialabs/exodusii",
            "html": (
                '<svg stroke="currentColor" fill="currentColor" stroke-width="0" '
                'viewBox="0 0 16 16"><path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 '
                "3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37"
                "-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01"
                " 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3"
                ".64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-"
                ".21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2"
                '-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 '
                "3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 "
                '8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path></svg>'
            ),
            "class": "",
        },
    ],
}

html_title = f"{project} {version}"

# Copy-button: skip prompts
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

# ---------------------------------------------------------------------------
# Autodoc configuration
# ---------------------------------------------------------------------------

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "inherited-members": False,
    "special-members": "__bool__",
}

# Show type annotations in the signature line.
autodoc_typehints = "signature"
autodoc_typehints_format = "short"

# Let autodoc render __init__ parameters in the class docstring.
autoclass_content = "class"

# Generate autosummary stubs automatically.
autosummary_generate = True
autosummary_generate_overwrite = True

# ---------------------------------------------------------------------------
# numpydoc configuration
# ---------------------------------------------------------------------------

numpydoc_show_class_members = False  # autosummary handles this
numpydoc_show_inherited_class_members = False
numpydoc_class_members_toctree = False
numpydoc_xref_param_type = True
# Only validate checks that catch genuine documentation bugs; suppress
# stylistic warnings (GL01 summary placement, ES01 extended summary,
# SA01 See Also) and checks that numpydoc mis-fires on dataclass fields.
numpydoc_validation_checks = {"PR04", "PR07", "RT01", "SS01"}
numpydoc_validation_exclude = {
    r"exodusii\.compat\.",
    r"exodusii\.core\.names\.",
    r"exodusii\.core\.schema\.",
    r"exodusii\.core\.strings\.",
    r"exodusii\.core\.time\.",
    r"exodusii\.core\.selectors\.",
    r"exodusii\.io\.",
    r"exodusii\.region\.",
    r"exodusii\.allclose\.",
    r"exodusii\.similar\.",
    r"exodusii\.copy\.",
    r"exodusii\.file\.",
    r"exodusii\.parallel_file\.",
    r"exodusii\.lineout\.",
    r"exodusii\.element\.",
    r"exodusii\.exodus_h\.",
    r"exodusii\.util\.",
    r"exodusii\.extension\.",
}

# ---------------------------------------------------------------------------
# Intersphinx — cross-references to external packages
# (requires network access; silently skipped if unreachable)
# ---------------------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "netCDF4": ("https://unidata.github.io/netcdf4-python", None),
}
# Do not fail the build when remote inventories are unreachable.
intersphinx_disabled_reftypes = ["*"]

# ---------------------------------------------------------------------------
# Doctest
# ---------------------------------------------------------------------------

doctest_global_setup = """
import numpy as np
import exodusii
"""
