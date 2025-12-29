# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
import os
import sys

sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.path.abspath('../gw_signal_tools/'))


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'gw-signal-tools'
author = 'Max Melching, Frank Ohme'
copyright = '2024, ' + author

from gw_signal_tools._version import version as VERSION  # noqa: E402

release = VERSION
# release = 'v0.2.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.todo',  # includes todos
    'sphinx.ext.viewcode',  # syntax highlighting
    'sphinx.ext.autodoc',  # includes documentation from docstrings
    'sphinx.ext.napoleon',  # support other docstring formats
    'sphinx.ext.mathjax',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosectionlabel',  # Enables :ref:`section name`
    'sphinxcontrib.programoutput',
    'numpydoc',
    'nbsphinx',
    'nbsphinx_link',
    'myst_parser',  # Modern markdown parser - replaces m2r2
]

nbsphinx_execute = 'never'
nbsphinx_allow_errors = True

# Enable markdown files to be processed by myst_parser
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# Autodoc options
autoclass_content = 'class'
autodoc_default_flags = ['show-inheritance', 'members', 'inherited-members']
autosummary_generate = True


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
html_theme = 'sphinx_rtd_theme'

todo_include_todos = True

napoleon_numpy_docstring = True
numpydoc_use_blockquotes = True


html_static_path = ['_static']

html_theme_options = {
    'logo_only': False,
    'display_version': True,
    # ----- Options for TOC sidebar -----
    'collapse_navigation': False,  # Makes navigation expandable
    'sticky_navigation': True,
    'navigation_depth': 4,
    'titles_only': False,
}
