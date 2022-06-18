# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'GElib'
copyright = '2021-2022, Risi Kondor'
author = 'Risi Kondor'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc','sphinx.ext.intersphinx','sphinx.ext.autosummary','sphinx.ext.napoleon',
              'sphinx.ext.imgmath'
              #,'sphinx_copybutton'
              ]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'furo'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

intersphinx_mapping = {'https://docs.python.org/': None}

imgmath_image_format = 'svg'

#imgmath_image_format = 'png'
#imgmath_dvipng_args=['-gamma', '1.5', '-D', '110', '-bg', 'Transparent']
imgmath_use_preview=True

imgmath_font_size = 14

autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': True, #'__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__,__module__,__hash__'
#    'add_module_names': False
}

add_module_names = False
autodoc_member_order = 'bysource'
