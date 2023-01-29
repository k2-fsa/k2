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
import os
import re
import sys
sys.path.insert(0, os.path.abspath('../../k2/python'))
sys.path.insert(0, os.path.abspath('../../build/lib'))

import sphinx_rtd_theme

# -- Project information -----------------------------------------------------

project = 'k2'
copyright = '2020-2022, k2 development team'
author = 'k2 development team'


def get_version():
    cmake_file = '../../CMakeLists.txt'
    with open(cmake_file) as f:
        content = f.read()

    version = re.search(r'set\(K2_VERSION (.*)\)', content).group(1)
    return version.strip('"')


# The full version, including alpha/beta/rc tags
version = get_version()
release = version

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'recommonmark',
    'sphinx.ext.autodoc',
    'sphinx.ext.githubpages',
    'sphinx.ext.linkcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    'sphinx_autodoc_typehints',
    'sphinx_rtd_theme',
    'sphinxcontrib.bibtex',
    'sphinxcontrib.youtube',
]
bibtex_bibfiles = ['refs.bib']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}
master_doc = 'index'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['installation/images/*.md']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_show_sourcelink = True

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

pygments_style = 'sphinx'

numfig = True

html_context = {
    'display_github': True,
    'github_user': 'k2-fsa',
    'github_repo': 'k2',
    'github_version': 'master',
    'conf_py_path': '/docs/source/',
}

# refer to
# https://sphinx-rtd-theme.readthedocs.io/en/latest/configuring.html
html_theme_options = {
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': True,
}

autodoc_default_options = {
    'content': 'both',
    'members': None,
    'member-order': 'bysource',
    #  'special-members': '__init__'
    'undoc-members': True,
    'exclude-members': '__weakref__'
}


# Resolve function for the linkcode extension.
# Modified from https://github.com/rwth-i6/returnn/blob/master/docs/conf.py
def linkcode_resolve(domain, info):

    def find_source():
        # try to find the file and line number, based on code from numpy:
        # https://github.com/numpy/numpy/blob/master/doc/source/conf.py#L286
        obj = sys.modules[info['module']]
        for part in info['fullname'].split('.'):
            obj = getattr(obj, part)
        import inspect
        import os
        fn = inspect.getsourcefile(obj)
        fn = os.path.relpath(fn, start='k2')
        source, lineno = inspect.getsourcelines(obj)
        return fn, lineno, lineno + len(source) - 1

    if domain != 'py' or not info['module']:
        return None
    try:
        filename = '{}#L{}-L{}'.format(*find_source())
    except Exception:
        return None

    if '_k2' in filename:
        return None

    idx = filename.rfind('k2')
    filename = filename[idx:]
    return f'https://github.com/k2-fsa/k2/blob/master/k2/python/{filename}'

# Replace key with value in the generated doc
REPLACE_PATTERN = {
  # somehow it results in errors
  # Handler <function process_docstring at 0x7f47a290aca0> for event
  # 'autodoc-process-docstring' threw an exception (exception:
  # <module '_k2.ragged'> is a built-in module)
  #
  #  '_k2.ragged': 'k2.ragged',
  'at::Tensor': 'torch.Tensor'
}

def replace(s):
    replaced = True
    while replaced:
        replaced = False
        for key in REPLACE_PATTERN:
            if key in s:
                s = s.replace(key, REPLACE_PATTERN[key])
                replaced = True
    return s

# see https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#event-autodoc-process-docstring
def replace_doc(app, what, name, obj, options, lines):
    num_lines = len(lines)
    for i in range(num_lines):
        lines[i] = replace(lines[i])

# see https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#event-autodoc-process-signature
def replace_signature(app, what, name, obj, options, signature, return_annotation):
    if signature:
        signature = replace(signature)

    if return_annotation:
        return_annotation = replace(return_annotation)
    return (signature, return_annotation)

# Note: setup is called by sphinx automatically
#
# See https://www.sphinx-doc.org/en/master/extdev/appapi.html#extension-setup
def setup(app):
    app.add_css_file('custom.css')
    app.connect('autodoc-process-signature', replace_signature)
    app.connect('autodoc-process-docstring', replace_doc)
