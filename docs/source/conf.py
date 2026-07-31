# spatialmath
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
import sys
import sphinx_rtd_theme
import re


# -- Project information -----------------------------------------------------

project = 'Spatial Geometry'
copyright = '2020, Jesse Haviland and Peter Corke'
author = 'Jesse Haviland and Peter Corke'

# Parse version number out of pyproject.toml
_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
with open(os.path.join(_root, 'pyproject.toml'), encoding='utf-8') as f:
    pyproject_src = f.read()
    m = re.search(r'^version\s*=\s*"([0-9.]*)"', pyproject_src, re.MULTILINE)
    version = m.group(1) if m else "unknown"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'sphinx.ext.coverage',
    'sphinx.ext.doctest',
    'sphinx.ext.inheritance_diagram',
    'sphinx_pyrunblock',
    'sphinx_copybutton',
    'sphinx_codeautolink',
    'sphinxcontrib.mermaid',
]

# sphinx_copybutton: strip the leading '>>> '/'... ' prompts when copying
copybutton_prompt_text = r'>>> |\.\.\. '
copybutton_prompt_is_regexp = True

# sphinx_codeautolink: link names in code blocks to their API docs
codeautolink_autodoc_inject = False

# sphinxcontrib.mermaid: size the SVG to its content instead of a fixed
# 500px height, which pads short/wide diagrams with blank space
mermaid_height = "auto"

autosummary_generate = True
autodoc_member_order = 'bysource'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

exclude_patterns = ['test_*']

# options for spinx_pyrunblock, used for inline examples
#  Python session setup, turn off color printing for SE3, set NumPy precision
autorun_languages = {}
autorun_languages['pycon_runfirst'] = """
from spatialmath import SE3
SE3._color = False
import numpy as np
np.set_printoptions(precision=4, suppress=True)
from ansitable import ANSITable
ANSITable._color = False
"""

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'

html_theme_options = {
    'logo_only': False,
    'prev_next_buttons_location': 'both',
    'analytics_id': 'G-11Q6WJM565',
    'style_external_links': True,
}
html_last_updated_fmt = '%d-%b-%Y'
show_authors = True

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_css_files = [
    'css/custom.css',
]
# autodoc_mock_imports = ["numpy", "scipy"]


# -- Options for LaTeX/PDF output --------------------------------------------
latex_engine = 'xelatex'
# maybe need to set graphics path in here somewhere
# \graphicspath{{figures/}{../figures/}{C:/Users/me/Documents/project/figures/}}
# https://stackoverflow.com/questions/63452024/how-to-include-image-files-in-sphinx-latex-pdf-files
latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    'papersize': 'a4paper',
    # 'releasename':" ",
    # Sonny, Lenny, Glenn, Conny, Rejne, Bjarne and Bjornstrup
    # 'fncychap': '\\usepackage[Lenny]{fncychap}',
    'fncychap': '\\usepackage{fncychap}',
    'maketitle': "blah blah blah"
}

# Use RVC book notation for maths
# see https://stackoverflow.com/questions/9728292/creating-latex-math-macros-within-sphinx
mathjax3_config = {
    'TeX': {
        'Macros': {
            # RVC Math notation
            #  - not possible to do the if/then/else approach
            #  - subset only
            "presup": [r"\,{}^{\scriptscriptstyle #1}\!", 1],
            # groups
            "SE": [r"\mathbf{SE}(#1)", 1],
            "SO": [r"\mathbf{SO}(#1)", 1],
            "se": [r"\mathbf{se}(#1)", 1],
            "so": [r"\mathbf{so}(#1)", 1],
            # vectors
            "vec": [r"\boldsymbol{#1}", 1],
            "dvec": [r"\dot{\boldsymbol{#1}}", 1],
            "ddvec": [r"\ddot{\boldsymbol{#1}}", 1],
            "fvec": [r"\presup{#1}\boldsymbol{#2}", 2],
            "fdvec": [r"\presup{#1}\dot{\boldsymbol{#2}}", 2],
            "fddvec": [r"\presup{#1}\ddot{\boldsymbol{#2}}", 2],
            "norm": [r"\Vert #1 \Vert", 1],
            # matrices
            "mat": [r"\mathbf{#1}", 1],
            "dmat": [r"\dot{\mathbf{#1}}", 1],
            "fmat": [r"\presup{#1}\mathbf{#2}", 2],
            # skew matrices
            "sk": [r"\left[#1\right]", 1],
            "skx": [r"\left[#1\right]_{\times}", 1],
            "vex": [r"\vee\left( #1\right)", 1],
            "vexx": [r"\vee_{\times}\left( #1\right)", 1],
            # quaternions
            "q": r"\mathring{q}",
            "fq": [r"\presup{#1}\mathring{q}", 1],

        }
    }
}
