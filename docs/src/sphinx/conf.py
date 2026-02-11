import os
import sys
sys.path.insert(0, os.path.abspath('../../..'))

project = 'Thor Semantic Audio Agent'
copyright = '2026, LHZN'
author = 'LHZN'
release = '0.1.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'myst_parser'
]

templates_path = ['_templates']
exclude_patterns = []

html_theme = 'alabaster'
html_static_path = ['_static']
