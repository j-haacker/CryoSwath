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
import subprocess
import sys
from datetime import datetime
from pathlib import Path
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

sys.path.insert(0, os.path.abspath('..'))
# sys.path.insert(0, os.path.abspath(os.path.join('..', 'cryoswath')))
# print(sys.path)


# -- Project information -----------------------------------------------------

project = 'cryoswath'
copyright = f"2024-{datetime.now().year}, Jan Haacker"
author = "Jan Haacker"

# The full version, including alpha/beta/rc tags
try:
    with (Path(__file__).resolve().parents[1] / "pyproject.toml").open("rb") as f:
        release = tomllib.load(f)["project"]["version"]
except Exception:
    try:
        import cryoswath

        release = cryoswath.__version__
    except Exception:
        release = "unknown"
version = release


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.githubpages',
    'sphinx.ext.linkcode',
    # 'myst_parser'
]

autodoc_typehints = "description"


_GIT_SHA_RE = re.compile(r"^[0-9a-fA-F]{7,40}$")


def _valid_git_sha(value):
    """Return a normalized SHA if value looks like a git commit hash."""
    if not value:
        return None
    value = value.strip()
    return value if _GIT_SHA_RE.fullmatch(value) else None


def _git_head_ref():
    """Return git HEAD commit hash for this repository, if available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            check=True,
            text=True,
        )
    except Exception:
        return None
    return _valid_git_sha(result.stdout)


def _resolve_source_ref(env=None, git_head_resolver=None):
    """Resolve source URL git ref using commit hash + fallback chain."""
    env = os.environ if env is None else env
    git_head_resolver = _git_head_ref if git_head_resolver is None else git_head_resolver

    commit_hash = _valid_git_sha(env.get("READTHEDOCS_GIT_COMMIT_HASH"))
    if commit_hash:
        return commit_hash

    head_hash = git_head_resolver()
    if head_hash:
        return head_hash

    git_identifier = (env.get("READTHEDOCS_GIT_IDENTIFIER") or "").strip()
    if git_identifier:
        return git_identifier

    return "main"


# Resolve once so all links within one docs build point to the same ref.
SOURCE_CODE_GIT_REF = _resolve_source_ref()


# for extension linkcode to work
# copied from numpy
def linkcode_resolve(domain, info):
    """
    Determine the URL corresponding to Python object
    """
    if domain != 'py':
        return None

    import inspect
    import cryoswath

    modname = info['module']
    fullname = info['fullname']

    submod = sys.modules.get(modname)
    if submod is None:
        return None

    obj = submod
    for part in fullname.split('.'):
        try:
            obj = getattr(obj, part)
        except Exception:
            return None

    # strip decorators, which would resolve to the source of the decorator
    # possibly an upstream bug in getsourcefile, bpo-1764286
    try:
        unwrap = inspect.unwrap
    except AttributeError:
        pass
    else:
        obj = unwrap(obj)

    fn = None
    lineno = None

    if fn is None:
        try:
            fn = inspect.getsourcefile(obj)
        except Exception:
            fn = None
        if not fn:
            return None

        # Ignore re-exports as their source files are not within the cryoswath repo
        module = inspect.getmodule(obj)
        if module is not None and not module.__name__.startswith("cryoswath"):
            return None

        try:
            source, lineno = inspect.getsourcelines(obj)
        except Exception:
            lineno = None

        fn = os.path.relpath(fn, start=os.path.dirname(cryoswath.__file__))

    if lineno:
        linespec = "#L%d-L%d" % (lineno, lineno + len(source) - 1)
    else:
        linespec = ""

    return "https://github.com/j-haacker/cryoswath/blob/%s/cryoswath/%s%s" % (
        SOURCE_CODE_GIT_REF,
        fn,
        linespec,
    )


# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'pydata_sphinx_theme'
html_theme_options = {
    "logo": {"text": f"CryoSwath v{release}"},
    "show_nav_level": 2,
    "navigation_with_keys": True,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/j-haacker/cryoswath",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        }
    ],
}

# Remove the empty "Section Navigation" panel from the left sidebar.
html_sidebars = {"**": []}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_css_files = ['custom.css']

html_baseurl = os.environ.get("READTHEDOCS_CANONICAL_URL", "/")
