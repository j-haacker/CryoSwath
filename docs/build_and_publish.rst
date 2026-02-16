Build and Publish Docs
======================

Read the Docs configuration
---------------------------

Documentation publishing is controlled by ``.readthedocs.yaml`` at the
repository root:

- It defines the build environment
- It points to the Sphinx configuration in ``docs/conf.py``
- It uses ``environment.yml`` for dependency resolution

Sphinx project configuration
----------------------------

``docs/conf.py`` configures:

- API autodoc via ``sphinx.ext.autodoc`` and ``sphinx.ext.napoleon``
- GitHub source links via ``sphinx.ext.linkcode``
- HTML theme via ``pydata_sphinx_theme``
- Source-link git ref resolution with fallbacks:
  ``READTHEDOCS_GIT_COMMIT_HASH`` -> ``git rev-parse HEAD`` ->
  ``READTHEDOCS_GIT_IDENTIFIER`` -> ``main``

The documentation imports local package sources by adding the project
root to ``sys.path``.

Build locally
-------------

Recommended (Pixi-managed, lockfile-backed):

.. code-block:: sh

   pixi install --locked -e docs
   pixi run -e docs docs-build

Alternative (pip-based local build):

.. code-block:: sh

   pip install -r docs/requirements.txt
   pip install --editable .
   make -C docs html

Or directly with Sphinx:

.. code-block:: sh

   sphinx-build -b html docs docs/_build/html

The built pages are written to ``docs/_build/html``.
