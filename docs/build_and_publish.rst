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

The documentation assumes CryoSwath is installed in the build environment;
the Pixi and pip commands below install it before running Sphinx.

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

PyPI publishing
---------------

Package publishing is handled by ``.github/workflows/pypi-publish.yml``.

- Creating a GitHub release automatically builds the sdist and wheel.
- The workflow checks that the release tag matches ``pyproject.toml``'s
  version (``v0.2.5`` -> ``0.2.5``).
- If the version check passes, the workflow publishes to PyPI via GitHub
  trusted publishing.

Release checklist
-----------------

1. Merge the intended release changes into ``main`` and create a release branch
   from the updated branch.
2. Move the ``Unreleased`` changelog entries under a dated release heading and
   set the same version in ``pyproject.toml``.
3. Run the dependency compatibility jobs and the release-style checks from
   committed ``HEAD``::

      pixi run -e test test-fresh-committed

4. Build and inspect the distributions locally::

      python -m build
      python -m twine check dist/*

5. Merge the release preparation after required CI passes, then create a GitHub
   release tagged ``v<version>`` from that exact commit. Publishing the release
   triggers the PyPI workflow; creating only a tag does not.
6. Confirm the new version and both wheel and source distribution on PyPI.

Conda-forge preflight and publishing
------------------------------------

Before publishing upstream or pushing a feedstock update, build the final source
distribution and prepare the version update in a personal fork of
``conda-forge/cryoswath-feedstock``. Temporarily point the local recipe at the
local source archive, rerender when required, and run the feedstock's CI image::

   python build-locally.py

Require the package build, recipe tests, output validation, and an installation
from ``build_artifacts`` to pass. Do not push the feedstock branch or open its
pull request until this local build succeeds.

After PyPI publishes, replace the temporary source with the final PyPI source
archive URL and checksum. If the published archive differs from the tested local
archive, rerun ``build-locally.py`` before pushing. The feedstock recipe must
reset its build number to ``0`` and reflect the release's Python, xarray, and
runtime dependency requirements. Once the feedstock pull request passes and is
merged, conda-forge builds and uploads the package automatically.
