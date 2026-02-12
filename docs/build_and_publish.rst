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

The documentation imports local package sources by adding the project
root to ``sys.path``.

Build locally
-------------

Install docs dependencies and package:

.. code-block:: sh

   pip install -r docs/requirements.txt
   pip install --editable .

Then build:

.. code-block:: sh

   make -C docs html

Or directly with Sphinx:

.. code-block:: sh

   sphinx-build -b html docs docs/_build/html

The built pages are written to ``docs/_build/html``.
