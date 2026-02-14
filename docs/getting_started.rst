Getting started
===============

If you have not completed setup yet, read :doc:`prerequisites` first.

Quick checklist
---------------

.. code-block:: sh

   pip install --editable ./cryoswath
   mkdir <project_dir>
   cd <project_dir>
   cryoswath-init

Then open ``scripts/config.ini`` in your project directory and confirm
that the ``[path]`` section points to the intended ``data`` location.

.. warning::
   Use a dedicated environment for CryoSwath. Installing into a shared
   Python environment can break either CryoSwath or unrelated packages.
   Supported Python version is 3.11 or newer. Regular testing currently
   covers Python 3.11 and 3.12.

.. warning::
   Starting **Monday, February 16, 2026**, users need an
   `ESA EO account <https://eoiam-idp.eo.esa.int/>`_
   before running CryoSat download workflows.


Processing the first waveform
-----------------------------

If you are new to swath processing, it may be informative to follow the
tutorial on (processing and) viewing a single waveform. While this is
not the kind of data that you will need to analyze a glaciers evolution,
this is its foundation. Knowledge about the smallest parts of the data
can help you understand the larger picture.

Notebook:
`scripts/tutorial__process_first_waveform.ipynb
<https://github.com/j-haacker/cryoswath/blob/main/scripts/tutorial__process_first_waveform.ipynb>`_


Processing the first swath
--------------------------

Similar to viewing a single waveform, you might wonder how swath data of
a single track look like. Again, a single track will not tell you much
about a glacier's evolution, but it may help to understand higher level
data.

Notebook:
`scripts/tutorial__process_first_swath.ipynb
<https://github.com/j-haacker/cryoswath/blob/main/scripts/tutorial__process_first_swath.ipynb>`_
will give you a hand using CryoSwath to process a single track and
visualize the data.


General step-by-step tutorial
-----------------------------

The general step-by-step tutorial takes you from processing a waveform
to producing a map of elevation change trends. It balances understanding
the background processes with quickly enabling to use CryoSwath.

Notebook:
`scripts/tutorial__general_step-by-step.ipynb
<https://github.com/j-haacker/cryoswath/blob/main/scripts/tutorial__general_step-by-step.ipynb>`_.
