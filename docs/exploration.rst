Development and exploration
===========================

CryoSwath production functions sometimes expose opt-in diagnostic hooks.
These hooks are meant for development, improvement, exploration,
introspection, validation, and, when something behaves unexpectedly,
debugging.

The hook pattern keeps the numerical code quiet by default. Core routines do
not import plotting libraries or external workflow helpers. Instead, callers
can pass a small callback when they want to inspect intermediate state:

.. code-block:: python

   events = []

   def capture_event(name, payload):
       events.append((name, payload))

   filled = cryoswath.misc.interpolate_hypsometrically(
       ds,
       "_median",
       "_iqr",
       diagnostic_hook=capture_event,
   )

The callback receives a stable event name and a payload with the objects that
were already computed at that stage. For example, hypsometric interpolation
can emit events such as ``hypsometry.model_preview`` and
``hypsometry.fit_fill_mask``. Plotting helpers in
``cryoswath.test_plots.hypsometry`` can turn those payloads into quick-look
figures.

Lab environment
---------------

For interactive exploration with the optional ``snippets`` helpers, use the
Pixi ``lab`` environment:

.. code-block:: sh

   pixi install -e lab
   pixi shell -e lab

The environment adds an IPython kernel and the ``snippets`` package. It does
not install a notebook server; open notebooks from your editor or a Jupyter
frontend of your choice.

If you manage environments with conda or mamba, create the environment there
and install the lab extra with pip:

.. code-block:: sh

   mamba create -n cryoswath-lab python=3.12
   mamba activate cryoswath-lab
   pip install "cryoswath[lab]"

Writing artifacts
-----------------

The optional ``snippets.debugging.Diagnostics`` helper is useful in external
scripts and exploratory notebooks when you want selected plots or reports
written to disk:

.. code-block:: python

   from snippets.debugging import Diagnostics
   from cryoswath.test_plots import hypsometry

   diagnostics = Diagnostics(
       enabled=True,
       outdir="debug/hypsometry",
       only={"hypsometry.fit_fill_mask"},
   )

   def write_diagnostic(name, payload):
       def write(path):
           result = hypsometry.plot_event(name, payload)
           fig = result if hasattr(result, "savefig") else result.figure
           fig.savefig(path / "plot.png", dpi=150, bbox_inches="tight")

       diagnostics.emit(name, write)

CryoSwath itself does not depend on ``snippets``. This keeps production
workflows dependency-light while still making rich introspection available to
home-grown workflows.

Tutorial
--------

The packaged notebook
``tutorial__diagnostic_hooks.ipynb`` walks through the same workflow with
synthetic data. Copy it into a project with:

.. code-block:: sh

   cryoswath get-tutorials

Then open ``tutorials/tutorial__diagnostic_hooks.ipynb`` with the ``lab``
kernel or with any environment that has CryoSwath installed.
