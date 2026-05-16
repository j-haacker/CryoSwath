# Contributing

Use a dedicated environment for CryoSwath development. The recommended local
workflow uses `pixi`, which keeps the runtime, test, and documentation
dependencies aligned with the checked-in lock file.

## Setup

Install the test/development environment:

```sh
pixi install -e test
```

Install the pre-commit hooks:

```sh
pixi run -e test pre-commit install
```

## Checks

Before submitting changes, run the focused checks that match your patch:

```sh
pixi run -e test pre-commit-validate
pixi run -e test lint
pixi run -e test format-check
pixi run -e test test-unit
```

For package-facing changes, also run the installed-package test. It builds the
wheel, installs it into a temporary environment outside the repository, and runs
the unit suite without importing CryoSwath from the source checkout:

```sh
pixi run -e test test-installed
```

For notebook-facing changes, run the relevant notebook task. These tasks create
isolated test project directories under `tests/*/artifacts/project`, set
`CRYOSWATH_CONFIG`, and fetch the auxiliary-data baseline when needed:

```sh
pixi run -e test test-notebooks
pixi run -e test test-tutorial-notebooks
```

Before a release or after dependency-sensitive changes, run the slow fresh
environment check. It copies the current tracked worktree, installs the locked
Pixi test environment in a temporary checkout, and runs `test-all` with a fresh
home directory:

```sh
pixi run -e test test-fresh
```

For a release-style check that ignores uncommitted tracked changes and tests
committed `HEAD`, run:

```sh
pixi run -e test test-fresh-committed
```

To run selected GitHub Actions jobs locally, use the Pixi `ci` environment.
These commands require Docker or a compatible container runtime and are an
approximation of hosted Ubuntu CI:

```sh
pixi run -e ci local-ci-pixi-test
pixi run -e ci local-ci-docs
pixi run -e ci local-ci-dependency-matrix
```

Ruff is the intended linting and formatting tool for Python files and
notebooks.

## Guidelines

Keep changes focused and avoid broad refactors unless they are necessary for
the behavior being changed. Add or update tests for behavior changes, and
update documentation when user-facing setup or workflows change.

Do not commit credentials, local configuration with secrets, downloaded
auxiliary data, or generated cache/build artifacts.
