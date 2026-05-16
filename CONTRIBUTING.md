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

Ruff is the intended linting and formatting tool for Python files and
notebooks. The existing `[flake8]` section in `tox.ini` is kept only for
manual compatibility; the current tox commands do not invoke Flake8.

## Guidelines

Keep changes focused and avoid broad refactors unless they are necessary for
the behavior being changed. Add or update tests for behavior changes, and
update documentation when user-facing setup or workflows change.

Do not commit credentials, local configuration with secrets, downloaded
auxiliary data, or generated cache/build artifacts.
