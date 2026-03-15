# Gravitational Lensing

![PyPI version](https://img.shields.io/pypi/v/GraviLens.svg)

Functions for 3d and 2d graphical representation and calculations for Mp-waves

* Created by **[David Johann](https://github.com/GitArrenDave)**
  * PyPI: https://pypi.org/user/none/
* PyPI package: https://pypi.org/project/GraviLens/
* Free software: MIT License

## Features

* TODO

## Documentation

Documentation is built with [Zensical](https://zensical.org/) and deployed to GitHub Pages.

* **Live site:** https://GitArrenDave.github.io/GraviLens/
* **Preview locally:** `just docs-serve` (serves at http://localhost:8000)
* **Build:** `just docs-build`

API documentation is auto-generated from docstrings using [mkdocstrings](https://mkdocstrings.github.io/).

Docs deploy automatically on push to `main` via GitHub Actions. To enable this, go to your repo's Settings > Pages and set the source to **GitHub Actions**.

## Development

To set up for local development:

```bash
# Clone your fork
git clone git@github.com:your_username/GraviLens.git
cd GraviLens

# Install in editable mode with live updates
uv tool install --editable .
```

This installs the CLI globally but with live updates - any changes you make to the source code are immediately available when you run `GraviLens`.

Run tests:

```bash
uv run pytest
```

Run quality checks (format, lint, type check, test):

```bash
just qa
```

## Author

Gravitational Lensing was created in 2026 by David Johann.

Built with [Cookiecutter](https://github.com/cookiecutter/cookiecutter) and the [audreyfeldroy/cookiecutter-pypackage](https://github.com/audreyfeldroy/cookiecutter-pypackage) project template.
