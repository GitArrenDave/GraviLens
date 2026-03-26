# GraviLens

![Coverage](https://raw.githubusercontent.com/GitArrenDave/GraviLens/master/.github/badges/coverage.svg)
Functions for geometric calculations and visualisation of plane waves and Mp-waves in general relativity.

Created by **[David Johann](https://github.com/GitArrenDave)**

License: MIT

Repository:
https://github.com/GitArrenDave/GraviLens


## Features

- Plane wave space-time models
- Geodesic integration in Brinkmann coordinates
- Lightcone and comoving source constructions
- Frequency shift and angle formulas
- 3D visualisation of geodesics


## Documentation

Documentation is built with **Sphinx** and deployed automatically with GitHub Pages.

Live documentation:

https://GitArrenDave.github.io/GraviLens/

Build locally:

pip install -r docs/requirements.txt
sphinx-build docs docs/_build/html
open docs/_build/html/index.html



## Installation (local development)

Clone the repository:

git clone https://github.com/GitArrenDave/GraviLens.git

cd GraviLens



# Install in editable mode with live updates
pip install -e


## Author

GraviLens was created in 2026 by David Johann.

Built using

- Python
- NumPy
- Matplotlib
- Pytest
- Sphinx
- GitHub Actions



Built with [Cookiecutter](https://github.com/cookiecutter/cookiecutter) and the [audreyfeldroy/cookiecutter-pypackage](https://github.com/audreyfeldroy/cookiecutter-pypackage) project template.
