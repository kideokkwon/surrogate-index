# surrogate_index

[![PyPI - Version](https://img.shields.io/pypi/v/surrogate-index.svg)](https://pypi.org/project/surrogate-index)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/surrogate-index.svg)](https://pypi.org/project/surrogate-index)

-----

## Table of Contents

- [Installation](#installation)
- [License](#license)

## Installation

Installing the core package (no ML dependencies):
```console
pip install surrogate-index
```

To use the surrogate index estimator with XGBoost and scikit-learn:
```console
pip install "surrogate-index[ml]"
```

Although, if you're a conda-user, may be better to do:
```console
conda install -c conda-forge xgboost scikit-learn pandas numpy
pip install surrogate-index
```

To work on the codebase (includes dev tools like ruff, black, pytest, mypy):
```console
pip install -e ".[dev,ml]"
```

## License

`surrogate-index` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
