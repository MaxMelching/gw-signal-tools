[![pipeline status](https://gitlab.aei.uni-hannover.de/fohme/gw-signal-tools/badges/main/pipeline.svg)](https://gitlab.aei.uni-hannover.de/fohme/gw-signal-tools/-/commits/main)
[![coverage report](https://gitlab.aei.uni-hannover.de/fohme/gw-signal-tools/badges/main/coverage.svg)](https://gitlab.aei.uni-hannover.de/fohme/gw-signal-tools/-/commits/main)
[![Latest Release](https://gitlab.aei.uni-hannover.de/fohme/gw-signal-tools/-/badges/release.svg)](https://gitlab.aei.uni-hannover.de/fohme/gw-signal-tools/-/releases)

# gw-signal-tools

## Description

Repository with files surrounding computations with waveforms from lal.

## Installation

First of all, install the Python version of your choice (as long as it is >= 3.9).

After that, clone the repo to some destination and then navigate there and run

```shell
pip install .
```

to install it as a Python package (adding `-e` after `install` enables editable mode).
This syntax is convenient because it allows to install options via

```shell
pip install .[option]
```

Valid options for this package are `jupyter`, `dev`.

Another possibility without the extra cloning step is to run

```shell
pip install git+https://gitlab.aei.uni-hannover.de/fohme/gw-signal-tools.git
```

or (recommended if you have a SSH key pair) running

```shell
pip install git+ssh://git@gitlab.aei.uni-hannover.de/fohme/gw-signal-tools.git
```

The last two commands also allow to install arbitrary versions of the package by adding, for example, `@v0.0.1` at the end of the prompt.

## Project Status

By now, most of the code should be in good shape, i.e. no big API changes are expected/planned at the moment.

## Change Log

Click [here](https://gitlab.aei.uni-hannover.de/fohme/gw-signal-tools/-/releases).

## Documentation

Has no domain yet, but is in progress.
