# gw_signal_tools

## Description

Repository with files surrounding computations with waveforms from lal.

## Installation

Since there is no lal-version for Python 3.12 yet, Python 3.11 is required
to install this package.

After that, clone the repo to some destination and then navigate
there and run

```shell
pip install .
```

to install it as a Python package (adding `-e` after `install` enables editable
mode). This syntax is convenient because it allows to install options via

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

## Project Status

In progress.
