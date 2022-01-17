# TREXIO tools

[![PyPI version](https://badge.fury.io/py/trexio-tools.svg)](https://badge.fury.io/py/trexio-tools)

Set of tools for TREXIO files.


## Requirements

- python3 (>=3.6)
- trexio (>=1.0.0) [Python API]
- numpy (>=1.17.3)
- resultsFile [for GAMESS/GAU$$IAN conversion]
- docopt [for CLI]


## Installation

0. Clone the repository
- `git clone https://github.com/TREX-CoE/trexio_tools.git`
1. Create an isolated virtual environment, for example using
- `python3 -m venv trexio_tools`
2. Activate the previously created environment, for example using
- `source trexio_tools/bin/activate`
3. Install/upgrade the Python setup tools
- `pip install --upgrade setuptools wheel build`
4. Install the Python packages that are required for `trexio-tools` to work
- `pip install -r requirements.txt`
5. Install the `trexio-tools` package using one of the following methods:
- `pip install trexio-tools` (installation from PyPI, periodically updated)
- `pip install .` (installation from source, always contains recent updates)

Only the last step has to be repeated to upgrade the `trexio-tools` package.
This means that the virtual environment can stay the same, unless there have been
critical updates in `trexio` or `resultsFile` packages.


## Instructions for users

After installation, `trexio-tools` provides an entry point, which can be accessed via CLI:

`trexio --help`

This will list all currently supported command line arguments. For example,

`trexio convert-from -t gamess -i data/GAMESS_CAS.log -b hdf5 trexio_cas.hdf5`

converts data from the `GAMESS_CAS.log` output file of the GAMESS code
(note also `-t gamess` argument) into the TREXIO file called `trexio_cas.hdf5`
using `-b hdf5` back end of TREXIO. 

