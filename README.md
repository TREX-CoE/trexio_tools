# TREXIO tools

[![PyPI version](https://badge.fury.io/py/trexio-tools.svg)](https://badge.fury.io/py/trexio-tools)

Set of tools for TREXIO files.


## Requirements

- python3 (>=3.6)
- trexio (>=1.0.0) [Python API]
- numpy (>=1.17.3)
- resultsFile [for GAMESS/GAU$$IAN conversion]
- docopt [for CLI]
- pyscf [only if you use the pyscf->trexio converter] 

**Deprecation note:** the pyscf--->trexio converter in this repository is no longer maintained. Please install the [pyscf-forge](https://github.com/pyscf/pyscf-forge) plugin for the latest and the most complete pyscf<--->trexio interface (including multi-reference wave function I/O).


## Installation

### Installation via PyPI, periodically updated

`pip install trexio-tools` 

### Installation from source code

`pip install git+https://github.com/TREX-CoE/trexio_tools`


## Instructions for users

After installation, `trexio-tools` provides an entry point, which can be accessed via CLI:

`trexio --help`

This will list all currently supported command line arguments. For example,

`trexio convert-from -t gamess -i data/GAMESS_CAS.log -b hdf5 trexio_cas.hdf5`

converts data from the `GAMESS_CAS.log` output file of the GAMESS code
(note also `-t gamess` argument) into the TREXIO file called `trexio_cas.hdf5`
using `-b hdf5` back end of TREXIO. 

and,

`trexio convert-from -t orca -i data/h2o.json -b hdf5 trexio_orca.hdf5`

Note that since ORCA AOs and MOs are in spherical coordinates, one needs to convert
these to cartesian to be able to use `trexio` functions.

`trexio convert-to -t cartesian -o trexio_orca_cart.hdf5 trexio_orca.hdf5`

converts data from the `h2o.json` output file of the ORCA code
into the TREXIO file called `trexio_orca.hdf5` using `-b hdf5` back end of TREXIO
followed by converting the spherical AOs and MOs to cartesian coordinates.

