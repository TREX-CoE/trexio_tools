# TREXIO tools

[![PyPI version](https://badge.fury.io/py/trexio-tools.svg)](https://badge.fury.io/py/trexio-tools)

Set of tools for TREXIO files.


## Requirements

- python3 (>=3.6)
- trexio (>=1.0.0) [Python API]
- numpy (>=1.17.3)
- resultsFile [optional, for GAMESS/GAU$$IAN conversion]
- VeloxChem [optional, for VLX/VeloxChem HDF5 conversion]
- docopt [for CLI]
**Note:** the pyscf<--->trexio converter has been moved to [pyscf-forge](https://github.com/pyscf/pyscf-forge). Please install the pyscf-forge plugin for the pyscf<--->trexio interface (including multi-reference wave function I/O).


## Installation

### Installation via PyPI, periodically updated

`pip install trexio-tools` 

### Installation from source code

`pip install git+https://github.com/TREX-CoE/trexio_tools`

Optional converter back ends rely on additional packages in the active Python
environment. In particular, the VLX converter requires `veloxchem`, while the
Gaussian/GAMESS converters require `resultsFile`.


## Instructions for users

After installation, `trexio-tools` provides an entry point, which can be accessed via CLI:

`trexio --help`

This will list all currently supported command line arguments. For example,

`trexio convert-from -t gamess -i data/GAMESS_CAS.log -b hdf5 trexio_cas.hdf5`

converts data from the `GAMESS_CAS.log` output file of the GAMESS code
(note also `-t gamess` argument) into the TREXIO file called `trexio_cas.hdf5`
using `-b hdf5` back end of TREXIO. 

For ORCA,

`trexio convert-from -t orca -i data/h2o.json -b hdf5 trexio_orca.hdf5`

converts data from the `h2o.json` output file of the ORCA code into the TREXIO
file called `trexio_orca.hdf5` using the `hdf5` back end of TREXIO.

Note that ORCA AOs and MOs are written in spherical coordinates. If you need a
cartesian TREXIO file, run the additional conversion step below.

`trexio convert-to -t cartesian -o trexio_orca_cart.hdf5 trexio_orca.hdf5`

This converts the spherical AOs and MOs from `trexio_orca.hdf5` into cartesian
coordinates and writes the result to `trexio_orca_cart.hdf5`.

For VeloxChem,

`trexio convert-from -w -t vlx -i /path/to/biphenyl-scf.h5 -b hdf5 biphenyl-scf-trexio.hdf5`

converts data from a VeloxChem HDF5 file into the TREXIO file called
`biphenyl-scf-trexio.hdf5` using the `hdf5` back end. The `-w` flag is useful
when rerunning the converter against an existing output file. The active Python
environment must provide the `veloxchem` package.


## VeloxChem validation

The VeloxChem converter writes and validates the following quantities when they
are present in the input HDF5 file: molecular geometry, electron counts,
Gaussian basis data, AO shell map, MO coefficients, MO energies, MO occupations,
AO overlap matrix, and SCF energy.

For local regression checks, use:

`python utilities/run_vlx_regression.py --case biphenyl=/path/to/biphenyl-scf.h5 --case tempo_roscf=/path/to/tempo-roscf.h5 --case trityl_uscf=/path/to/trityl-uscf.h5`

This script converts each VeloxChem file to TREXIO and then compares the written
TREXIO contents numerically against the source VeloxChem data using
`utilities/compare_vlx_trexio.py`.

