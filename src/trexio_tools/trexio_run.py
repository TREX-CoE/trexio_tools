#!/usr/bin/env python3
"""
Set of tools to interact with trexio files.

Usage:
      trexio check-basis      [-n N_POINTS]  [-b BACK_END]  TREXIO_FILE
      trexio check-mos        [-n N_POINTS]  [-b BACK_END]  TREXIO_FILE
      trexio convert-to       -t TYPE -o OUTPUT_FILE [-y SPIN_ORDER]  TREXIO_FILE
      trexio convert-from     -t TYPE -i INPUT_FILE  [-b BACK_END] [-S STATE_SUFFIX ] [-x MO_TYPE]  [-m MULTIPLICITY]  [-w]  TREXIO_FILE
      trexio convert-backend  -i INPUT_FILE  -o OUTPUT_FILE  -b BACK_END  -j TREX_JSON_FILE  [-s BACK_END_FROM]  [-w]
      trexio (-h | --help)

Options:
      -h, --help                    Print help message.
      -n, --n_points=N_POINTS       Number of integration points.  [default: 81]
      -i, --input=INPUT_FILE        Name of the input file.
      -o, --output=OUTPUT_FILE      Name of the output file.
      -b, --back_end=BACK_END       [hdf5 | text | auto]  The TREXIO back end.  [default: hdf5]
      -s, --back_end_from=BACK_END  [hdf5 | text | auto]  The input TREXIO back end.  [default: auto]
      -S, --state_suffix=STATE_SUFFIX  Suffix for the generated TREXIO file for a multistate calculation conversion.  [default: state]
      -m, --multiplicity=MULTIPLICITY  Spin multiplicity for the Crystal converter.
      -j, --json=TREX_JSON_FILE     TREX configuration file (in JSON format).
      -w, --overwrite               Overwrite the output TREXIO file if it already exists.  [default: True]
      -t, --type=TYPE               [gaussian | gamess | pyscf | orca | crystal | fcidump | molden | cartesian ] File format.
      -x, --motype=MO_TYPE          Type of the molecular orbitals. For example, GAMESS has RHF, MCSCF, GUGA, and Natural as possible MO types.
      -y, --spin_order=TYPE         [block | interleave] How to organize spin orbitals when converting to FCIDUMP [default: block]
"""

from docopt import docopt
import trexio
import os


def remove_trexio_file(filename:str, overwrite:bool) -> None:
    """Remove the TREXIO file/directory if it exists."""
    if os.path.exists(filename):
        if overwrite:
            # dummy check
            if '*' in filename:
                raise ValueError(f'TREXIO filename {filename} contains * symbol. Are you sure?')
            # check that the file is actually TREXIO file
            try:
                is_trexio = False
                with trexio.File(filename, 'r', trexio.TREXIO_AUTO) as tfile:
                    if trexio.has_metadata_package_version(tfile):
                        is_trexio = True
                        
                if is_trexio: os.system(f'rm -rf -- {filename}')
                
            except:
                raise Exception(f'Output file {filename} exists but it is not a TREXIO file. Are you sure?')
        else:
            raise Exception(f'Output file {filename} already exists but overwrite option is not provided. Consider using the `-w` CLI argument.')

    return


def main(filename=None, args=None) -> None:
    """Main entry point"""

    if filename is None and args is None:
        args = docopt(__doc__)
        filename = args["TREXIO_FILE"]

    n_points = int(args["--n_points"]) if args["--n_points"] else 81

    overwrite = not str(args["--overwrite"]).lower() in ['false', '0']

    if args["convert-backend"]:
        remove_trexio_file(args["--output"], overwrite)

    if args["convert-from"]:
        remove_trexio_file(args["TREXIO_FILE"], overwrite)

    if args["--back_end"]:
        if str(args["--back_end"]).lower() == "hdf5":
            back_end = trexio.TREXIO_HDF5
        elif str(args["--back_end"]).lower() == "text":
            back_end = trexio.TREXIO_TEXT
        elif str(args["--back_end"]).lower() == "auto":
            back_end = trexio.TREXIO_AUTO
        else:
            raise ValueError("Supported back ends: text, hdf5, auto.")
    else:
        if args["convert-backend"]:
            raise Exception("Missing argument for the target back end: specify --back_end or -b.")

    if args["--back_end_from"]:
        if str(args["--back_end_from"]).lower() == "hdf5":
            back_end_from = trexio.TREXIO_HDF5
        elif str(args["--back_end_from"]).lower() == "text":
            back_end_from = trexio.TREXIO_TEXT
        elif str(args["--back_end_from"]).lower() == "auto":
            back_end_from = trexio.TREXIO_AUTO
        else:
            raise ValueError("Supported options: text, hdf5, auto (default).")
    else:
        back_end_from = trexio.TREXIO_AUTO

    spin = int(args["--multiplicity"]) if args["--multiplicity"] else None

    if args["check-basis"]:
        trexio_file = trexio.File(filename, 'r', back_end=back_end)
        if trexio_file is None:
            raise IOError

        from .group_tools.check_basis import run
        run(trexio_file,n_points)

    elif args["check-mos"]:
        trexio_file = trexio.File(filename, 'r', back_end=back_end)
        if trexio_file is None:
            raise IOError

        from .group_tools.check_mos import run
        run(trexio_file,n_points)

    elif args["convert-from"]:
        from .converters.convert_from import run
        run(args["TREXIO_FILE"], args["--input"], args["--type"], back_end=back_end, spin=spin, motype=args["--motype"], state_suffix=args["--state_suffix"], 
            overwrite=args["--overwrite"])

    elif args["convert-to"]:
        from .converters.convert_to import run
        run(args["TREXIO_FILE"], args["--output"], args["--type"], args["--spin_order"])

    elif args["convert-backend"]:
        from .converters.convert_back_end import run
        run(
            args["--input"],
            args["--output"],
            back_end_to=back_end,
            back_end_from=back_end_from,
            json_filename=args["--json"]
            )

    else:
        pass


if __name__ == '__main__':
    args = docopt(__doc__)
    filename = args["TREXIO_FILE"]
    main(filename, args)
