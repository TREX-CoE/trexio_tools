#!/usr/bin/env python3
"""
Set of tools to interact with trexio files.

Usage:
      trexio check-basis      [-n N_POINTS]  [-b BACK_END]  TREXIO_FILE
      trexio check-mos        [-n N_POINTS]  [-b BACK_END]  TREXIO_FILE
      trexio convert-to       -t TYPE -o OUTPUT_FILE        TREXIO_FILE
      trexio convert-from     -t TYPE -i INPUT_FILE  [-b BACK_END]  TREXIO_FILE
      trexio convert-backend  -i INPUT_FILE  -o OUTPUT_FILE  -b BACK_END  -j TREX_JSON_FILE  [-s BACK_END_FROM]  [-w OVERWRITE]
      trexio (-h | --help)

Options:
      -h, --help                    Print help message.
      -n, --n_points=N_POINTS       Number of integration points.  [default: 81]
      -i, --input=INPUT_FILE        Name of the input file.
      -o, --output=OUTPUT_FILE      Name of the output file.
      -b, --back_end=BACK_END       [hdf5 | text]  The TREXIO back end.  [default: hdf5]
      -s, --back_end_from=BACK_END  [hdf5 | text | auto]  The input TREXIO back end.  [default: auto]
      -j, --json=TREX_JSON_FILE     TREX configuration file (in JSON format).
      -w, --overwrite=OVERWRITE     Overwrite flag for the conversion of back ends.  [default: True]
      -t, --type=TYPE               [gaussian | gamess | fcidump | molden | cartesian | spherical ] File format.
"""

from docopt import docopt
import trexio
import os


def main(filename=None, args=None):
    """Main entry point"""

    if filename is None and args is None:
        args = docopt(__doc__)
        filename = args["TREXIO_FILE"]

    if args["--n_points"]:
       n_points = int(args["--n_points"])
    else:
       n_points = 81

    if args["convert-backend"]:
        if str(args["--overwrite"]).lower() in ['false', '0']:
            overwrite = False
        else:
            overwrite = True
            print(f'File {args["--output"]} will be overwritten.')

    if args["--back_end"]:
        if str(args["--back_end"]).lower() == "hdf5":
            back_end = trexio.TREXIO_HDF5
        elif str(args["--back_end"]).lower() == "text":
            back_end = trexio.TREXIO_TEXT
        else:
            raise ValueError("Supported back ends: text, hdf5.")
    else:
        if args["convert-backend"]:
            raise Exception("Missing argument for the target back end: specify --back_end or -b.")
        else:
            back_end = trexio.TREXIO_HDF5

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


    if args["check-basis"]:
        trexio_file = trexio.File(filename, 'r', back_end=back_end)
        if trexio_file is None:
            raise IOError

        from group_tools.check_basis import run
        run(trexio_file,n_points)

    elif args["check-mos"]:
        trexio_file = trexio.File(filename, 'r', back_end=back_end)
        if trexio_file is None:
            raise IOError

        from group_tools.check_mos import run
        run(trexio_file,n_points)

    elif args["convert-from"]:
        from converters.convert_from import run
        run(args["TREXIO_FILE"], args["--input"], args["--type"], back_end=back_end)

    elif args["convert-to"]:
        from converters.convert_to import run
        run(args["TREXIO_FILE"], args["--output"], args["--type"])

    elif args["convert-backend"]:
        from converters.convert_back_end import run
        run(
            args["--input"],
            args["--output"],
            back_end_to=back_end,
            back_end_from=back_end_from,
            overwrite=overwrite,
            json_filename=args["--json"]
            )

    else:
        pass


if __name__ == '__main__':
    args = docopt(__doc__)
    filename = args["TREXIO_FILE"]
    main(filename, args)
