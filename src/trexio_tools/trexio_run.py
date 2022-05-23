#!/usr/bin/env python3
"""
Set of tools to interact with trexio files.

Usage:
      trexio check-basis   [-n N_POINTS]  [-b BACK_END]  TREXIO_FILE
      trexio check-mos     [-n N_POINTS]  [-b BACK_END]  TREXIO_FILE
      trexio convert-to     -t TYPE -o OUTPUT_FILE       TREXIO_FILE
      trexio convert-from   -t TYPE -i INPUT_FILE  [-x MO_TYPE]  [-b BACK_END]  TREXIO_FILE
      trexio convert2champ  -i GAMESS_INPUT_FILE   [-x MO_TYPE]  [-b BACK_END]  TREXIO_FILE
      trexio (-h | --help)

Options:
      -h --help                 Print help message.
      -n --n_points=N_POINTS    Number of integration points. [default: 81]
      -i --input=INPUT_FILE     Name of the input file.
      -o --output=OUTPUT_FILE   Name of the output file.
      -b --back_end=BACK_END    [hdf5 | text]  The TREXIO back end. [default: hdf5]
      -t --type=TYPE            [gaussian | gamess | fcidump | molden | champ | cartesian | spherical ] File format.
      -x --motype=MO_TYPE       [natural | initial | guga-initial | guga-natural] The type of the molecular orbitals.
"""

from docopt import docopt
import trexio
import os


def main(filename=None, args=None):
    """Main entry point"""

    if filename is None and args is None:
        args = docopt(__doc__)
        filename = args["TREXIO_FILE"]


    if args["--n_points"] is not None:
       n_points = int(args["--n_points"])
    else:
       n_points = 81

    if args["--back_end"] is not None:
        if str(args["--back_end"]).lower() == "hdf5":
            back_end = trexio.TREXIO_HDF5
        elif str(args["--back_end"]).lower() == "text":
            back_end = trexio.TREXIO_TEXT
        else:
            raise ValueError("Supported back ends: text, hdf5.")
    else:
        back_end = trexio.TREXIO_HDF5


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

    elif args["convert2champ"]:
        from converters.trex2champ import run
        run(filename, gamessfile = args["--input"], back_end=back_end, motype=args["--motype"])

    elif args["convert-from"]:
        from converters.convert_from import run
        run(args["TREXIO_FILE"], args["--input"], args["--type"], back_end=back_end, motype=args["--motype"])

    elif args["convert-to"]:
        from converters.convert_to import run
        run(args["TREXIO_FILE"], args["--output"], args["--type"])

    else:
        pass



if __name__ == '__main__':
    args = docopt(__doc__)
    filename = args["TREXIO_FILE"]
    main(filename, args)


