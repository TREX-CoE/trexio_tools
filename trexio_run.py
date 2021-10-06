#!/usr/bin/env python3
"""
Set of tools to interact with trexio files.

Usage:
      trexio check-basis [-n n_points]  TREXIO_FILE             [-b back_end]
      trexio check-mos   [-n n_points]  TREXIO_FILE             [-b back_end]
      trexio convert                    TEXT_FILE TREXIO_FILE   [-b back_end]
      trexio convert2champ              TREXIO_FILE CHAMP_FILE  [-b back_end]

Options:
      -n --n_points=n     Number of integration points. Default is 81.
      -b --back_end=b     The TREXIO back end (HDF5 or TEXT). Default is HDF5. 

"""

from docopt import docopt
import trexio
import os


def main(filename, args):
    """Main entry point"""

    print("File name: %s"%filename)
    print("File exists: ", os.path.exists(filename))
    print(args)

    if args["--n_points"] is not None:
       n_points = int(args["--n_points"])
    else:
       n_points = 81

    if args["--back_end"] is not None:
        if str(args["--back_end"]) == "HDF5":
            back_end = trexio.TREXIO_HDF5
        elif str(args["--back_end"]) == "TEXT":
            back_end = trexio.TREXIO_TEXT
        else:
            raise ValueError
    else:
        back_end = trexio.TREXIO_HDF5


    if args["check-basis"]:
        trexio_file = trexio.File(filename, 'r', back_end=back_end)
        if trexio_file is None:
            raise IOError

        from src.check_basis import run
        run(trexio_file,n_points)

    elif args["check-mos"]:
        trexio_file = trexio.File(filename, 'r', back_end=back_end)
        if trexio_file is None:
            raise IOError

        from src.check_mos import run
        run(trexio_file,n_points)

    elif args["convert"]:
        from src.convert import run
        run(filename, back_end, args["TEXT_FILE"])

    elif args["convert2champ"]:
        from src.trex2champ import run
        run(filename, back_end, args["CHAMP_FILE"])

    else:
        pass



if __name__ == '__main__':
    args = docopt(__doc__)
    filename = args["TREXIO_FILE"]
    main(filename, args)


