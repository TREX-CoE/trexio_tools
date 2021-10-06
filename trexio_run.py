#!/usr/bin/env python3
"""
Set of tools to interact with trexio files.

Usage:
      trexio check-basis [-n n_points]  TREXIO_FILE             [-be back_end HDF5 | TEXT]
      trexio check-mos   [-n n_points]  TREXIO_FILE             [-be back_end HDF5 | TEXT]
      trexio convert                    TEXT_FILE TREXIO_FILE   [-be back_end HDF5 | TEXT]

Options:
      -n --n_points=n     Number of integration points. Default is 81.
      -be 

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

    if args["check-basis"]:
        trexio_file = trexio.File(filename, 'r', back_end=trexio.TREXIO_HDF5)
        if trexio_file is None:
            raise IOError

        from src.check_basis import run
        run(trexio_file,n_points)

    elif args["check-mos"]:
        trexio_file = trexio.File(filename, 'r', back_end=trexio.TREXIO_HDF5)
        if trexio_file is None:
            raise IOError

        from src.check_mos import run
        run(trexio_file,n_points)

    elif args["convert"]:
        from src.convert import run
        run(args["TREXIO_FILE"], args["TEXT_FILE"])

    else:
        pass



if __name__ == '__main__':
    args = docopt(__doc__)
    filename = args["TREXIO_FILE"]
    main(filename, args)


