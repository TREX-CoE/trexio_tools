#!/usr/bin/env python3
"""
Set of tools to interact with trexio files.

Usage:
      trexio check-basis [-n n_points]  TREXIO_FILE

Options:
      -n --n_points=n     Number of integration points. Default is 81.
"""

from docopt import docopt
import trexio
import os


def main(filename, args):
    """Main entry point"""

    print("File name: %s"%filename)
    print("File exists: ", os.path.exists(filename))

    trexio_file = trexio.File(filename, 'r', back_end=trexio.TREXIO_HDF5)
    if trexio_file is None:
        raise IOError

    if args["check-basis"]:
        from src.check_basis import run
        if "--n_points" in args:
           n_points = int(args["--n_points"])
        else:
           n_points = 81
        run(trexio_file,n_points)

    else:
        pass



if __name__ == '__main__':
    args = docopt(__doc__)
    filename = args["TREXIO_FILE"]
    main(filename, args)


