#!/usr/bin/env python3
"""
Set of tools to interact with trexio files.

Usage:
      trexio check-basis TREXIO_FILE

"""

import trexio
import os


def main(filename, args):
    """Main entry point"""

    print("File name: %s"%filename)
    print("File exists: ", os.path.exists(filename))

    trexio_file = trexio.File(filename, 'r', back_end=trexio.TREXIO_HDF5)
    if trexio_file is None:
        raise IOError

    if str(args[1])=="check-basis":
        from src.check_basis import run
        run(trexio_file)

    else:
        pass



if __name__ == '__main__':
    #from docopt import docopt
    #args = docopt(__doc__)
    #filename = args["TREXIO_FILE"]
    import sys
    args = sys.argv
    filename = str(args[-1])
    main(filename, args)


