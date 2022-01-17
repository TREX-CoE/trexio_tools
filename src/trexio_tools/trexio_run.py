#!/usr/bin/env python3
"""
Set of tools to interact with trexio files.

Usage:
      trexio check-basis   [-n n_points]           [-b back_end]  TREXIO_FILE          
      trexio check-mos     [-n n_points]           [-b back_end]  TREXIO_FILE          
      trexio convert2champ  -i input_file          [-b back_end]  TREXIO_FILE 
      trexio convert-from   -t type -i input_file  [-b back_end]  TREXIO_FILE 
      trexio convert-to     -t type -o output_file                TREXIO_FILE 

Options:
      -n --n_points=n              Number of integration points. Default is 81.
      -i --input=input_file        Name of the input file 
      -o --output=output_file      Name of the output file 
      -b --back_end=[hdf5 | text]  The TREXIO back end. Default is hdf5.
      -t --type=[gaussian | gamess | fcidump | molden | champ]
                                   File format
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
            raise ValueError
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
        run(filename, gamessfile = args["--input"], back_end=back_end)

    elif args["convert-from"]:
        from converters.convert_from import run
        run(args["TREXIO_FILE"], args["--input"], args["--type"], back_end=back_end)

    elif args["convert-to"]:
        from converters.convert_to import run
        run(args["TREXIO_FILE"], args["--output"], args["--type"])

    else:
        pass



if __name__ == '__main__':
    args = docopt(__doc__)
    filename = args["TREXIO_FILE"]
    main(filename, args)


