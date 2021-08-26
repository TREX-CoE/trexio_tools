#!/usr/bin/env python3

def run(trexio_file):
    """
    Computes numerically the overlap matrix in the AO basis and compares it to
    the matrix stored in the file.
    """

    n = trexio.read_nucleus_num(trexio_file)
    print("Num: %d"%n)

    charges = trexio.read_nucleus_charge(trexio_file,dim=n)
    print(charges)

