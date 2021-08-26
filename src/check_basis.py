#!/usr/bin/env python3

import trexio

def debug(s):
    print(s)

def read_nucleus(trexio_file):
    r = {}

    r["num"] =  trexio.read_nucleus_num(trexio_file)
    r["charge"] =  trexio.read_nucleus_charge(trexio_file, dim=r["num"])
    r["coord"] =  trexio.read_nucleus_coord(trexio_file, dim=r["num"])
    r["label"] =  trexio.read_nucleus_label(trexio_file, dim=r["num"])

    return r

def read_basis(trexio_file):
    r = {}

    nucleus = read_nucleus(trexio_file)
    r["basis_type"]         =  trexio.read_basis_type(trexio_file)
    r["num"]                =  trexio.read_basis_num(trexio_file)
    r["prim_num"]           =  trexio.read_basis_prim_num(trexio_file)
    r["nucleus_index"]      =  trexio.read_basis_nucleus_index(trexio_file, dim=nucleus["num"])
    r["nucleus_shell_num"]  =  trexio.read_basis_nucleus_shell_num(trexio_file, dim=nucleus["num"])
    r["shell_ang_mom"]      =  trexio.read_basis_shell_ang_mom(trexio_file, dim=r["num"])
    r["shell_prim_num"]     =  trexio.read_basis_shell_prim_num(trexio_file, dim=r["num"])
    r["shell_factor"]       =  trexio.read_basis_shell_factor(trexio_file, dim=r["num"])
    r["shell_prim_index"]   =  trexio.read_basis_shell_prim_index(trexio_file, dim=r["num"])
    r["exponent"]           =  trexio.read_basis_exponent(trexio_file, dim=r["prim_num"])
    r["coefficient"]        =  trexio.read_basis_coefficient(trexio_file, dim=r["prim_num"])
    r["prim_factor"]        =  trexio.read_basis_prim_factor(trexio_file, dim=r["prim_num"])

    return r


def run(trexio_file):
    """
    Computes numerically the overlap matrix in the AO basis and compares it to
    the matrix stored in the file.
    """

    basis = read_basis(trexio_file)
    debug(basis)

