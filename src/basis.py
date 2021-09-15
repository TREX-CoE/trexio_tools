#!/usr/bin/env python3

import trexio
import numpy as np
import nucleus

def read(trexio_file):
    r = {}

    r["nucleus"]            =  nucleus.read(trexio_file)

    r["type"]               =  trexio.read_basis_type(trexio_file)
    r["num"]                =  trexio.read_basis_num(trexio_file)
    r["prim_num"]           =  trexio.read_basis_prim_num(trexio_file)
    r["nucleus_index"]      =  trexio.read_basis_nucleus_index(trexio_file)
    r["nucleus_shell_num"]  =  trexio.read_basis_nucleus_shell_num(trexio_file)
    r["shell_ang_mom"]      =  trexio.read_basis_shell_ang_mom(trexio_file)
    r["shell_prim_num"]     =  trexio.read_basis_shell_prim_num(trexio_file)
    r["shell_factor"]       =  trexio.read_basis_shell_factor(trexio_file)
    r["shell_prim_index"]   =  trexio.read_basis_shell_prim_index(trexio_file)
    r["exponent"]           =  trexio.read_basis_exponent(trexio_file)
    r["coefficient"]        =  trexio.read_basis_coefficient(trexio_file)
    r["prim_factor"]        =  trexio.read_basis_prim_factor(trexio_file)

    return r


