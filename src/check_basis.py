#!/usr/bin/env python3

import trexio
import numpy as np

def debug(s):
    print("\n", s, "\n")

def read_nucleus(trexio_file):
    r = {}

    r["num"] =  trexio.read_nucleus_num(trexio_file)
    r["charge"] =  trexio.read_nucleus_charge(trexio_file, dim=r["num"])
    r["coord"] =  np.reshape(trexio.read_nucleus_coord(trexio_file, dim=r["num"]*3), (r["num"],3))
    r["label"] =  trexio.read_nucleus_label(trexio_file, dim=r["num"])

    return r

def read_basis(trexio_file):
    r = {}

    nucleus = read_nucleus(trexio_file)
    r["type"]               =  trexio.read_basis_type(trexio_file)
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


def basis_value(r, nucleus, basis):
    """
    Evaluates all the radial parts of the basis functions at R=(x,y,z)
    """

    coord              =  nucleus["coord"]
    nucleus_num        =  nucleus["num"]

    basis_num          =  basis["num"]
    prim_num           =  basis["prim_num"]
    nucleus_shell_num  =  basis["nucleus_shell_num"]
    nucleus_index      =  basis["nucleus_index"]
    shell_prim_num     =  basis["shell_prim_num"]
    shell_prim_index   =  basis["shell_prim_index"]

    coefficient = basis["coefficient"] * basis["prim_factor"]
    exponent    = basis["exponent"]

    # Compute all primitives
    prims = np.zeros(prim_num)
    for i_nucl in range(nucleus_num):
       dr = r - coord[i_nucl]
       r2 = np.dot(dr,dr)

       i_shell = nucleus_index[i_nucl]
       i_prim = shell_prim_index[i_shell]
       istart = i_prim

       try:
         i_shell = nucleus_index[i_nucl+1]
         i_prim = shell_prim_index[i_shell]
         iend = i_prim
       except IndexError:
         iend = prim_num+1

       expo_r = exponent[istart:iend] * r2
       prims[istart:iend] = coefficient[istart:iend] * np.exp(-expo_r)

    # Compute contractions
    result = np.zeros(basis_num)
    for i_nucl in range(nucleus_num):
       for i in range(nucleus_shell_num[i_nucl]):
          i_shell = nucleus_index[i_nucl] + i
          n_prim = shell_prim_num[i_shell]
          i_prim = shell_prim_index[i_shell]
          result[i_shell] = sum(prims[i_prim:i_prim+n_prim])

    return result

def run(trexio_file):
    """
    Computes numerically the overlap matrix in the AO basis and compares it to
    the matrix stored in the file.
    """

    nucleus = read_nucleus(trexio_file)
    debug(nucleus)

    basis = read_basis(trexio_file)
    debug(basis)
#    assert basis["type"] == "Gaussian"
#
    x = basis_value( np.array([1.,1.,1.]), nucleus, basis)
    print(x)


