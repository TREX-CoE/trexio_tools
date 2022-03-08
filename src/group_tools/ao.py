#!/usr/bin/env python3

import trexio
import numpy as np
from . import basis as trexio_basis

def read(trexio_file):
    r = {}

    r["basis"]     = trexio_basis.read(trexio_file)
    r["basis_old"] = trexio_basis.convert_to_old(r["basis"])
    r["num"]       = trexio.read_ao_num(trexio_file)
    r["shell"]     = trexio.read_ao_shell(trexio_file)
    r["factor"]    = trexio.read_ao_normalization(trexio_file)

    return r

powers = [
 [(0,0,0)],
 [(1,0,0), (0,1,0), (0,0,1)],
 [(2,0,0), (1,1,0), (1,0,1), (0,2,0), (0,1,1), (0,0,2)],
 [(3,0,0), (2,1,0), (2,0,1), (1,2,0), (1,1,1), (1,0,2),
  (0,3,0), (0,2,1), (0,1,2), (0,0,3)]
  ]


def value(ao,r):
    """
    Evaluates all the basis functions at R=(x,y,z)
    """

    basis          =  ao["basis"]

    nucleus        =  basis["nucleus"]
    coord          =  nucleus["coord"]
    nucleus_num    =  nucleus["num"]

    basis_num      =  basis["shell_num"]
    prim_num       =  basis["prim_num"]
    shell_ang_mom  =  basis["shell_ang_mom"]

    # to reconstruct for compatibility with TREXIO < v.2.0.0
    basis_old         = ao["basis_old"]
    nucleus_index     = basis_old["nucleus_index"]
    nucleus_shell_num = basis_old["nucleus_shell_num"]
    shell_prim_index  = basis_old["shell_prim_index"]
    shell_prim_num    = basis_old["shell_prim_num"]

    coefficient = basis["coefficient"] * basis["prim_factor"]
    exponent    = basis["exponent"]

    norm        = ao["factor"]

    # Compute all primitives and powers
    prims = np.zeros(prim_num)
    pows  = [ None for i in range(basis_num) ]

    for i_nucl in range(nucleus_num):

       i_shell = nucleus_index[i_nucl]
       i_prim = shell_prim_index[i_shell]
       istart = i_prim

       try:
         i_shell_end = nucleus_index[i_nucl+1]
         i_prim = shell_prim_index[i_shell_end]
         iend = i_prim
       except IndexError:
         iend = prim_num+1
         i_shell_end = basis_num

       dr = r - coord[i_nucl]
       r2 = dr[0]*dr[0] + dr[1]*dr[1] + dr[2]*dr[2]
       expo_r = exponent[istart:iend] * r2
       prims[istart:iend] = coefficient[istart:iend] * np.exp(-expo_r)

       old = None
       for i in range(i_shell,i_shell_end):
         if shell_ang_mom[i] != old:
            old = shell_ang_mom[i]
            x = np.array([ np.power(dr, p) for p in powers[old] ])
            x = np.prod(x,axis=1)
         pows[i] = x

    # Compute contractions
    rr = np.zeros(basis_num)
    for i_nucl in range(nucleus_num):
       for i in range(nucleus_shell_num[i_nucl]):
          i_shell = nucleus_index[i_nucl] + i
          n_prim = shell_prim_num[i_shell]
          i_prim = shell_prim_index[i_shell]
          rr[i_shell] = sum(prims[i_prim:i_prim+n_prim])

    result = np.concatenate( [ rr[i] * p for i,p in enumerate(pows) ] )

    return result * norm

