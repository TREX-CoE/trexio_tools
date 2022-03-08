#!/usr/bin/env python3

import trexio
import numpy as np
from . import nucleus as trexio_nucleus
from . import basis as trexio_basis
from . import ao as trexio_ao
from . import mo as trexio_mo

def run(trexio_file, n_points):
    """
    Computes numerically the overlap matrix in the AO basis and compares it to
    the matrix stored in the file.
    """

    mo = trexio_mo.read(trexio_file)
    ao = mo["ao"]
    basis = ao["basis"]
    nucleus = basis["nucleus"]
    assert basis["type"] == "Gaussian"

    rmin = np.array( list([ np.min(nucleus["coord"][:,a]) for a in range(3) ]) )
    rmax = np.array( list([ np.max(nucleus["coord"][:,a]) for a in range(3) ]) )

    shift = np.array([5.,5.,5.])
    linspace = [ None for i in range(3) ]
    step = [ None for i in range(3) ]
    for a in range(3):
      linspace[a], step[a] = np.linspace(rmin[a]-shift[a], rmax[a]+shift[a], num=n_points, retstep=True)

    print("Integration steps:", step)
    dv = step[0]*step[1]*step[2]

    mo_num = mo["num"]
    S = np.zeros( [ mo_num, mo_num ] )
    for x in linspace[0]:
      #print(".",end='',flush=True)
      for y in linspace[1]:
        for z in linspace[2]:
           chi = trexio_mo.value(mo, np.array( [x,y,z] ) )
           S += np.outer(chi, chi)*dv
    print()

    S_ex = np.eye(mo_num)
    S_diff = S - S_ex
    print ("Norm of the error: %f"%(np.linalg.norm(S_diff)))
    #print(S_diff)

    for i in range(mo_num):
      for j in range(i,mo_num):
        print("%3d %3d %15f %15f"%(i,j,S[i][j],S_ex[i,j]))



