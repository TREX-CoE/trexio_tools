#!/usr/bin/env python3

import trexio
import numpy as np
from . import nucleus as trexio_nucleus
from . import basis as trexio_basis
from . import ao as trexio_ao
from .becke_grid import becke_grid

def run(trexio_file, n_points):
    """
    Computes numerically the overlap matrix in the AO basis and compares it to
    the matrix stored in the file.
    """

    print(trexio.read_basis_type(trexio_file))
    if trexio.read_basis_type(trexio_file) == "Numerical":
        from . import nao as trexio_ao
    else:
        from . import ao as trexio_ao
    ao = trexio_ao.read(trexio_file)
    basis = ao["basis"]
    nucleus = basis["nucleus"]
    assert basis["type"] in [ "Gaussian", "Numerical", "Slater" ]

    ao_num = ao["num"]

    # Generate Becke integration grid
    point, weights = becke_grid(nucleus["coord"], nucleus["charge"],
                                n_radial=n_points, n_angular=15)
    point_num = len(point)
    print("Number of grid points:", point_num)

    if trexio.has_ao_1e_int_overlap(trexio_file):
        S_ex = trexio.read_ao_1e_int_overlap(trexio_file)
    else:
        S_ex = np.zeros((ao_num,ao_num))


    try:
        import qmckl

        trexio_filename = trexio_file.filename
        context = qmckl.context_create()
        qmckl.trexio_read(context, trexio_filename)

        qmckl.set_point(context, 'N', point_num, np.reshape(point, (point_num*3)))
        chi = qmckl.get_ao_basis_ao_value(context, point_num*ao_num)
        chi = np.reshape( chi, (point_num,ao_num) )
        S = chi.T @ (chi * weights[:, np.newaxis])

    except ModuleNotFoundError:

        chi = []
        for xyz in point:
          chi += [ trexio_ao.value(ao, np.array(xyz)) ]

    chi = np.reshape( chi, (point_num,ao_num) )
    S = chi.T @ (chi * weights[:, np.newaxis])

    print()


    for i in range(ao_num):
      for j in range(i,ao_num):
        print("%3d %3d %15f %15f"%(i,j,S[i][j],S_ex[i,j]))
    S_diff = S - S_ex
    print("Norm of the error: %f"%(np.linalg.norm(S_diff)))

    print("Diagonal entries:")
    for i in range(ao_num):
        print("%3d %15f"%(i,S[i][i]))


