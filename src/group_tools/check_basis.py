#!/usr/bin/env python3

import trexio
import numpy as np
from . import nucleus as trexio_nucleus
from . import basis as trexio_basis
from . import ao as trexio_ao

try:
    import qmckl
    def run(trexio_file, n_points):
        """
        Computes numerically the overlap matrix in the AO basis and compares it to
        the matrix stored in the file.
        """

        if not trexio.has_ao_1e_int_overlap(trexio_file):
          raise Exception(
            "One-electron overlap integrals are missing in the TREXIO file. Required for check-basis."
            )
        trexio_filename = trexio_file.filename
        context = qmckl.context_create()
        qmckl.trexio_read(context, trexio_filename)

        ao = trexio_ao.read(trexio_file)
        basis = ao["basis"]
        nucleus = basis["nucleus"]
        assert basis["type"] == "Gaussian"

        rmin = np.array( list([ np.min(nucleus["coord"][:,a]) for a in range(3) ]) )
        rmax = np.array( list([ np.max(nucleus["coord"][:,a]) for a in range(3) ]) )

        shift = np.array([8.,8.,8.])
        linspace = [ None for i in range(3) ]
        step = [ None for i in range(3) ]
        for a in range(3):
          linspace[a], step[a] = np.linspace(rmin[a]-shift[a], rmax[a]+shift[a], num=n_points, retstep=True)

        print("Integration steps:", step)
        dv = step[0]*step[1]*step[2]

        point = []
        for x in linspace[0]:
          #print(".",end='',flush=True)
          for y in linspace[1]:
            for z in linspace[2]:
               point += [ [x, y, z] ]
        point = np.array(point)
        point_num = len(point)
        ao_num = ao["num"]

        qmckl.set_point(context, 'N', point_num, np.reshape(point, (point_num*3)))
        chi = qmckl.get_ao_basis_ao_value(context, point_num*ao_num)
        chi = np.reshape( chi, (point_num,ao_num) )
        S = chi.T @ chi * dv
        print()

        S_ex = trexio.read_ao_1e_int_overlap(trexio_file)

        # This produces a lot of output for large molecules, maybe wrap up in ``if debug`` statement ?
        for i in range(ao_num):
          for j in range(i,ao_num):
            print("%3d %3d %15f %15f"%(i,j,S[i,j],S_ex[i,j]))
        S_diff = S - S_ex
        print("Norm of the error: %f"%(np.linalg.norm(S_diff)))




except ImportError:

    def run(trexio_file, n_points):
        """
        Computes numerically the overlap matrix in the AO basis and compares it to
        the matrix stored in the file.
        """

        if not trexio.has_ao_1e_int_overlap(trexio_file):
          raise Exception(
            "One-electron overlap integrals are missing in the TREXIO file. Required for check-basis."
            )

        print(trexio.read_basis_type(trexio_file))
        if trexio.read_basis_type(trexio_file) == "Numerical":
            from . import nao as trexio_ao
        ao = trexio_ao.read(trexio_file)
        basis = ao["basis"]
        nucleus = basis["nucleus"]
        assert basis["type"] == "Gaussian" or basis["type"] == "Numerical"

        rmin = np.array( list([ np.min(nucleus["coord"][:,a]) for a in range(3) ]) )
        rmax = np.array( list([ np.max(nucleus["coord"][:,a]) for a in range(3) ]) )

        # TODO Extension of NAO is finite -> use this
        shift = np.array([8.,8.,8.])
        linspace = [ None for i in range(3) ]
        step = [ None for i in range(3) ]
        for a in range(3):
          linspace[a], step[a] = np.linspace(rmin[a]-shift[a], rmax[a]+shift[a], num=n_points, retstep=True)

        print("Integration steps:", step)
        dv = step[0]*step[1]*step[2]
        ao_num = ao["num"]

        restricted = True
        if trexio.has_electron_dn_num(trexio_file) and trexio.has_electron_up_num(trexio_file):
            if trexio.read_electron_dn_num(trexio_file) != trexio.read_electron_up_num(trexio_file):
                restricted = False

        do_mos = trexio.has_mo_num(trexio_file) and trexio.has_mo_1e_int_overlap(trexio_file) and trexio.has_mo_coefficient(trexio_file)
        if do_mos:
            mo_num = trexio.read_mo_num(trexio_file)
            S_mo = np.zeros([mo_num, mo_num])
            coeffs = trexio.read_mo_coefficient(trexio_file)
            # As of this writing, there is a bug with non-square matrices
            if coeffs.shape[1] != mo_num and coeffs.shape[0] != ao_num:
                print("Bugfix")
                #tmp = np.zeros((coeffs.shape[1], coeffs.shape[0]), coeffs.dtype)
                #for i in range(ao_num):
                #    tmp[i, :ao_num] = coeffs[2*i, :]
                #    tmp[i, ao_num:] = coeffs[2*i + 1, :]
                #coeffs = tmp
                coeffs = coeffs.reshape((ao_num, mo_num))
            coeffs = coeffs.T
            #for i in range(coeffs.shape[0]):
            #    print(i, "\t", coeffs[i])
            S_mo_ex = trexio.read_mo_1e_int_overlap(trexio_file)


        S = np.zeros( [ ao_num, ao_num ] )

        for x in linspace[0]:
          #print(".",end='',flush=True)
          for y in linspace[1]:
            for z in linspace[2]:
               chi = trexio_ao.value(ao, np.array( [x,y,z] ) )
               S += np.outer(chi, chi)*dv

               if do_mos:
                  #print("AO:", chi)
                  chi = coeffs @ chi
                  #print("MO:", chi)
                  S_mo += np.outer(chi, chi)*dv
        print()

        S_ex = trexio.read_ao_1e_int_overlap(trexio_file)

        # This produces a lot of output for large molecules, maybe wrap up in ``if debug`` statement ?
        for i in range(ao_num):
          for j in range(i,ao_num):
            print("%3d %3d %15f %15f"%(i,j,S[i][j],S_ex[i,j]))
        S_diff = S - S_ex
        print("Norm of the error: %f"%(np.linalg.norm(S_diff)))

        print("Diagonal entries:")
        for i in range(ao_num):
            print("%3d %15f"%(i,S[i][i]))

        # Idem for mos
        if do_mos:
            for i in range(mo_num):
              for j in range(i, mo_num):
                print("%3d %3d %15f %15f"%(i,j,S_mo[i][j],S_mo_ex[i,j]))
            S_diff = S_mo - S_mo_ex
            print("Norm of the error: %f"%(np.linalg.norm(S_diff)))

            print("Diagonal entries:")
            for i in range(mo_num):
                print("%3d %15f"%(i,S_mo[i][i]))



