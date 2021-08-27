#!/usr/bin/env python3

import trexio
import numpy as np

def read_nucleus(trexio_file):
    r = {}

    r["num"] =  trexio.read_nucleus_num(trexio_file)
    r["charge"] =  trexio.read_nucleus_charge(trexio_file)
    r["coord"] =  trexio.read_nucleus_coord(trexio_file)
    r["label"] =  trexio.read_nucleus_label(trexio_file)

    return r

def read_basis(trexio_file):
    r = {}

    nucleus = read_nucleus(trexio_file)
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


def read_ao(trexio_file):
    r = {}

    r["num"]    = trexio.read_ao_num(trexio_file)
    r["shell"]  = trexio.read_ao_shell(trexio_file)
    r["factor"] = trexio.read_ao_normalization(trexio_file)

    return r

powers = [
 [(0,0,0)],
 [(1,0,0), (0,1,0), (0,0,1)],
 [(2,0,0), (1,1,0), (1,0,1), (0,2,0), (0,1,1), (0,0,2)],
 [(3,0,0), (2,1,0), (2,0,1), (1,2,0), (1,1,1), (1,0,2),
  (0,3,0), (0,2,1), (0,1,2), (0,0,3)]
  ]


def ao_value(r, nucleus, basis):
    """
    Evaluates all the radial parts of the basis functions at R=(x,y,z)
    """

    coord              =  nucleus["coord"]
    nucleus_num        =  nucleus["num"]

    basis_num          =  basis["num"]
    prim_num           =  basis["prim_num"]
    nucleus_shell_num  =  basis["nucleus_shell_num"]
    nucleus_index      =  basis["nucleus_index"]
    shell_ang_mom      =  basis["shell_ang_mom"]
    shell_prim_num     =  basis["shell_prim_num"]
    shell_prim_index   =  basis["shell_prim_index"]

    coefficient = basis["coefficient"] * basis["prim_factor"]
    exponent    = basis["exponent"]

    # Compute all primitives and powers
    prims = np.zeros(prim_num)
    pows  = [ None for i in range(basis_num) ]

    for i_nucl in range(nucleus_num):
       dr = r - coord[i_nucl]
       r2 = np.dot(dr,dr)

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

    return result


def run(trexio_file):
    """
    Computes numerically the overlap matrix in the AO basis and compares it to
    the matrix stored in the file.
    """

    nucleus = read_nucleus(trexio_file)
    basis = read_basis(trexio_file)
    ao = read_ao(trexio_file)
    assert basis["type"] == "Gaussian"

    rmin = np.array( list([ np.min(nucleus["coord"][:,a]) for a in range(3) ]) )
    rmax = np.array( list([ np.max(nucleus["coord"][:,a]) for a in range(3) ]) )

    shift = np.array([10.,10.,10.])
    linspace = [ None for i in range(3) ]
    step = [ None for i in range(3) ]
    for a in range(3):
      linspace[a], step[a] = np.linspace(rmin[a]-shift[a], rmax[a]+shift[a], num=40, retstep=True)
    print (step)
    dv = step[0]*step[1]*step[2]

    norm = ao["factor"]
    S = np.zeros( [ ao["num"], ao["num"]] )
    for x in linspace[0]:
      print(".",end='',flush=True)
      for y in linspace[1]:
        for z in linspace[2]:
           chi = ao_value( np.array( [x,y,z] ), nucleus, basis) * norm
           S += np.outer(chi, chi)*dv
    print()

    ao_num = ao["num"]
    S_ex = trexio.read_ao_1e_int_overlap(trexio_file)

    for i in range(ao_num):
      for j in range(i,ao_num):
        print("%3d %3d %15f %15f"%(i,j,S[i][j],S_ex[i,j]))



