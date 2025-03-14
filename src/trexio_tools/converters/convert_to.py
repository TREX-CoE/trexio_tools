#!/usr/bin/env python3
"""
convert output of GAMESS/GAU$$IAN to trexio
"""

import sys
import os
from functools import reduce
from . import cart_sphe as cart_sphe
import numpy as np

import trexio

"""
Converter from trexio to fcidump
Symmetry labels are not included

Written by Johannes Günzl, TU Dresden 2023
"""
def run_fcidump(trexfile, filename, spin_order):
    # The Fortran implementation takes i_start and i_end as arguments; here,
    # whether an orbital is active is taken from the field in the file
    # Active orbitals are carried over as-is, core orbitals are incorporated
    # into the integrals; all others are deleted

    if not spin_order in ["block", "interleave"]:
        raise ValueError("Supported spin_order options: block (default), interleave")

    with open(filename, "w") as ofile:
        if not trexio.has_mo_num(trexfile):
            raise Exception("The provided trexio file does not include "\
                            "the number of molecular orbitals.")
        mo_num = trexio.read_mo_num(trexfile)
        if not trexio.has_electron_num(trexfile):
            raise Exception("The provided trexio file does not include "\
                            "the number of electrons.")
        elec_num = trexio.read_electron_num(trexfile)

        occupation = 2
        ms2 = 0
        if trexio.has_electron_up_num(trexfile) and trexio.has_electron_dn_num(trexfile):
            ms2 = trexio.read_electron_up_num(trexfile) \
                    - trexio.read_electron_dn_num(trexfile)
            if ms2 != 0:
                occupation = 1

        spins = None
        # Used to check the order of spins in UHF files
        if ms2 != 0 and trexio.has_mo_spin(trexfile):
            spins = trexio.read_mo_spin(trexfile)

        if trexio.has_mo_class(trexfile):
            n_act = 0
            n_core = 0
            classes = trexio.read_mo_class(trexfile)
            # Id of an active orbital among the other active orbs
            # -1 means core, -2 means deleted
            orb_ids = np.zeros(len(classes), dtype=int)
            act_ids = []
            for i, c in enumerate(classes):
                if c.lower() == "active":
                    orb_ids[i] = n_act
                    act_ids.append(i)
                    n_act += 1
                elif c.lower() == "core":
                    orb_ids[i] = -1
                    n_core += 1
                else:
                    orb_ids[i] = -2
        else:
            # Consider everything active
            n_act = mo_num
            n_core = 0
            orb_ids = np.array([i for i in range(n_act)])
            act_ids = orb_ids

        if n_core != 0 and ms2 != 0:
            raise Exception("Core orbitals are not supported for spin polarized systems")

        # Write header
        print("&FCI", file=ofile, end = " ")
        print(f"NORB={n_act},", file=ofile, end = " ")
        print(f"NELEC={elec_num - occupation*n_core},", file=ofile, end = " ")
        print(f"MS2={ms2},", file=ofile)
        print("ORBSYM=", end="", file=ofile)
        for i in range(n_act):
            # The symmetry formats between trexio and FCIDUMP differ, so this
            # information is not carried over automatically
            print("1,", end="", file=ofile)
        print("\nISYM=1,", end="", file=ofile)
        if ms2 != 0:
            print("\nUHF=.TRUE.,", file=ofile)
        print("\n&END", file=ofile)

        # Can be used to switch up the indices of printed integrals if necessary
        out_index = np.array([i+1 for i in range(n_act)])

        # If the orbitals are spin-dependent, the order alpha-beta-alpha-beta...
        # should be used for ouput (as it is expected e.g. by NECI)
        if not spins is None and not np.all(spins == spins[0]):
            current_spin_order = "none"
            # Check if the current order is alpha-alpha...beta-beta
            up = spins[0]
            for n, spin in enumerate(spins):
                # Check whether first half of orbitals is up, second is down
                if not (n < len(spins) // 2 and spin == up \
                        or n >= len(spins) // 2 and spin != up):
                    break
                current_spin_order = "block"

            if current_spin_order == "none":
                for n, spin in enumerate(spins):
                    if not (n % 2 == 0 and spin == up \
                            or n % 2 == 1 and spin != up):
                        break
                    current_spin_order = "interleave"

            if current_spin_order == "none":
                print("WARNING: Spin order within the TREXIO file was not recognized.", \
                        "The order will be kept as-is.")

            if not current_spin_order == spin_order:
                print("WARNING: The order of spin orbitals will be changed as requested.", \
                        "This might break compatibility with other data in the TREXIO file, " \
                        "e.g. CI determinant information")
                if current_spin_order == "block" and spin_order == "interleave":
                    # The (1-n_act) term sets back the beta orbitals by the number of alpha orbitals
                    out_index = np.array([2*i + (1 - n_act)*(i // (n_act // 2)) + 1 for i in range(n_act)])
                elif current_spin_order == "interleave" and spin_order == "block":
                    out_index = np.array((i%2)*(n_act // 2) + i // 2 + 1 for i in range(n_act))

        fcidump_threshold = 1e-10
        int3 = np.zeros((n_act, n_act, 2), dtype=float)
        int2 = np.zeros((n_act, n_act, 2), dtype=float)

        # Two electron integrals
        offset = 0
        buffer_size = 1000
        integrals_eof = False
        if trexio.has_mo_2e_int_eri(trexfile):
            while not integrals_eof:
                indices, vals, read_integrals, integrals_eof \
                    = trexio.read_mo_2e_int_eri(trexfile, offset, buffer_size)
                offset += read_integrals

                for integral in range(read_integrals):
                    val = vals[integral]

                    if np.abs(val) < fcidump_threshold:
                        continue
                    ind = indices[integral]
                    ii = ind[0]
                    jj = ind[1]
                    kk = ind[2]
                    ll = ind[3]
                    act_ind = [orb_ids[x] for x in ind]
                    i = act_ind[0]
                    j = act_ind[1]
                    k = act_ind[2]
                    l = act_ind[3]

                    if i >= 0 and j >= 0 and k >= 0 and l >= 0:
                        # Convert from dirac to chemists' notation
                        print(val, out_index[i], out_index[k], out_index[j], out_index[l], file=ofile)

                    # Since the integrals are added, the multiplicity needs to be screened
                    if not (ii >= kk and ii >= jj and ii >= ll and jj >= ll and (ii != jj or ll >= kk)):
                        continue

                    if i >= 0 and k >= 0 and j == -1 and jj == ll:
                        int3[i, k, 0] += val
                        if i != k:
                            int3[k, i, 0] += val

                    if i >= 0 and l >= 0 and j == -1 and jj == kk:
                        int3[i, l, 1] += val
                        if i != l:
                            int3[l, i, 1] += val
                    elif i >= 0 and j >= 0 and l == -1 and ll == kk:
                        int3[i, j, 1] += val
                        if i != j:
                            int3[j, i, 1] += val

                    if j >= 0 and l >= 0 and i == -1 and ii == kk:
                        int3[j, l, 0] += val
                        if j != l:
                            int3[l, j, 0] += val

                    if j >= 0 and k >= 0 and i == -1 and ii == ll:
                        int3[j, k, 1] += val
                        if j != k:
                            int3[k, j, 1] += val
                    elif l >= 0 and k >= 0 and i == -1 and ii == jj:
                        int3[l, k, 1] += val
                        if l != k:
                            int3[k, l, 1] += val

                    if i == -1 and ii == kk and j == -1 and jj == ll:
                        int2[ii, jj, 0] = val
                        int2[jj, ii, 0] = val

                    if i == -1 and ii == ll and j == -1 and jj == kk:
                        int2[ii, jj, 1] = val
                        int2[jj, ii, 1] = val
                    if i == -1 and ii == jj and k == -1 and kk == ll:
                        int2[ii, kk, 1] = val
                        int2[kk, ii, 1] = val

        # Hamiltonian
        if trexio.has_mo_1e_int_core_hamiltonian(trexfile):
            core_ham = trexio.read_mo_1e_int_core_hamiltonian(trexfile)
            # Add core Fock operator
            for j in range(n_act):
                jj = act_ids[j]
                for i in range(n_act):
                    ii = act_ids[i]
                    int3[i, j, 0] = core_ham[ii, jj] + occupation*int3[i, j, 0] - int3[i, j, 1]

            for a in range(n_act):
                for b in range(a, n_act):
                    val = int3[a, b, 0]
                    if np.abs(val) > fcidump_threshold:
                        print(val, out_index[b], out_index[a], 0, 0, file=ofile)

        # Core energy
        if trexio.has_nucleus_repulsion(trexfile):
            core = trexio.read_nucleus_repulsion(trexfile)
            for i in range(mo_num):
                if orb_ids[i] == -1:
                    core += occupation*core_ham[i, i]
                    for j in range(mo_num):
                        if orb_ids[j] == -1:
                            core += occupation*int2[i, j, 0] - int2[i, j, 1]

            print(core, 0, 0, 0, 0, file=ofile)

def run_molden(t, filename):

    out = ["[Molden Format]"]
    out += ["Converted from TREXIO"]

    out += ["[Atoms] AU"]

    nucl_num = trexio.read_nucleus_num(t)
    charge = trexio.read_nucleus_charge(t)
    if trexio.has_ecp_z_core(t):
       z_core = trexio.read_ecp_z_core(t)
       charge = [ x + y for x,y in zip(charge,z_core) ]
    name   = trexio.read_nucleus_label(t)
    coord  = trexio.read_nucleus_coord(t)
    for i in range(nucl_num):
        out += [ "%3s %4d %4d %18.14f %18.14f %18.14f"%tuple(
                 [name[i], i+1, int(charge[i])] + list(coord[i])  ) ]



    basis_type = trexio.read_basis_type(t)
    if basis_type.lower() == "gaussian":

        out += ["[GTO]"]
        prim_num = trexio.read_basis_prim_num(t)
        shell_num = trexio.read_basis_shell_num(t)
        nucleus_index = trexio.read_basis_nucleus_index(t)
        shell_ang_mom = trexio.read_basis_shell_ang_mom(t)
        shell_factor = trexio.read_basis_shell_factor(t)
        shell_index = trexio.read_basis_shell_index(t)
        exponent = trexio.read_basis_exponent(t)
        coefficient = trexio.read_basis_coefficient(t)
        prim_factor = trexio.read_basis_prim_factor(t)

        # For Gaussian basis sets, basis_r_power is zero by default
        if trexio.has_basis_r_power(t):
            basis_r_power = trexio.read_basis_r_power(t)
        else:
            basis_r_power = [0.0 for _ in range(basis_shell_num) ]

        contr = [ { "exponent"      : [],
                    "coefficient"   : [],
                    "prim_factor"   : []  }  for _ in range(shell_num) ]
        for j in range(prim_num):
            i = shell_index[j]
            contr[i]["exponent"]    += [ exponent[j] ]
            contr[i]["coefficient"] += [ coefficient[j] ]
            contr[i]["prim_factor"] += [ prim_factor[j] ]

        basis = {}
        for k in range(nucl_num):
            basis[k] = { "shell_ang_mom" : [],
                        "shell_factor"  : [],
                        "shell_index"   : [],
                        "contr"         : [] }

        for i in range(shell_num):
            k = nucleus_index[i]
            basis[k]["shell_ang_mom"] += [ shell_ang_mom[i] ]
            basis[k]["shell_factor"]  += [ shell_factor[i] ]
            basis[k]["shell_index"]   += [ shell_index[i] ]
            basis[k]["contr"]         += [ contr[i] ]


        ang_mom_conv = [ "s", "p", "d", "f", "g" ]
        for k in range(nucl_num):
          out += [ "%6d  0"%(k+1) ]
          for l in range(len(basis[k]["shell_index"])):
              ncontr = len(basis[k]["contr"][l]["exponent"])
              out += [ "%2s %8d 1.00" % (
                        ang_mom_conv[ basis[k]["shell_ang_mom"][l] ],
                        ncontr) ]
              for j in range(ncontr):
                out += [ "%20.10E    %20.10E"%(
                          basis[k]["contr"][l]["exponent"][j],
                          basis[k]["contr"][l]["coefficient"][j] ) ]
          out += [""]

    # end if basis_type.lower() == "gaussian"



#   5D: D 0, D+1, D-1, D+2, D-2
#   6D: xx, yy, zz, xy, xz, yz
#
#   7F: F 0, F+1, F-1, F+2, F-2, F+3, F-3
#  10F: xxx, yyy, zzz, xyy, xxy, xxz, xzz, yzz, yyz, xyz
#
#   9G: G 0, G+1, G-1, G+2, G-2, G+3, G-3, G+4, G-4
#  15G: xxxx yyyy zzzz xxxy xxxz yyyx yyyz zzzx zzzy,
#       xxyy xxzz yyzz xxyz yyxz zzxy

    mo_num = trexio.read_mo_num(t)
    cartesian = trexio.read_ao_cartesian(t)
    if cartesian:
      order = [ [0],
                [0, 1, 2],
                [0, 3, 5, 2, 3, 4],
                [0, 6, 9, 3, 1, 2, 5, 8, 7, 4],
                [0, 10, 14, 1, 2, 6, 11, 9, 13, 3, 5, 12, 4, 7, 8] ]
    else:
       out += [ "[5D]", "[7F]", "[9G]" ]
       order = [ [0], [1, 2, 0],
                 [ i for i in range(5) ],
                 [ i for i in range(7) ],
                 [ i for i in range(9) ] ]

    ao_num = trexio.read_ao_num(t)
    o = []
    icount = 0
    for i in range(shell_num):
       l = shell_ang_mom[i]
       for k in order[l]:
          o.append( icount+k )
       icount += len(order[l])


    elec_alpha_num = trexio.read_electron_up_num(t)
    elec_beta_num  = trexio.read_electron_dn_num(t)
    if trexio.has_mo_occupation(t):
       occ = trexio.read_mo_occupation(t)
    else:
       occ = [ 0. for i in range(mo_num) ]
       for i in range(elec_alpha_num):
         occ[i] += 1.
       for i in range(elec_beta_num):
         occ[i] += 1.

    mo_coef = trexio.read_mo_coefficient(t)
    if trexio.has_mo_symmetry(t):
      sym = trexio.read_mo_symmetry(t)
    else:
      sym = [ "A1" for _ in range(mo_num) ]


    out += ["[MO]"]

    for i in range(mo_num):
       out += [ "Sym= %s"%(sym[i]),
#               "Ene= 0.0",
                "Spin= Alpha",
                "Occup= %f"%(occ[i]) ]
       for k in range(ao_num):
           out += [ "%6d  %20.10E"%(k+1, mo_coef[i,o[k]]) ]

    if trexio.has_ecp_z_core(t):
       out += [ "[CORE]" ]
       for i,k in enumerate(z_core):
           out += [ "%4d : %4d"%(i, k) ]

    out += [ "" ]
    out_file = open(filename,"w")
    out_file.write("\n".join(out))

    return




def run_cart_phe(inp, filename, to_cartesian):
    out = trexio.File(filename, 'u', inp.back_end)

    shell_ang_mom = trexio.read_basis_shell_ang_mom(inp)

    # Build transformation matrix
    count_sphe = 0
    count_cart = 0
    accu = []
    shell = []
    # This code iterates over all shells, and should do it in the order they are in ao_shells
    ao_shell = trexio.read_ao_shell(inp)
    i = 0
    while i < len(ao_shell):
      she = ao_shell[i]
      l = shell_ang_mom[she]
      p, r = count_cart, count_sphe
      (x,y) = cart_sphe.data[l].shape
      count_cart += x
      count_sphe += y
      q, s = count_cart, count_sphe
      accu.append( (l, p,q, r,s) )
      if to_cartesian == 1:
          n = x
          i += y
      elif to_cartesian == 0:
          n = y
          i += x
      for _ in range(n):
          shell.append(she)

    cart_normalization = np.ones(count_cart)
    R = np.zeros( (count_cart, count_sphe) )
    for (l, p,q, r,s) in accu:
      R[p:q,r:s] = cart_sphe.data[l]
      cart_normalization[p:q] = cart_sphe.normalization[l]

    ao_num_in  = trexio.read_ao_num(inp)

    normalization = np.array( [ 1. ] * ao_num_in )
    if trexio.has_ao_normalization(inp):
      normalization = trexio.read_ao_normalization(inp)

    for i,f in enumerate(normalization):
      cart_normalization[i] *= f

    if to_cartesian == 1:  # sphe -> cart
        S = np.zeros((count_cart, count_cart))
        S = R.T @ R
        S_inv = np.linalg.inv(S)

        ao_num_out = count_cart
        R0 = R @ S_inv
        R1 = R
        R = R1

    elif to_cartesian == 0:   # cart -> sphe
        S = np.zeros((count_cart, count_cart))
        S = R @ R.T
        S_inv = np.linalg.pinv(S)

        ao_num_out = count_sphe
        R0 = R.T
        R1 = R.T @ S_inv
        R = R1
        cart_normalization = np.array([1. for _ in range(count_sphe)])


    elif to_cartesian == -1:
        R = np.eye(ao_num_in)

    # Update AOs
    trexio.write_ao_cartesian(out, to_cartesian)
    trexio.write_ao_num(out, ao_num_out)
    trexio.write_ao_shell(out, shell)

    trexio.write_ao_normalization(out, cart_normalization)

    basis_type = trexio.read_basis_type(inp)
    if basis_type.lower() == "numerical":
        """
        Although d_z^2 is the reference for both sphe and cart,
        the actual definition of said orbital is different -> shell_factor must be adapted
        """
        if trexio.has_basis_shell_factor(inp) and trexio.has_basis_shell_ang_mom(inp):
            shell_fac = trexio.read_basis_shell_factor(inp)
            l = trexio.read_basis_shell_ang_mom(inp)

            for i in range(len(shell_fac)):
                if l[i] == 2 or l[i] == 3:
                    shell_fac[i] *= 2
                elif l[i] == 4 or l[i] == 5:
                    shell_fac[i] *= 8
                elif l[i] == 4 or l[i] == 5:
                    shell_fac[i] *= 8
                elif l[i] == 6 or l[i] == 7:
                    shell_fac[i] *= 16
            trexio.write_basis_shell_factor(out, shell_fac)

        """
        If spherical harmonics are used for the angular part, radial and angular
        coordinates are completely seperated. The cartesian polynomials, however,
        mix radial and angular coordinates. Thus, r_power needs to be adapted to cancel
        out the radial dependence of the polynomials.
        """

        r_power = [0.0 for _ in shell_ang_mom ]

        r_power_sign = -1
        if to_cartesian == 0:
            r_power_sign = +1
        for i, ang_mom in enumerate(shell_ang_mom):
            r_power[i] = ang_mom * r_power_sign

        trexio.write_basis_r_power(out, r_power)

    # Update MOs
    if trexio.has_mo_coefficient(inp):
      X = trexio.read_mo_coefficient(inp)
      Y = X @ R0.T
      trexio.write_mo_coefficient(out, Y)

    # Update 1e Integrals
    if trexio.has_ao_1e_int_overlap(inp):
      X = trexio.read_ao_1e_int_overlap(inp)
      Y = R @ X @ R.T
      trexio.write_ao_1e_int_overlap(out, Y)


    if trexio.has_ao_1e_int_kinetic(inp):
      X = trexio.read_ao_1e_int_kinetic(inp)
      trexio.write_ao_1e_int_kinetic(out, R @ X @ R.T)

    if trexio.has_ao_1e_int_potential_n_e(inp):
      X = trexio.read_ao_1e_int_potential_n_e(inp)
      trexio.write_ao_1e_int_potential_n_e(out, R @ X @ R.T)

    if trexio.has_ao_1e_int_ecp(inp):
      X = trexio.read_ao_1e_int_ecp(inp)
      trexio.write_ao_1e_int_ecp(out, R @ X @ R.T)

    if trexio.has_ao_1e_int_core_hamiltonian(inp):
      X = trexio.read_ao_1e_int_core_hamiltonian(inp)
      trexio.write_ao_1e_int_core_hamiltonian(out, R @ X @ R.T)

    if trexio.has_ao_2e_int(inp) and False:
      m = ao_num_in
      n = ao_num_out
      size_max = trexio.read_ao_2e_int_eri_size(inp)

      offset = 0
      icount = size_max+1
      feof = False
      print("Reading integrals...")
      W = np.zeros( (m,m,m,m) )
      while not feof:
          buffer_index, buffer_values, icount, feof = trexio.read_ao_2e_int_eri(inp, offset, icount)
          print (icount, feof)
          offset += icount
          for p in range(icount):
              i, j, k, l = buffer_index[p]
              print (i,j,k,l)
              W[i,j,k,l] = buffer_values[p]
              W[k,j,i,l] = buffer_values[p]
              W[i,l,k,j] = buffer_values[p]
              W[k,l,i,j] = buffer_values[p]
              W[j,i,l,k] = buffer_values[p]
              W[j,k,l,i] = buffer_values[p]
              W[l,i,j,k] = buffer_values[p]
              W[l,k,j,i] = buffer_values[p]
      print("Transformation #1")
      T = W.reshape( (m, m*m*m) )
      U = T.T @ R.T
      print("Transformation #2")
      W = U.reshape( (m, m*m*n) )
      T = W.T @ R.T
      print("Transformation #3")
      U = T.reshape( (m, m*n*n) )
      W = U.T @ R.T
      print("Transformation #4")
      T = W.reshape( (m, n*n*n) )
      U = T.T @ R.T
      W = U.reshape( (n,n,n,n) )

      buffer_index = []
      buffer_values = []
      print (m, " -> ", n )
      for l in range(n):
        for k in range(n):
          for j in range(l,n):
            for i in range(k,n):
                if i==j and k<l:
                    continue
                if i<j:
                    continue
                x = W[i,j,k,l]
                if abs(x) < 1.e-12:
                    continue
                buffer_index += [i,j,k,l]
                buffer_values += [ x ]

      offset = 0
      icount =  len(buffer_values)
      trexio.write_ao_2e_int_eri(out, offset, icount, buffer_index, buffer_values)




def run_normalized_aos(t, filename):
    # Start by copying the file
    os.system('cp -r %s %s' % (t.filename, filename))
    run_cart_phe(t, filename, to_cartesian=-1)
    return


def run_cartesian(t, filename):
    # Start by copying the file
    os.system('cp -r %s %s' % (t.filename, filename))
    cartesian = trexio.read_ao_cartesian(t)
    if cartesian > 0:
        return

    run_cart_phe(t, filename, to_cartesian=1)
    return

def run_spherical(t, filename):
    # Start by copying the file
    os.system('cp -r %s %s' % (t.filename, filename))
    cartesian = trexio.read_ao_cartesian(t)
    if cartesian == 0:
        return

    run_cart_phe(t, filename, to_cartesian=0)
    return


def run(trexio_filename, filename, filetype, spin_order):

    print (filetype)
    trexio_file = trexio.File(trexio_filename,mode='r',back_end=trexio.TREXIO_AUTO)

    if filetype.lower() == "molden":
        run_molden(trexio_file, filename)
    elif filetype.lower() == "cartesian":
        run_cartesian(trexio_file, filename)
    elif filetype.lower() == "spherical":
        run_spherical(trexio_file, filename)
#    elif filetype.lower() == "normalized_aos":
#        run_normalized_aos(trexio_file, filename)
    elif filetype.lower() == "fcidump":
        run_fcidump(trexio_file, filename, spin_order)
    else:
        raise NotImplementedError(f"Conversion from TREXIO to {filetype} is not supported.")

