#!/usr/bin/env python3
"""
convert output of GAMESS/GAU$$IAN to trexio
"""

import sys
import os
from functools import reduce
try:
    import trexio
except:
    print("Error: The TREXIO Python library is not installed")
    sys.exit(1)

try:
    from resultsFile import *
except:
    print("Error: The resultsFile Python library is not installed")
    sys.exit(1)



def run(trexio_filename, back_end, filename):

    trexio_file = trexio.File(trexio_filename,mode='w',back_end=back_end)
    try:
        res = getFile(filename)
    except:
        raise
    else:
        print(filename, 'recognized as', str(res).split('.')[-1].split()[0])

    res.clean_uncontractions()

    # Metadata
    # --------

    trexio.write_metadata_code_num(trexio_file, 1)
    trexio.write_metadata_code(trexio_file, [str(res).split('.')[-1].split()[0].replace("File","")])
    trexio.write_metadata_author_num(trexio_file, 1)
    trexio.write_metadata_author(trexio_file, [res.author])
    trexio.write_metadata_description(trexio_file, res.title)

    # Electrons
    # ---------

    trexio.write_electron_up_num(trexio_file,res.num_alpha)
    trexio.write_electron_dn_num(trexio_file,res.num_beta)

    # Nuclei
    # ------

    charge = []
    coord = []

    for a in res.geometry:
        charge.append(a.charge)
        if res.units != 'BOHR':
            coord.append([a.coord[0] / a0, a.coord[1] / a0, a.coord[2] / a0])
        else:
            coord.append([a.coord[0], a.coord[1], a.coord[2]])

    trexio.write_nucleus_num(trexio_file, len(res.geometry))
    trexio.write_nucleus_charge(trexio_file, charge)
    trexio.write_nucleus_coord(trexio_file, coord)

    # Transformt H1 into H
    import re
    p = re.compile(r'(\d*)$')
    label = [p.sub("", x.name).capitalize() for x in res.geometry]
    trexio.write_nucleus_label(trexio_file, label)

    trexio.write_nucleus_point_group(trexio_file, res.point_group)


    # Basis

    trexio.write_basis_type(trexio_file, "Gaussian")

    basis = []
    nucleus_index = []
    nucl_shell_num = []
    shell_ang_mom = []
    shell_prim_num = []
    shell_prim_index = []
    shell_factor = []
    exponent = []
    coefficient = []
    prim_factor = []
    curr_shell = -1
    curr_shell_idx = 0
    ao_shell = []
    prev_idx = None
    geom = [ a.coord for a in res.geometry ]
    for b in res.basis:
      if "y" in b.sym or "z" in b.sym:
        pass
      else:
        basis.append(b)
        shell_ang_mom.append(b.sym.count("x"))
        curr_shell += 1
        curr_shell_idx = len(exponent)
        shell_prim_index.append(curr_shell_idx)
        shell_prim_num.append(len(b.prim))
        exponent += [x.expo for x in b.prim]
        coefficient += b.coef
        prim_factor += [1./x.norm for x in b.prim]
        idx = geom.index(b.center)
        shell_factor.append(1./b.norm)
        if idx != prev_idx:
           nucleus_index.append(len(basis)-1)
           prev_idx = idx
           if len(nucleus_index) > 1:
             nucl_shell_num.append(nucleus_index[-1]-nucleus_index[-2])
      ao_shell.append(curr_shell)

    nucl_shell_num.append(nucleus_index[-1]-nucleus_index[-2])

    trexio.write_basis_num(trexio_file,len(basis))
    trexio.write_basis_prim_num(trexio_file,len(exponent))
    trexio.write_basis_nucleus_index(trexio_file,nucleus_index)
    trexio.write_basis_nucleus_shell_num(trexio_file,nucl_shell_num)
    trexio.write_basis_shell_ang_mom(trexio_file,shell_ang_mom)
    trexio.write_basis_shell_prim_num(trexio_file,shell_prim_num)
    trexio.write_basis_shell_factor(trexio_file,shell_factor)
    trexio.write_basis_shell_prim_index(trexio_file,shell_prim_index)
    trexio.write_basis_exponent(trexio_file,exponent)
    trexio.write_basis_coefficient(trexio_file,coefficient)
    trexio.write_basis_prim_factor(trexio_file,prim_factor)


    # AO
    # --

    res.convert_to_cartesian()
    trexio.write_ao_cartesian(trexio_file, True)
    trexio.write_ao_num(trexio_file, len(res.basis))
    trexio.write_ao_shell(trexio_file, ao_shell)

    ao_ordering = []
    accu = []
    normalization = []
    prev_shell = None

    # Re-order AOs (xx,xy,xz,yy,yz,zz)
    for i,b in enumerate(res.basis):
      shell = ao_shell[i]
      if shell != prev_shell:
          accu.sort()
          ao_ordering += accu
          accu = []
      accu += [(b.sym, i)]
      prev_shell = shell
    accu.sort()
    ao_ordering += accu

    ao_ordering = [ i for (_,i) in ao_ordering ]

    # Normalization
    normalization = []
    for i,k in enumerate(ao_ordering):
      b = res.basis[k]
      orig = res.basis[ao_shell.index(ao_shell[k])]
      prim = b.prim
      prim_norm = [ j.norm for j in prim ]
      oprim = orig.prim
      oprim_norm = [ j.norm for j in oprim ]
      sum = 0.
      for i, ci in enumerate(b.coef):
         ci /= prim_norm[i]
         for j, cj in enumerate(orig.coef):
           cj /= oprim_norm[j]
           sum += ci*cj * oprim[i].overlap(oprim[j])
      sum /= orig.norm**2
      normalization.append(sum)
    trexio.write_ao_normalization(trexio_file, normalization)


    # MOs
    # ---

    MoTag = res.determinants_mo_type
    MO_type = MoTag
    allMOs = res.mo_sets[MO_type]

    trexio.write_mo_type(trexio_file, MO_type)

    try:
        closed = [(allMOs[i].eigenvalue, i) for i in res.closed_mos]
        active = [(allMOs[i].eigenvalue, i) for i in res.active_mos]
        virtual = [(allMOs[i].eigenvalue, i) for i in res.virtual_mos]
    except:
        closed = []
        virtual = []
        active = [(allMOs[i].eigenvalue, i) for i in range(len(allMOs))]

    closed = [x[1] for x in closed]
    active = [x[1] for x in active]
    virtual = [x[1] for x in virtual]
    MOindices = closed + active + virtual

    MOs = []
    for i in MOindices:
        MOs.append(allMOs[i])

    mo_num = len(MOs)
    while len(MOindices) < mo_num:
        MOindices.append(len(MOindices))

    MOmap = list(MOindices)
    for i in range(len(MOindices)):
        MOmap[i] = MOindices.index(i)

    energies = []
    for i in range(mo_num):
        energies.append(MOs[i].eigenvalue)

    MoMatrix = []
    sym0 = [i.sym for i in res.mo_sets[MO_type]]
    sym = [i.sym for i in res.mo_sets[MO_type]]
    for i in range(len(sym)):
        sym[MOmap[i]] = sym0[i]

    MoMatrix = []
    for i in range(len(MOs)):
        m = MOs[i]
        for j in ao_ordering:
            MoMatrix.append(m.vector[j])

    while len(MoMatrix) < len(MOs[0].vector)**2:
        MoMatrix.append(0.)

    trexio.write_mo_num(trexio_file, mo_num)
    trexio.write_mo_coefficient(trexio_file, MoMatrix)
    trexio.write_mo_symmetry(trexio_file, sym)

#    print(res.occ_num)
#    if res.occ_num is not None:
#        OccNum = []
#        for i in MOindices:
#            OccNum.append(res.occ_num[MO_type][i])
#
#        while len(OccNum) < mo_num:
#            OccNum.append(0.)
#        trexio.write_mo_occupation(trexio_file, OccNum)

    return




def todo():
    print("Pseudos\t\t...\t", end=' ')
    try:
        lmax = 0
        nucl_charge_remove = []
        klocmax = 0
        kmax = 0
        nucl_num = len(res.geometry)
        for ecp in res.pseudo:
            lmax_local = ecp['lmax']
            lmax = max(lmax_local, lmax)
            nucl_charge_remove.append(ecp['zcore'])
            klocmax = max(klocmax, len(ecp[str(lmax_local)]))
            for l in range(lmax_local):
                kmax = max(kmax, len(ecp[str(l)]))
        lmax = lmax-1
        trexio.write_pseudo_pseudo_lmax(lmax)
        trexio.write_pseudo_nucl_charge_remove(nucl_charge_remove)
        trexio.write_pseudo_pseudo_klocmax(klocmax)
        trexio.write_pseudo_pseudo_kmax(kmax)
        pseudo_n_k = [[0  for _ in range(nucl_num)] for _ in range(klocmax)]
        pseudo_v_k = [[0. for _ in range(nucl_num)] for _ in range(klocmax)]
        pseudo_dz_k = [[0. for _ in range(nucl_num)] for _ in range(klocmax)]
        pseudo_n_kl = [[[0  for _ in range(nucl_num)] for _ in range(kmax)] for _ in range(lmax+1)]
        pseudo_v_kl = [[[0. for _ in range(nucl_num)] for _ in range(kmax)] for _ in range(lmax+1)]
        pseudo_dz_kl = [[[0. for _ in range(nucl_num)] for _ in range(kmax)] for _ in range(lmax+1)]
        for ecp in res.pseudo:
            lmax_local = ecp['lmax']
            klocmax = len(ecp[str(lmax_local)])
            atom = ecp['atom']-1
            for kloc in range(klocmax):
                try:
                    v, n, dz = ecp[str(lmax_local)][kloc]
                    pseudo_n_k[kloc][atom] = n-2
                    pseudo_v_k[kloc][atom] = v
                    pseudo_dz_k[kloc][atom] = dz
                except:
                    pass
            for l in range(lmax_local):
                for k in range(kmax):
                    try:
                        v, n, dz = ecp[str(l)][k]
                        pseudo_n_kl[l][k][atom] = n-2
                        pseudo_v_kl[l][k][atom] = v
                        pseudo_dz_kl[l][k][atom] = dz
                    except:
                        pass
        trexio.write_pseudo_pseudo_n_k(pseudo_n_k)
        trexio.write_pseudo_pseudo_v_k(pseudo_v_k)
        trexio.write_pseudo_pseudo_dz_k(pseudo_dz_k)
        trexio.write_pseudo_pseudo_n_kl(pseudo_n_kl)
        trexio.write_pseudo_pseudo_v_kl(pseudo_v_kl)
        trexio.write_pseudo_pseudo_dz_kl(pseudo_dz_kl)

        n_alpha = res.num_alpha
        n_beta = res.num_beta
        for i in range(nucl_num):
            charge[i] -= nucl_charge_remove[i]
            n_alpha -= nucl_charge_remove[i]/2
            n_beta -= nucl_charge_remove[i]/2
        trexio.write_nuclei_nucl_charge(charge)
        trexio.write_electrons_elec_alpha_num(n_alpha)
        trexio.write_electrons_elec_beta_num(n_beta)

    except:
        trexio.write_pseudo_do_pseudo(False)
    else:
        trexio.write_pseudo_do_pseudo(True)

    print("OK")





