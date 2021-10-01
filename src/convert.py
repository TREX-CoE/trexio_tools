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



def run(trexio_filename,filename):

    trexio_file = trexio.File(trexio_filename,mode='w',back_end=trexio.TREXIO_TEXT)
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


    # AO Basis

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
    prev_idx = None
    geom = [ a.coord for a in res.geometry ]
    for b in res.basis:
      if "y" in b.sym or "z" in b.sym:
        pass
      else:
        basis.append(b)
        shell_ang_mom.append(b.sym.count("x"))
        shell_prim_index.append(len(exponent))
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

    return

def pouet():

    at = []
    num_prim = []
    power_x = []
    power_y = []
    power_z = []
    coefficient = []
    exponent = []

    res.convert_to_cartesian()

    for b in res.basis:
        c = b.center
        for i, atom in enumerate(res.geometry):
            if atom.coord == c:
                at.append(i + 1)
        num_prim.append(len(b.prim))
        s = b.sym
        power_x.append(str.count(s, "x"))
        power_y.append(str.count(s, "y"))
        power_z.append(str.count(s, "z"))
        coefficient.append(b.coef)
        exponent.append([p.expo for p in b.prim])


    trexio.write_ao_cartesian(trexio_file, true)
    trexio.write_ao_num(len(res.basis))
    trexio.write_ao_nucl(at)

#    trexio.write_ao_prim_num(num_prim)
#    trexio.write_ao_power(power_x + power_y + power_z)

    prim_num_max = trexio.get_ao_basis_ao_prim_num_max()

    for i in range(len(res.basis)):
        coefficient[
            i] += [0. for j in range(len(coefficient[i]), prim_num_max)]
        exponent[i] += [0. for j in range(len(exponent[i]), prim_num_max)]

    coefficient = reduce(lambda x, y: x + y, coefficient, [])
    exponent = reduce(lambda x, y: x + y, exponent, [])

    coef = []
    expo = []
    for i in range(prim_num_max):
        for j in range(i, len(coefficient), prim_num_max):
            coef.append(coefficient[j])
            expo.append(exponent[j])

    # ~#~#~#~#~ #
    # W r i t e #
    # ~#~#~#~#~ #

    trexio.write_ao_basis_ao_coef(coef)
    trexio.write_ao_basis_ao_expo(expo)
    trexio.write_ao_basis_ao_basis("Read by resultsFile")

    print("OK")

    #   _
    #  |_)  _.  _ o  _
    #  |_) (_| _> | _>
    #

    print("Basis\t\t...\t", end=' ')
    # ~#~#~#~ #
    # I n i t #
    # ~#~#~#~ #

    coefficient = []
    exponent = []

    # ~#~#~#~#~#~#~ #
    # P a r s i n g #
    # ~#~#~#~#~#~#~ #

    nbasis = 0
    nucl_center = []
    curr_center = -1
    nucl_shell_num = []
    ang_mom = []
    nshell = 0
    shell_prim_index = [1]
    shell_prim_num = []
    for b in res.basis:
        s = b.sym
        if str.count(s, "y") + str.count(s, "x") == 0:
          c = b.center
          nshell += 1
          if c != curr_center:
             curr_center = c
             nucl_center.append(nbasis+1)
             nucl_shell_num.append(nshell)
             nshell = 0
          nbasis += 1
          coefficient += b.coef[:len(b.prim)]
          exponent += [p.expo for p in b.prim]
          ang_mom.append(str.count(s, "z"))
          shell_prim_index.append(len(exponent)+1)
          shell_prim_num.append(len(b.prim))

    nucl_shell_num.append(nshell+1)
    nucl_shell_num = nucl_shell_num[1:]

    # ~#~#~#~#~ #
    # W r i t e #
    # ~#~#~#~#~ #

    trexio.write_basis_basis("Read from ResultsFile")
    trexio.write_basis_basis_nucleus_index(nucl_center)
    trexio.write_basis_prim_num(len(coefficient))
    trexio.write_basis_shell_num(len(ang_mom))
    trexio.write_basis_nucleus_shell_num(nucl_shell_num)
    trexio.write_basis_prim_coef(coefficient)
    trexio.write_basis_prim_expo(exponent)
    trexio.write_basis_shell_ang_mom(ang_mom)
    trexio.write_basis_shell_prim_num(shell_prim_num)
    trexio.write_basis_shell_prim_index(shell_prim_index)

    print("OK")

    #                _
    # |\/|  _   _   |_)  _.  _ o  _
    # |  | (_) _>   |_) (_| _> | _>
    #

    print("MOS\t\t...\t", end=' ')
    # ~#~#~#~ #
    # I n i t #
    # ~#~#~#~ #

    MoTag = res.determinants_mo_type
    trexio.write_mo_basis_mo_label('Orthonormalized')
    MO_type = MoTag
    allMOs = res.mo_sets[MO_type]

    # ~#~#~#~#~#~#~ #
    # P a r s i n g #
    # ~#~#~#~#~#~#~ #

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

    if res.occ_num is not None:
        OccNum = []
        for i in MOindices:
            OccNum.append(res.occ_num[MO_type][i])

        while len(OccNum) < mo_num:
            OccNum.append(0.)

    MoMatrix = []
    sym0 = [i.sym for i in res.mo_sets[MO_type]]
    sym = [i.sym for i in res.mo_sets[MO_type]]
    for i in range(len(sym)):
        sym[MOmap[i]] = sym0[i]

    MoMatrix = []
    for i in range(len(MOs)):
        m = MOs[i]
        for coef in m.vector:
            MoMatrix.append(coef)

    while len(MoMatrix) < len(MOs[0].vector)**2:
        MoMatrix.append(0.)

    # ~#~#~#~#~ #
    # W r i t e #
    # ~#~#~#~#~ #

    trexio.write_mo_basis_mo_num(mo_num)
    trexio.write_mo_basis_mo_occ(OccNum)
    trexio.write_mo_basis_mo_coef(MoMatrix)
    print("OK")


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





