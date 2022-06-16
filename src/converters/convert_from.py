#!/usr/bin/env python3
"""
Convert output file from a given code/format into TREXIO
"""

import os
from group_tools import basis as trexio_basis
from group_tools import determinant as trexio_det

try:
    import trexio
except ImportError as exc:
    raise ImportError("trexio Python package is not installed.") from exc

try:
    from resultsFile import getFile, a0, get_lm
except ImportError as exc:
    raise ImportError("resultsFile Python package is not installed.") from exc



def run_resultsFile(trexio_filename, filename, back_end):

    if os.path.exists(filename):
        os.system("rm -rf -- "+trexio_filename)

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
    trexio.write_metadata_code(trexio_file,
               [str(res).split('.')[-1].split()[0].replace("File","")] )
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
    nucleus_num = len(res.geometry)

    for a in res.geometry:
        charge.append(a.charge)
        if res.units != 'BOHR':
            coord.append([a.coord[0] / a0, a.coord[1] / a0, a.coord[2] / a0])
        else:
            coord.append([a.coord[0], a.coord[1], a.coord[2]])

    trexio.write_nucleus_num(trexio_file, nucleus_num)
    trexio.write_nucleus_coord(trexio_file, coord)
    # nucleus_charge will be written later after removing core electrons with ECP

    # Transform H1 into H
    import re
    p = re.compile(r'(\d*)$')
    label = [p.sub("", x.name).capitalize() for x in res.geometry]
    trexio.write_nucleus_label(trexio_file, label)

    trexio.write_nucleus_point_group(trexio_file, res.point_group)


    # Basis

    trexio.write_basis_type(trexio_file, "Gaussian")

    # Check whether the basis is Spherical or Cartesian

    cartesian = True
    for b in res.basis:
        if "d+0" in b.sym:
            cartesian = False
            break
        elif "xx" in b.sym:
            break

    # Build basis set
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
        # Warning: assumes +0 is always 1st of spherical functions
        if ("y" in b.sym) or ("z" in b.sym):
            pass
        elif (not cartesian) and (b.sym not in ["s", "x"]) and ("0" not in b.sym):
            pass
        else:
            curr_shell += 1
            # count the max_ang_mom of a given shell
            if cartesian:
                shell_ang_mom.append(b.sym.count("x"))
            elif b.sym == "s":
                shell_ang_mom.append(0)
            elif b.sym == "x":
                shell_ang_mom.append(1)
            elif "0" in b.sym:
                l, _ = get_lm(b.sym)
                shell_ang_mom.append(l)
            curr_shell_idx = len(exponent)
            shell_prim_index.append(curr_shell_idx)
            shell_prim_num.append(len(b.prim))
            exponent += [x.expo for x in b.prim]
            coefficient += b.coef
            prim_factor += [1./x.norm for x in b.prim]
            shell_factor.append(1./b.norm)
            idx = geom.index(b.center)
            if idx != prev_idx:
                nucleus_index.append(curr_shell)
                if len(nucleus_index) > 1:
                    nucl_shell_num.append(nucleus_index[-1]-nucleus_index[-2])

            prev_idx = idx

        ao_shell.append(curr_shell)

    if len(nucleus_index) > 1:
        nucl_shell_num.append(nucleus_index[-1]-nucleus_index[-2])
    else:
        # cade of a single atom
        nucl_shell_num.append(curr_shell + 1)

    shell_num = curr_shell+1
    prim_num = len(exponent)

    # Fix x,y,z in Spherical (don't move this before basis set detection!)
    if cartesian:
        pass
    else:
        for b in res.basis:
            if b.sym == 'z':
               b.sym = 'p+0'
            elif b.sym == 'x':
               b.sym = 'p+1'
            elif b.sym == 'y':
               b.sym = 'p-1'

    # ========================================================================== #
    # Conversion below is needed to convert arrays according to TREXIO v.2.0
    nucleus_index_per_shell = trexio_basis.lists_to_map(nucleus_index, nucl_shell_num)
    shell_index_per_prim = trexio_basis.lists_to_map(shell_prim_index, shell_prim_num)
    # ========================================================================= #

    # write total number of shell and primitives
    trexio.write_basis_shell_num(trexio_file,shell_num)
    trexio.write_basis_prim_num(trexio_file,prim_num)

    # write mappings to reconstruct per-atom and per-shell quantities
    trexio.write_basis_nucleus_index(trexio_file,nucleus_index_per_shell)
    trexio.write_basis_shell_ang_mom(trexio_file,shell_ang_mom)
    trexio.write_basis_shell_index(trexio_file,shell_index_per_prim)

    # write normalization factor for each shell
    trexio.write_basis_shell_factor(trexio_file,shell_factor)

    # write parameters of the primitives
    trexio.write_basis_exponent(trexio_file,exponent)
    trexio.write_basis_coefficient(trexio_file,coefficient)
    trexio.write_basis_prim_factor(trexio_file,prim_factor)


    # AO
    # --

    #res.convert_to_cartesian()
    trexio.write_ao_cartesian(trexio_file, cartesian)
    trexio.write_ao_num(trexio_file, len(res.basis))
    trexio.write_ao_shell(trexio_file, ao_shell)

    ao_ordering = []
    accu = []
    normalization = []
    prev_shell = None

    # Re-order AOs (xx,xy,xz,yy,yz,zz) or (d+0,+1,-1,-2,+2,-2,...)
    def f_sort(x):
        if '+' in x or '-' in x:
            l, m = get_lm(x)
            if m>=0:
                return 2*m
            else:
                return -2*m+1
        else:
            return x

    for i,b in enumerate(res.basis):
        shell = ao_shell[i]
        if shell != prev_shell:
            accu.sort()
            ao_ordering += accu
            accu = []
        accu += [(f_sort(b.sym), i, b.sym )]
        prev_shell = shell
    accu.sort()
    ao_ordering += accu

    ao_ordering = [ i for (_,i,_) in ao_ordering ]

    # Normalization
    normalization = []
    for i,k in enumerate(ao_ordering):
        b = res.basis[k]
        orig = res.basis[ao_shell.index(ao_shell[k])]
        prim = b.prim
        prim_norm = [ j.norm for j in prim ]
        oprim = orig.prim
        oprim_norm = [ j.norm for j in oprim ]
        accum = 0.
        for i, ci in enumerate(b.coef):
            ci /= prim_norm[i]
            for j, cj in enumerate(orig.coef):
                cj /= oprim_norm[j]
                accum += ci*cj * oprim[i].overlap(oprim[j])
        accum /= orig.norm**2
        normalization.append(accum)
    trexio.write_ao_normalization(trexio_file, normalization)


    # MOs
    # ---

    MO_type = res.determinants_mo_type
    #print ("available motypes", res.mo_types)

    allMOs = res.mo_sets[MO_type]
    trexio.write_mo_type(trexio_file, MO_type)

    full_mo_set  = [(allMOs[i].eigenvalue, i) for i in range(len(allMOs))]
    MOindices = [x[1] for x in full_mo_set]

    ## The following commented portion for the future use.
    # try:
    #     closed  = [(allMOs[i].eigenvalue, i) for i in res.closed_mos]
    #     virtual = [(allMOs[i].eigenvalue, i) for i in res.virtual_mos]
    #     active  = [(allMOs[i].eigenvalue, i) for i in res.active_mos]
    # except:
    #     closed  = []
    #     virtual = []
    #     active  = [(allMOs[i].eigenvalue, i) for i in range(len(allMOs))]

    # closed  = [x[1] for x in closed]
    # active  = [x[1] for x in active]
    # virtual = [x[1] for x in virtual]
    # MOindices = closed + active + virtual

    MOs = []
    for i in MOindices:
        MOs.append(allMOs[i])

    mo_num = len(MOindices)
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
        if sym0[i] is None:
            sym[MOmap[i]] = 'A'
        else:
            sym[MOmap[i]] = sym0[i]

    MoMatrix = []
    for i in range(len(MOs)):
        m = MOs[i]
        for j in ao_ordering:
            MoMatrix.append(m.vector[j])


    trexio.write_mo_num(trexio_file, mo_num)
    trexio.write_mo_coefficient(trexio_file, MoMatrix)
    trexio.write_mo_symmetry(trexio_file, sym)

#       TODO: occupations are not always provided in the output file ??
#    print(res.occ_num)
#    if res.occ_num is not None:
#        OccNum = []
#        for i in MOindices:
#           OccNum.append(res.occ_num[MO_type][i])
#    # Not sure about the part below as it might overwrite values from the
#    # previous step !  while len(OccNum) < mo_num:
#            OccNum.append(0.)
#        trexio.write_mo_occupation(trexio_file, OccNum)

    lmax = 0
    nucl_charge_remove = []

    nucl_num = len(res.geometry)
    lmax_plus_1_per_atom = []

    map_l = []
    map_nucleus = []

    if res.pseudo:
      ecp_num_total = 0
      ecp_coef_total = []
      ecp_exp_total = []
      ecp_power_total = []
      for ecp in res.pseudo:
          lmax_atomic = ecp['lmax']
          atom = ecp['atom']-1

          lmax_plus_1_per_atom.append(lmax_atomic)

          nucl_charge_remove.append(ecp['zcore'])

          for l in range(lmax_atomic+1):
              l_str = str(l)

              n_per_l = len(ecp[l_str])

              map_nucleus.extend([atom for _ in range(n_per_l) if n_per_l != 0])
              map_l.extend([l for _ in range(n_per_l) if n_per_l != 0])

              ecp_num_total += n_per_l

              coef_per_l = [arr[0] for arr in ecp[l_str]]
              # shift powers by 2 because of the format
              power_per_l = [arr[1]-2 for arr in ecp[l_str]]
              exp_per_l = [arr[2] for arr in ecp[l_str]]

              ecp_coef_total.extend(coef_per_l)
              ecp_power_total.extend(power_per_l)
              ecp_exp_total.extend(exp_per_l)


      # lmax+1 is one higher that the max angular momentum of the core orbital
      # to be removed (per atom)
      trexio.write_ecp_max_ang_mom_plus_1(trexio_file, lmax_plus_1_per_atom)
      # write core charges to be removed
      trexio.write_ecp_z_core(trexio_file, nucl_charge_remove)
      # write total num of ECP elements
      trexio.write_ecp_num(trexio_file, ecp_num_total)
      # write 1-to-1 mapping needed to reconstruct ECPs
      trexio.write_ecp_ang_mom(trexio_file, map_l)
      trexio.write_ecp_nucleus_index(trexio_file, map_nucleus)
      # write ECP quantities in the TREXIO file
      trexio.write_ecp_power(trexio_file, ecp_power_total)
      trexio.write_ecp_coefficient(trexio_file, ecp_coef_total)
      trexio.write_ecp_exponent(trexio_file, ecp_exp_total)


      for i in range(nucl_num):
          charge[i] -= nucl_charge_remove[i]

    # end if res.pseudo:
    trexio.write_nucleus_charge(trexio_file, charge)

    # Determinants
    # ---------

    # resultsFile has non-empty det_coefficients sometimes
    if len(res.det_coefficients[0]) > 1:

        int64_num       = int((mo_num-1)/64) + 1
        determinant_num = len(res.det_coefficients[0])

        # sanity check
        if res.num_states > 1:
            assert determinant_num == len(res.det_coefficients[1])

        # construct the determinant_list of integer bitfields from resultsFile determinants reprsentation
        det_list = []
        for i in range(determinant_num):
            det_tmp      = []
            orb_list_up  = [ orb+1 for orb in res.determinants[i].get("alpha") ]
            det_tmp     += trexio_det.to_determinant_list(orb_list_up, int64_num)
            orb_list_dn  = [ orb+1 for orb in res.determinants[i].get("beta") ]
            det_tmp     += trexio_det.to_determinant_list(orb_list_dn, int64_num)

            det_list.append(det_tmp)

        # write the CI determinants
        offset_file = 0
        trexio.write_determinant_list(trexio_file, offset_file, determinant_num, det_list)

        # write the CI coefficients
        offset_file = 0
        for s in range(res.num_states):
            trexio_file.set_state(s)
            trexio.write_determinant_coefficient(trexio_file, offset_file, determinant_num, res.det_coefficients[s])


    # close the file before leaving
    trexio_file.close()

    print("Conversion to TREXIO format has been completed.")

    return


def run(trexio_filename, filename, filetype, back_end):
    if filetype.lower() == "gaussian":
        run_resultsFile(trexio_filename, filename, back_end)
    elif filetype.lower() == "gamess":
        run_resultsFile(trexio_filename, filename, back_end)
    elif filetype.lower() == "fcidump":
        run_fcidump(trexio_filename, filename, back_end)
    elif filetype.lower() == "molden":
        run_molden(trexio_filename, filename, back_end)
    else:
        raise TypeError("Unknown file type")
