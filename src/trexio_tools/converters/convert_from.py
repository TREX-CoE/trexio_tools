#!/usr/bin/env python3
"""
Convert output file from a given code/format into TREXIO
"""

import os
from trexio_tools.group_tools import basis as trexio_basis
from trexio_tools.group_tools import determinant as trexio_det

from .pyscf_to_trexio import pyscf_to_trexio as run_pyscf
from .orca_to_trexio import orca_to_trexio as run_orca
from .crystal_to_trexio import crystal_to_trexio as run_crystal

import trexio

from ..trexio_run import remove_trexio_file

try:
    from resultsFile import getFile, a0, get_lm
    import resultsFile
except ImportError as exc:
    raise ImportError("resultsFile Python package is not installed.") from exc


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

#def file_cleanup(trexio_filename, back_end):
#    if os.path.exists(trexio_filename):
#        print(f"TREXIO file {trexio_filename} already exists and will be removed before conversion.")
#        if back_end == trexio.TREXIO_HDF5:
#            os.remove(trexio_filename)
#        else:
#            raise NotImplementedError(f"Please remove the {trexio_filename} directory manually.")

def run_resultsFile(trexio_file, filename_info, motype=None):

    filename = filename_info['filename']
    trexio_basename = filename_info['trexio_basename']
    state_suffix = filename_info['state_suffix']
    trexio_extension = filename_info['trexio_extension']
    state_id = filename_info['state']

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
    if res.author is not None:
        trexio.write_metadata_author_num(trexio_file, 1)
        trexio.write_metadata_author(trexio_file, [res.author])
    if res.title is not None:
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
    try:
      normf = res.normf
    except AttributeError:
      normf=0
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
            if normf == 0:
                shell_factor.append(1./b.norm)
            else:
                shell_factor.append(1.)
            idx = geom.index(b.center)
            if idx != prev_idx:
                nucleus_index.append(curr_shell)
                if len(nucleus_index) > 1:
                    nucl_shell_num.append(nucleus_index[-1]-nucleus_index[-2])

            prev_idx = idx

        ao_shell.append(curr_shell)

    shell_num = curr_shell+1
    prim_num = len(exponent)

    nucl_shell_num.append(shell_num-nucleus_index[-1])

    assert(sum(nucl_shell_num) == shell_num)

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

    # For Gaussian basis sets, basis_r_power is zero
    basis_r_power = [0.0 for _ in range(shell_num) ]
    trexio.write_basis_r_power(trexio_file,basis_r_power)

    # AO
    # --

    #res.convert_to_cartesian()
    trexio.write_ao_cartesian(trexio_file, cartesian)
    trexio.write_ao_num(trexio_file, len(res.basis))
    trexio.write_ao_shell(trexio_file, ao_shell)

    ao_ordering = []
    accu = []
    prev_shell = None

    # Re-order AOs (xx,xy,xz,yy,yz,zz) or (d+0,+1,-1,-2,+2,-2,...)
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

    if motype is None:
        MO_type = res.determinants_mo_type
    else:
        MO_type = motype
    print ("available motypes", res.mo_types)

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
#    if res.occ_num is not None:
#        OccNum = []
#        for i in MOindices:
#           OccNum.append(res.occ_num[MO_type][i])
#    # Not sure about the part below as it might overwrite values from the
#    # previous step !
#        while len(OccNum) < mo_num:
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

    # State group
    # ---------
    if state_id != trexio_file.get_state():
       print("Warning: State ID mismatch between the file and the TREXIO file.")

    state_id = trexio_file.get_state()
    trexio.write_state_num(trexio_file,res.num_states)
    try:
      trexio.write_state_energy(trexio_file,res.energy[0])
    except:
      pass
    trexio.write_state_current_label(trexio_file, f"State {state_id}")
    trexio.write_state_label(trexio_file, [f"State {i}" for i in range(res.num_states)])

    # Get the basename of the TREXIO file
    file_names = [ f"{trexio_basename}_{state_suffix}_{s}{trexio_extension}" for s in range(res.num_states) ]
    file_names[0] = f"{trexio_basename}{trexio_extension}"
    trexio.write_state_file_name(trexio_file, file_names)

    # CSF group
    # ---------
    if hasattr(res, 'csf_coefficients') and res.csf_coefficients[state_id]:
        try:
            num_csfs = len(res.csf_coefficients[state_id])
        except:
            num_csfs = len(res.det_coefficients[0])

        offset_file = 0
        trexio.write_csf_coefficient(trexio_file, offset_file, num_csfs, res.csf_coefficients[state_id])

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
            orb_list_up  = [ orb for orb in res.determinants[i].get("alpha") ]
            det_tmp     += trexio_det.to_determinant_list(orb_list_up, int64_num)
            orb_list_dn  = [ orb for orb in res.determinants[i].get("beta") ]
            det_tmp     += trexio_det.to_determinant_list(orb_list_dn, int64_num)

            det_list.append(det_tmp)

        # write the CI determinants
        offset_file = 0
        trexio.write_determinant_list(trexio_file, offset_file, determinant_num, det_list)

        # write the CI coefficients
        offset_file = 0
        trexio.write_determinant_coefficient(trexio_file, offset_file, determinant_num, res.det_coefficients[state_id])

        # close the file before leaving
        trexio_file.close()

        print("Conversion to TREXIO format has been completed for the state ", state_id, " in the file ", trexio_file.filename)

    return

def run_molden(trexio_file, filename, normalized_basis=True, multiplicity=None, ao_norm=0):
    import numpy as np

    with open(filename, 'r') as f:
        lines = f.readlines()

    if not lines[0].startswith("[Molden Format]"):
        print("File not in Molden format")
        raise TypeError

    title = lines[1].strip()
    atoms = []
    gto = []
    unit = None
    inside = None
    cartesian = True
    sym = []
    ene = []
    spin = []
    occup = []
    mo_coef = []
    mo = []
    for line in lines:
       line = line.strip()
       if line == "":
          continue
       if line.lower().startswith("[atoms]"):
          if "au" in line.lower().split()[1]:
            unit = "au"
          else:
            unit = "angs"
          inside = "Atoms"
          continue
       elif line.upper().startswith("[GTO]"):
          inside = "GTO"
          continue
       elif line.upper().startswith("[MO]"):
          inside = "MO"
          continue
       elif line.startswith("[5d]") \
         or line.startswith("[7f]") \
         or line.startswith("[9g]"):
           cartesian = False
           continue
       elif line.startswith("["):
          inside = None
       if inside == "Atoms":
          buffer = line.split()
          atoms.append( (buffer[0], int(buffer[2]), float(buffer[3]),
                       float(buffer[4]), float(buffer[5])) )
          continue
       elif inside == "GTO":
          gto.append(line)
          continue
       elif inside == "MO":
          in_coef = False
          if line.lower().startswith("sym"):
             sym.append ( line.split('=')[1].strip() )
          elif line.lower().startswith("ene"):
             ene.append ( float(line.split('=')[1].strip()) )
          elif line.lower().startswith("occ"):
             occup.append ( float(line.split('=')[1].strip()) )
          elif line.lower().startswith("spin"):
             if line.split('=')[1].strip().lower == "alpha":
                spin.append(0)
             else:
                spin.append(1)
          else:
             in_coef = True
          if in_coef:
             buffer = line.split()
             mo.append( (int(buffer[0])-1, float(buffer[1])) )
          if not in_coef and len(mo) > 0:
             mo_coef.append(mo)
             mo = []
          continue

    if len(mo) > 0:
       mo_coef.append(mo)

    # Metadata
    # --------

    trexio.write_metadata_code_num(trexio_file, 1)
    trexio.write_metadata_code(trexio_file, ["Molden"])
    trexio.write_metadata_author_num(trexio_file, 1)
    trexio.write_metadata_author(trexio_file, [os.environ["USER"]])
    trexio.write_metadata_description(trexio_file, title)

    # Electrons
    # ---------

    elec_num = int(sum(occup)+0.5)
    if multiplicity is None:
        up_num = 0
        dn_num = 0
        for o in occup:
            if o > 1.0:
                up_num += 1
                dn_num += 1
            elif o == 1.0:
                up_num += 1
    else:
        up_num = (multiplicity-1 + elec_num)/2
        dn_num = elec_num - up_num
    assert (elec_num == up_num + dn_num)
    trexio.write_electron_up_num(trexio_file,up_num)
    trexio.write_electron_dn_num(trexio_file,dn_num)

    # Nuclei
    # ------

    charge = []
    coord = []
    nucleus_num = len(atoms)

    coord = []
    label = []
    for a in atoms:
        charge.append(float(a[1]))
        label.append(a[0])
        if unit != 'au':
            coord.append([a[2] / a0, a[3] / a0, a[4] / a0])
        else:
            coord.append([a[2], a[3], a[4]])

    trexio.write_nucleus_num(trexio_file, len(atoms))
    trexio.write_nucleus_coord(trexio_file, coord)
    trexio.write_nucleus_charge(trexio_file, charge)
    trexio.write_nucleus_label(trexio_file, label)


    # Basis
    # -----

    nucleus_index = []
    shell_ang_mom = []
    shell_index = []
    shell_prim_index = []
    shell_factor = []
    exponent = []
    coefficient = []
    prim_factor = []
    contraction = None

    shell_id = -1
    prim_id = -1
    iatom = 0
    for line in gto:
       buffer = line.replace('D','E').split()
       if len(buffer) == 2 and buffer[1] == "0":
           iatom = int(buffer[0])-1
       elif len(buffer) == 3 and float(buffer[2]) == 1.0:
           if contraction is not None:
               if normalized_basis:
                    accum = 0.
                    n = [ x.norm for x in contraction.prim ]
                    for i, ci in enumerate(contraction.coef):
                        ci /= n[i]
                        for j, cj in enumerate(contraction.coef):
                            cj /= n[j]
                            accum += ci*cj * contraction.prim[i].overlap(contraction.prim[j])
                    shell_factor.append(1./accum)
               else:
                    shell_factor.append(1.)
           shell_id += 1
           ang_mom = buffer[0].lower()
           nprim = int(buffer[1])
           nucleus_index.append(iatom)
           if   ang_mom == "s": shell_ang_mom.append(0)
           elif ang_mom == "p": shell_ang_mom.append(1)
           elif ang_mom == "d": shell_ang_mom.append(2)
           elif ang_mom == "f": shell_ang_mom.append(3)
           elif ang_mom == "g": shell_ang_mom.append(4)
           if   ang_mom != "s": ang_mom = "x"*shell_ang_mom[-1]
           contraction = resultsFile.contraction()
       else:
           prim_id += 1
           e, c = float(buffer[0]), float(buffer[1])
           shell_prim_index.append(prim_id)
           exponent.append(e)
           coefficient.append(c)
           gauss = resultsFile.gaussian()
           gauss.center = coord[iatom]
           gauss.expo = e
           gauss.sym  =ang_mom
           contraction.append(c, gauss)
           prim_factor.append(1./gauss.norm)
           shell_index.append(shell_id)

    if contraction is not None:
        if normalized_basis:
            accum = 0.
            n = [ x.norm for x in contraction.prim ]
            for i, ci in enumerate(contraction.coef):
                ci /= n[i]
                for j, cj in enumerate(contraction.coef):
                    cj /= n[j]
                    accum += ci*cj * contraction.prim[i].overlap(contraction.prim[j])
            shell_factor.append(1./accum)
        else:
            shell_factor.append(1.)

    shell_num = shell_id + 1
    prim_num  = prim_id  + 1

    trexio.write_basis_type(trexio_file, "Gaussian")

    # write total number of shell and primitives
    trexio.write_basis_shell_num(trexio_file,shell_num)
    trexio.write_basis_prim_num(trexio_file,prim_num)

    # write mappings to reconstruct per-atom and per-shell quantities
    trexio.write_basis_nucleus_index(trexio_file,nucleus_index)
    trexio.write_basis_shell_ang_mom(trexio_file,shell_ang_mom)
    trexio.write_basis_shell_index(trexio_file,shell_index)

    # write normalization factor for each shell
    trexio.write_basis_shell_factor(trexio_file,shell_factor)

    # For Gaussian basis sets, basis_r_power is zero
    basis_r_power = [0.0 for _ in range(basis_shell_num) ]
    trexio.write_basis_r_power(trexio_file,basis_r_power)

    # write parameters of the primitives
    trexio.write_basis_exponent(trexio_file,exponent)
    trexio.write_basis_coefficient(trexio_file,coefficient)
    trexio.write_basis_prim_factor(trexio_file,prim_factor)


    # AOs
    # ---

    if max(shell_ang_mom) < 2:
       cartesian=True

    if cartesian:
        conv = [ [ 's' ], ['x', 'y', 'z'], ['xx', 'yy', 'zz', 'xy', 'xz', 'yz'],
                 ['xxx', 'yyy', 'zzz', 'xyy', 'xxy', 'xxz', 'xzz', 'yzz', 'yyz', 'xyz'],
                 ['xxxx', 'yyyy', 'zzzz', 'xxxy', 'xxxz', 'xyyy', 'yyyz', 'xzzz', 'yzzz',
                  'xxyy', 'xxzz', 'yyzz', 'xxyz', 'xyyz', 'xyzz'] ]
    else:
        conv = [ ['s'], ['p+1', 'p-1', 'p+0'], ['d+0', 'd+1', 'd-1', 'd+2', 'd-2'],
                 ['f+0', 'f+1', 'f-1', 'f+2', 'f-2', 'f+3', 'f-3'],
                 ['g+0', 'g+1', 'g-1', 'g+2', 'g-2', 'g+3', 'g-3', 'g+4', 'g-4'] ]

    norm = []
    for l in range(5):
       g = resultsFile.gaussian()
       gauss.center = (0.,0.,0.)
       gauss.expo = 1.0
       gauss.sym = conv[l][0]
       ref = gauss.norm
       norm.append([])
       for m in conv[l]:
            g = resultsFile.gaussian()
            gauss.center = (0.,0.,0.)
            gauss.expo = 1.0
            gauss.sym = m
            norm[l].append ( gauss.norm / ref )

    ao = []
    ao_normalization = []
    for l in shell_ang_mom:
       if l>4:
          raise TypeError("Angular momentum too high: l>4 not supported by Molden format.")
       ao.append(conv[l])
       ao_normalization.append(norm[l])

    ao_shell = []
    ao_ordering = []
    j = 0
    for k,l in enumerate(ao):
      ao_shell += [ k for _ in l ]
      accu = [ (f_sort(x), i+j, norm[shell_ang_mom[k]][i])  for i,x in enumerate(l) ]
      accu.sort()
      ao_ordering += accu
      j += len(l)
    ao_normalization = [ i for (_,_,i) in ao_ordering ]
    ao_ordering = [ i for (_,i,_) in ao_ordering ]
    ao_num = len(ao_ordering)

    trexio.write_ao_num(trexio_file, ao_num)
    trexio.write_ao_cartesian(trexio_file, cartesian)
    trexio.write_ao_shell(trexio_file, ao_shell)
    trexio.write_ao_normalization(trexio_file, ao_normalization)

    # MOs
    # ---

#    trexio.write_mo_type(trexio_file, MO_type)

    core   = []
    active = []
    virtual = []
    mo_class = []
    for i, o in enumerate(occup):
       if o >= 2.:
          core.append(i)
          mo_class.append("Core")
       elif o == 0.:
          virtual.append(i)
          mo_class.append("Virtual")
       else:
          active.append(i)
          mo_class.append("Active")

    trexio.write_mo_num(trexio_file, len(mo_class))
    MoMatrix = []
    for mo in mo_coef:
      vector = np.zeros(ao_num)
      for i, x in mo:
         vector[i] = x
      for i in ao_ordering:
         MoMatrix.append(vector[i])

    trexio.write_mo_spin(trexio_file, spin)
    trexio.write_mo_class(trexio_file, mo_class)
    trexio.write_mo_occupation(trexio_file, occup)
    trexio.write_mo_symmetry(trexio_file, sym)
    trexio.write_mo_coefficient(trexio_file, MoMatrix)


def run(trexio_filename, filename, filetype, back_end, spin=None, motype=None, state_suffix=None, overwrite=False):

    # Get the basename of the TREXIO file
    try:
        trexio_basename, trexio_extension = os.path.splitext(os.path.basename(trexio_filename))
    except Exception as e:
        trexio_basename = os.path.basename(trexio_filename)
        trexio_extension = ""


    filename_info = {}
    filename_info['filename'] = filename
    filename_info['trexio_basename'] = trexio_basename
    filename_info['state_suffix'] = state_suffix
    filename_info['trexio_extension'] = trexio_extension
    filename_info['state'] = 0

    if "pyscf" not in filetype.lower() and "gamess" not in filetype.lower():
        trexio_file = trexio.File(trexio_filename, mode='w', back_end=back_end)

    if filetype.lower() == "gaussian":
        run_resultsFile(trexio_file, filename_info, motype)

    elif filetype.lower() == "gamess":
        # Handle the case where the number of states is greater than 1
        try:
            res = getFile(filename)
        except Exception:
            print(f"An error occurred while parsing the file using resultsFile : {Exception}")

        # Open the TREXIO file for writing
        trexio_file = trexio.File(trexio_filename, mode='w', back_end=back_end)
        run_resultsFile(trexio_file, filename_info, motype)

        # Check the number of states in the quantum chemical calculation file first
        if res.num_states > 1:
            print(f"Number of states in the quantum chemical calculation file     {res.num_states}")
            # Create a separate TREXIO file for each state
            for s in range(1,res.num_states):
                trexio_filename = f"{trexio_basename}_{state_suffix}_{s}{trexio_extension}"
                remove_trexio_file(trexio_filename, overwrite)
                trexio_file = trexio.File(trexio_filename, mode='w', back_end=back_end)
                trexio_file.set_state(s)
                filename_info['state'] = s
                run_resultsFile(trexio_file, filename_info, motype)

    elif filetype.lower() == "pyscf":
        back_end_str = "text" if back_end==trexio.TREXIO_TEXT else "hdf5"
        run_pyscf(trexio_filename=trexio_filename, pyscf_checkfile=filename, back_end=back_end_str)

    elif filetype.lower() == "orca":
        back_end_str = "text" if back_end==trexio.TREXIO_TEXT else "hdf5"
        run_orca(filename=trexio_filename, orca_json=filename, back_end=back_end_str)

    elif filetype.lower() == "crystal":
        if spin is None: raise ValueError("You forgot to provide spin for the CRYSTAL->TREXIO converter.")
        back_end_str = "text" if back_end==trexio.TREXIO_TEXT else "hdf5"
        run_crystal(trexio_filename=trexio_filename, crystal_output=filename, back_end=back_end_str, spin=spin)

    elif filetype.lower() == "molden":
        run_molden(trexio_file, filename)

    else:
        raise NotImplementedError(f"Conversion from {filetype} to TREXIO is not supported.")
