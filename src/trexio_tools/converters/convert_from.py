#!/usr/bin/env python3
"""
Convert output file from a given code/format into TREXIO
"""

import os
from trexio_tools.group_tools import basis as trexio_basis

from .orca_to_trexio import orca_to_trexio as run_orca
from .crystal_to_trexio import crystal_to_trexio as run_crystal
from .vlx_to_trexio import vlx_to_trexio as run_vlx

import trexio

from ..trexio_run import remove_trexio_file

try:
    from resultsFile import getFile, a0, get_lm
    import resultsFile
except ImportError:
    getFile = None
    a0 = None
    get_lm = None
    resultsFile = None


def _require_resultsfile():
    if resultsFile is None or getFile is None or a0 is None or get_lm is None:
        raise ImportError("resultsFile Python package is not installed.")

    return getFile, a0, get_lm, resultsFile


# Re-order AOs (xx,xy,xz,yy,yz,zz) or (d+0,+1,-1,-2,+2,-2,...)
def f_sort(x):
  if '+' in x or '-' in x:
      _, _, get_lm_local, _ = _require_resultsfile()
      l, m = get_lm_local(x)
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


# --- Gaussian formatted checkpoint (.fchk) helpers --------------------------

# Element symbols indexed by atomic number (index 0 is a placeholder).
PERIODIC_TABLE = [
    "X",
    "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne",
    "Na", "Mg", "Al", "Si", "P",  "S",  "Cl", "Ar", "K",  "Ca",
    "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",  "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I",  "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U",  "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
]

# Number of values printed per line for each FCHK array type, used to skip over
# data blocks while scanning the file.
_FCHK_PER_LINE = {"I": 6, "R": 5, "C": 5, "L": 72, "H": 9}


def _double_factorial(n: int) -> float:
    """Double factorial (n)!! with the convention (-1)!! = 0!! = 1."""
    result = 1.0
    while n > 1:
        result *= n
        n -= 2
    return result


def _prim_ref_norm(l: int, alpha: float) -> float:
    """Normalization of a primitive cartesian Gaussian for the reference
    component (l, 0, 0), i.e. <g|g> = 1."""
    import math
    return (2.0 * alpha / math.pi) ** 0.75 * \
           math.sqrt((4.0 * alpha) ** l / _double_factorial(2 * l - 1))


def _ref_overlap(l: int, ai: float, aj: float) -> float:
    """Overlap of two *unnormalized* reference (l, 0, 0) primitives with
    exponents ai and aj, sharing the same center."""
    import math
    p = ai + aj
    return _double_factorial(2 * l - 1) * (math.pi / p) ** 1.5 / (2.0 * p) ** l


def _component_norm(label: str) -> float:
    """Normalization of an AO component relative to the reference (l, 0, 0)
    cartesian component. Pure-spherical components (labels carrying a magnetic
    quantum number such as ``d+1``) are already normalized, hence return 1."""
    import math
    if '+' in label or '-' in label:
        return 1.0
    a, b, c = label.count('x'), label.count('y'), label.count('z')
    l = a + b + c
    denom = _double_factorial(2 * a - 1) * \
            _double_factorial(2 * b - 1) * \
            _double_factorial(2 * c - 1)
    return math.sqrt(_double_factorial(2 * l - 1) / denom)


def _ao_sort_key(label: str):
    """Sort key mapping an AO component to its position in the TREXIO ordering.

    Cartesian monomials are ordered alphabetically (xx, xy, xz, yy, yz, zz),
    which matches the canonical TREXIO cartesian ordering. Spherical components
    are ordered by magnetic quantum number as 0, +1, -1, +2, -2, ...
    """
    if '+' in label or '-' in label:
        m = int(label[1:])
        return 2 * m if m >= 0 else -2 * m + 1
    return label


def _parse_fchk(filename: str) -> dict:
    """Parse a Gaussian formatted checkpoint file into a dictionary.

    Scalar fields map to a single int/float; array fields (``N=``) map to a list
    of int/float. Character and logical blocks are skipped but consumed so that
    the surrounding scalar/array fields keep parsing correctly.
    """
    # Stream the file line by line, buffering only the current array block, so
    # that memory usage stays bounded even for very large checkpoints.
    with open(filename, 'r') as fh:
        title = fh.readline()
        calc = fh.readline()
        if not title or not calc:
            raise TypeError(f"{filename} is not a valid Gaussian fchk file.")

        data = {
            '_title': title.strip(),
            '_calc':  calc.rstrip('\n'),
        }

        for line in fh:
            line = line.rstrip('\n')
            # A field line carries the type letter in column 44 (0-based 43).
            if len(line) < 44:
                continue
            name = line[:40].strip()
            rest = line[40:].split()
            if len(rest) < 2 or rest[0] not in _FCHK_PER_LINE:
                continue
            dtype = rest[0]

            if len(rest) >= 3 and rest[1] == 'N=':
                # Array field: read the following data lines.
                try:
                    count = int(rest[2])
                except ValueError:
                    continue
                per_line = _FCHK_PER_LINE[dtype]
                nlines = (count + per_line - 1) // per_line if count > 0 else 0
                # Consume the block for every dtype so the surrounding fields
                # keep parsing, but only buffer/store I and R arrays.
                block = [fh.readline() for _ in range(nlines)]
                if dtype == 'I':
                    toks = ' '.join(block).split()
                    data[name] = [int(t) for t in toks[:count]]
                elif dtype == 'R':
                    toks = ' '.join(block).split()
                    data[name] = [float(t.replace('D', 'E').replace('d', 'e'))
                                  for t in toks[:count]]
                # Character/logical blocks are consumed above but not stored.
            else:
                # Scalar field.
                try:
                    if dtype == 'I':
                        data[name] = int(rest[1])
                    elif dtype == 'R':
                        data[name] = float(rest[1].replace('D', 'E').replace('d', 'e'))
                except ValueError:
                    continue

    return data


def run_fchk(trexio_file, filename, normalized_basis=True):
    """Convert a Gaussian formatted checkpoint (.fchk) file into TREXIO.

    Supports RHF/ROHF (restricted) and UHF (unrestricted) checkpoints with
    Gaussian (s, p, sp, d, f, g) shells in either cartesian (6D/10F) or pure
    spherical (5D/7F) representations. SP shells are split into separate s and p
    shells as required by the TREXIO format.
    """
    import numpy as np

    fchk = _parse_fchk(filename)

    def require(key):
        if key not in fchk:
            raise KeyError(f"Missing '{key}' field in the fchk file {filename}.")
        return fchk[key]

    # Metadata
    # --------

    trexio.write_metadata_code_num(trexio_file, 1)
    trexio.write_metadata_code(trexio_file, ["Gaussian"])
    trexio.write_metadata_author_num(trexio_file, 1)
    trexio.write_metadata_author(trexio_file, [os.environ.get("USER", "unknown")])
    trexio.write_metadata_description(trexio_file, fchk['_title'])

    # Electrons
    # ---------

    up_num = require('Number of alpha electrons')
    dn_num = require('Number of beta electrons')
    trexio.write_electron_up_num(trexio_file, up_num)
    trexio.write_electron_dn_num(trexio_file, dn_num)

    # Nuclei
    # ------

    atomic_numbers = require('Atomic numbers')
    nucleus_num = require('Number of atoms')
    flat_coord = require('Current cartesian coordinates')  # always in bohr
    coord = [flat_coord[3 * a: 3 * a + 3] for a in range(nucleus_num)]

    if 'Nuclear charges' in fchk:
        charge = [float(z) for z in fchk['Nuclear charges']]
    else:
        charge = [float(z) for z in atomic_numbers]

    label = [PERIODIC_TABLE[z] if 0 < z < len(PERIODIC_TABLE) else "X"
             for z in atomic_numbers]

    trexio.write_nucleus_num(trexio_file, nucleus_num)
    trexio.write_nucleus_coord(trexio_file, coord)
    trexio.write_nucleus_charge(trexio_file, charge)
    trexio.write_nucleus_label(trexio_file, label)

    # Basis
    # -----

    shell_types = require('Shell types')
    prim_per_shell = require('Number of primitives per shell')
    shell_atom_map = require('Shell to atom map')          # 1-based atom index
    exps = require('Primitive exponents')
    coefs = require('Contraction coefficients')
    sp_coefs = fchk.get('P(S=P) Contraction coefficients')

    # Determine whether d and higher shells are cartesian or spherical. In an
    # fchk a positive shell type is cartesian, a negative one (other than the
    # SP code -1) is pure spherical; s, p and sp shells do not discriminate.
    has_cart = any(t >= 2 for t in shell_types)
    has_sphe = any(t <= -2 for t in shell_types)
    if has_cart and has_sphe:
        raise NotImplementedError(
            "Mixed cartesian and spherical shells are not supported by TREXIO.")
    cartesian = not has_sphe

    nucleus_index = []   # nucleus per (split) shell
    shell_ang_mom = []   # angular momentum per (split) shell
    shell_index = []     # shell per primitive
    exponent = []        # exponent per primitive
    coefficient = []     # contraction coefficient per primitive
    prim_factor = []     # primitive normalization factor
    shell_factor = []    # shell normalization factor

    def emit_shell(l, atom, these_exps, these_coefs):
        """Append one TREXIO shell (and its primitives) of momentum l."""
        shell_id = len(shell_ang_mom)
        shell_ang_mom.append(l)
        nucleus_index.append(atom)
        for e, c in zip(these_exps, these_coefs):
            shell_index.append(shell_id)
            exponent.append(e)
            coefficient.append(c)
            prim_factor.append(_prim_ref_norm(l, e))
        # Shell normalization so that the reference component integrates to 1.
        # With prim_factor[p] = N_p the stored primitives are normalized, hence
        # the self-overlap of the contraction is sum_ij c_i c_j N_i N_j S^u_ij.
        if normalized_basis:
            accum = 0.0
            norms = [_prim_ref_norm(l, e) for e in these_exps]
            for i, ci in enumerate(these_coefs):
                for j, cj in enumerate(these_coefs):
                    accum += (ci * cj * norms[i] * norms[j]
                              * _ref_overlap(l, these_exps[i], these_exps[j]))
            shell_factor.append(1.0 / np.sqrt(accum))
        else:
            shell_factor.append(1.0)

    cursor = 0
    for s, t in enumerate(shell_types):
        nprim = prim_per_shell[s]
        atom = shell_atom_map[s] - 1
        sl = slice(cursor, cursor + nprim)
        if t == -1:
            # SP (Gaussian 'L') shell: split into an s shell and a p shell.
            if sp_coefs is None:
                raise KeyError("SP shell found but P(S=P) coefficients are missing.")
            emit_shell(0, atom, exps[sl], coefs[sl])
            emit_shell(1, atom, exps[sl], sp_coefs[sl])
        else:
            emit_shell(abs(t), atom, exps[sl], coefs[sl])
        cursor += nprim

    # Every primitive must have been consumed exactly once; a mismatch means the
    # primitive/contraction arrays were truncated or parsed inconsistently.
    if cursor != len(exps):
        raise ValueError(
            f"Consumed {cursor} primitives but the fchk lists {len(exps)} "
            "primitive exponents; the basis section is inconsistent.")
    if len(coefs) != len(exps) or (sp_coefs is not None and len(sp_coefs) != len(exps)):
        raise ValueError(
            "Primitive exponent and contraction-coefficient arrays have "
            "mismatched lengths in the fchk file.")

    shell_num = len(shell_ang_mom)
    prim_num = len(exponent)

    trexio.write_basis_type(trexio_file, "Gaussian")
    trexio.write_basis_shell_num(trexio_file, shell_num)
    trexio.write_basis_prim_num(trexio_file, prim_num)
    trexio.write_basis_nucleus_index(trexio_file, nucleus_index)
    trexio.write_basis_shell_ang_mom(trexio_file, shell_ang_mom)
    trexio.write_basis_shell_index(trexio_file, shell_index)
    trexio.write_basis_shell_factor(trexio_file, shell_factor)
    # Gaussian basis functions have no r^n prefactor.
    trexio.write_basis_r_power(trexio_file, [0.0] * shell_num)
    trexio.write_basis_exponent(trexio_file, exponent)
    trexio.write_basis_coefficient(trexio_file, coefficient)
    trexio.write_basis_prim_factor(trexio_file, prim_factor)

    # AOs
    # ---

    # Within-shell component ordering as written by Gaussian. Cartesian and
    # spherical share the ordering used by the Molden converter.
    if cartesian:
        conv = [['s'], ['x', 'y', 'z'],
                ['xx', 'yy', 'zz', 'xy', 'xz', 'yz'],
                ['xxx', 'yyy', 'zzz', 'xyy', 'xxy', 'xxz', 'xzz', 'yzz', 'yyz', 'xyz'],
                ['xxxx', 'yyyy', 'zzzz', 'xxxy', 'xxxz', 'xyyy', 'yyyz', 'xzzz', 'yzzz',
                 'xxyy', 'xxzz', 'yyzz', 'xxyz', 'xyyz', 'xyzz']]
    else:
        conv = [['s'], ['p+1', 'p-1', 'p+0'],
                ['d+0', 'd+1', 'd-1', 'd+2', 'd-2'],
                ['f+0', 'f+1', 'f-1', 'f+2', 'f-2', 'f+3', 'f-3'],
                ['g+0', 'g+1', 'g-1', 'g+2', 'g-2', 'g+3', 'g-3', 'g+4', 'g-4']]

    ao_shell = []
    ao_normalization = []
    ao_ordering = []
    offset = 0
    for k, l in enumerate(shell_ang_mom):
        if l > 4:
            raise TypeError("Angular momentum l>4 is not supported by the fchk converter.")
        components = conv[l]
        ao_shell += [k for _ in components]
        accu = [(_ao_sort_key(lbl), offset + i, _component_norm(lbl))
                for i, lbl in enumerate(components)]
        accu.sort()
        ao_ordering += [idx for (_, idx, _) in accu]
        ao_normalization += [nrm for (_, _, nrm) in accu]
        offset += len(components)
    ao_num = len(ao_ordering)

    trexio.write_ao_num(trexio_file, ao_num)
    trexio.write_ao_cartesian(trexio_file, cartesian)
    trexio.write_ao_shell(trexio_file, ao_shell)
    trexio.write_ao_normalization(trexio_file, ao_normalization)

    nbasis = fchk.get('Number of basis functions')
    if nbasis is not None and nbasis != ao_num:
        raise ValueError(
            f"Computed {ao_num} AOs but the fchk reports {nbasis} basis functions.")

    # MOs
    # ---

    alpha_mo = require('Alpha MO coefficients')
    alpha_ene = fchk.get('Alpha Orbital Energies', [])
    beta_mo = fchk.get('Beta MO coefficients')
    unrestricted = beta_mo is not None

    if len(alpha_mo) % ao_num != 0:
        raise ValueError(
            f"Alpha MO coefficient count ({len(alpha_mo)}) is not a multiple of "
            f"the number of AOs ({ao_num}); the fchk file looks corrupted.")
    nmo = len(alpha_mo) // ao_num

    def reorder(flat, imo):
        vector = flat[imo * ao_num:(imo + 1) * ao_num]
        return [vector[i] for i in ao_ordering]

    mo_coefficient = []
    mo_spin = []
    mo_energy = []
    mo_occupation = []

    for imo in range(nmo):
        mo_coefficient += reorder(alpha_mo, imo)
        mo_spin.append(0)
        mo_energy.append(alpha_ene[imo] if imo < len(alpha_ene) else 0.0)
        if unrestricted:
            mo_occupation.append(1.0 if imo < up_num else 0.0)
        else:
            mo_occupation.append(2.0 if imo < dn_num
                                 else (1.0 if imo < up_num else 0.0))

    if unrestricted:
        beta_ene = fchk.get('Beta Orbital Energies', [])
        if len(beta_mo) % ao_num != 0:
            raise ValueError(
                f"Beta MO coefficient count ({len(beta_mo)}) is not a multiple "
                f"of the number of AOs ({ao_num}); the fchk file looks corrupted.")
        nmo_beta = len(beta_mo) // ao_num
        for imo in range(nmo_beta):
            mo_coefficient += reorder(beta_mo, imo)
            mo_spin.append(1)
            mo_energy.append(beta_ene[imo] if imo < len(beta_ene) else 0.0)
            mo_occupation.append(1.0 if imo < dn_num else 0.0)

    mo_class = []
    for occ in mo_occupation:
        if occ >= 2.0:
            mo_class.append("Core")
        elif occ == 0.0:
            mo_class.append("Virtual")
        else:
            mo_class.append("Active")

    trexio.write_mo_num(trexio_file, len(mo_spin))
    trexio.write_mo_coefficient(trexio_file, mo_coefficient)
    trexio.write_mo_spin(trexio_file, mo_spin)
    trexio.write_mo_occupation(trexio_file, mo_occupation)
    trexio.write_mo_energy(trexio_file, mo_energy)
    trexio.write_mo_class(trexio_file, mo_class)

    # Derive the MO type from the method field on the second line, if present.
    calc_tokens = fchk['_calc'].split()
    if len(calc_tokens) >= 2:
        trexio.write_mo_type(trexio_file, calc_tokens[1])


def run_resultsFile(trexio_file, filename_info, motype=None):
    getFile_local, a0_local, _, _ = _require_resultsfile()

    filename = filename_info['filename']
    trexio_basename = filename_info['trexio_basename']
    state_suffix = filename_info['state_suffix']
    trexio_extension = filename_info['trexio_extension']
    state_id = filename_info['state']

    try:
        res = getFile_local(filename)
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
            coord.append([a.coord[0] / a0_local, a.coord[1] / a0_local, a.coord[2] / a0_local])
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
                _, _, get_lm_local, _ = _require_resultsfile()
                l, _ = get_lm_local(b.sym)
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
            orb_list_up  = [ orb for orb in res.determinants[i].get("alpha") ]
            orb_list_dn  = [ orb for orb in res.determinants[i].get("beta") ]
            det_tmp      = [ trexio.to_bitfield_list(int64_num,orb_list_up), trexio.to_bitfield_list(int64_num,orb_list_dn) ]
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
    _, a0_local, _, resultsFile_local = _require_resultsfile()
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
            coord.append([a[2] / a0_local, a[3] / a0_local, a[4] / a0_local])
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
           contraction = resultsFile_local.contraction()
       else:
           prim_id += 1
           e, c = float(buffer[0]), float(buffer[1])
           shell_prim_index.append(prim_id)
           exponent.append(e)
           coefficient.append(c)
           gauss = resultsFile_local.gaussian()
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
        gauss = resultsFile_local.gaussian()
        gauss.center = (0.,0.,0.)
        gauss.expo = 1.0
        gauss.sym = conv[l][0]
        ref = gauss.norm
        norm.append([])
        for m in conv[l]:
            gauss = resultsFile_local.gaussian()
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

    if "gamess" not in filetype.lower():
        trexio_file = trexio.File(trexio_filename, mode='w', back_end=back_end)

    if filetype.lower() == "gaussian":
        run_resultsFile(trexio_file, filename_info, motype)

    elif filetype.lower() == "gamess":
        getFile_local, _, _, _ = _require_resultsfile()
        # Handle the case where the number of states is greater than 1
        try:
            res = getFile_local(filename)
        except Exception as exc:
            print(f"An error occurred while parsing the file using resultsFile : {exc}")
            raise

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

    elif filetype.lower() == "orca":
        back_end_str = "text" if back_end==trexio.TREXIO_TEXT else "hdf5"
        run_orca(filename=trexio_filename, orca_json=filename, back_end=back_end_str)

    elif filetype.lower() == "crystal":
        if spin is None: raise ValueError("You forgot to provide spin for the CRYSTAL->TREXIO converter.")
        back_end_str = "text" if back_end==trexio.TREXIO_TEXT else "hdf5"
        run_crystal(trexio_filename=trexio_filename, crystal_output=filename, back_end=back_end_str, spin=spin)

    elif filetype.lower() == "vlx":
        back_end_str = "text" if back_end==trexio.TREXIO_TEXT else "hdf5"
        run_vlx(vlx_h5=filename, filename=trexio_filename, back_end=back_end_str)

    elif filetype.lower() == "molden":
        run_molden(trexio_file, filename)

    elif filetype.lower() == "fchk":
        run_fchk(trexio_file, filename)

    else:
        raise NotImplementedError(f"Conversion from {filetype} to TREXIO is not supported.")
