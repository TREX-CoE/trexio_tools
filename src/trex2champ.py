#!/usr/bin/env python3
"""
Trexio to CHAMP input converter
"""

import sys
import os

try:
    import trexio
except:
    print("Error: The TREXIO Python library is not installed")
    sys.exit(1)



def run(trexio_filename,filename):

    trexio_file = trexio.File(trexio_filename,mode='r',back_end=trexio.TREXIO_HDF5)


    # Metadata
    # --------

    metadata_num = trexio.read_metadata_code_num(trexio_file)
    metadata_code  = trexio.read_metadata_code(trexio_file)
    metadata_description = trexio.read_metadata_description(trexio_file)

    # Electrons
    # ---------

    electron_up_num = trexio.read_electron_up_num(trexio_file)
    electron_dn_num = trexio.read_electron_dn_num(trexio_file)

    # Nuclei
    # ------

    nucleus_num = trexio.read_nucleus_num(trexio_file)
    nucleus_charge = trexio.read_nucleus_charge(trexio_file)
    nucleus_coord = trexio.read_nucleus_coord(trexio_file)
    nucleus_label = trexio.read_nucleus_label(trexio_file)
    nucleus_point_group = trexio.read_nucleus_point_group(trexio_file)

    # ECP
    # ------

    # ecp_z_core = trexio.read_ecp_z_core(trexio_file)
    # ecp_local_n = trexio.read_ecp_local_n(trexio_file)
    # ecp_local_num_n_max = trexio.read_ecp_local_num_n_max(trexio_file)
    # ecp_local_exponent = trexio.read_ecp_local_exponent(trexio_file)
    # ecp_local_coef = trexio.read_ecp_local_coef(trexio_file)
    # ecp_local_power = trexio.read_ecp_local_power(trexio_file)


    # Basis

    basis_type = trexio.read_basis_type(trexio_file)
    basis_num = trexio.read_basis_num(trexio_file)
    basis_prim_num = trexio.read_basis_prim_num(trexio_file)
    basis_nucleus_index = trexio.read_basis_nucleus_index(trexio_file)
    basis_nucleus_shell_num = trexio.read_basis_nucleus_shell_num(trexio_file)
    basis_shell_ang_mom = trexio.read_basis_shell_ang_mom(trexio_file)
    basis_shell_prim_num = trexio.read_basis_shell_prim_num(trexio_file)
    basis_shell_factor = trexio.read_basis_shell_factor(trexio_file)
    basis_shell_prim_index = trexio.read_basis_shell_prim_index(trexio_file)
    basis_exponent = trexio.read_basis_exponent(trexio_file)
    basis_coefficient = trexio.read_basis_coefficient(trexio_file)
    basis_prim_factor = trexio.read_basis_prim_factor(trexio_file)

    # AO
    # --

    ao_cartesian = trexio.read_ao_cartesian(trexio_file)
    ao_num = trexio.read_ao_num(trexio_file)
    ao_shell = trexio.read_ao_shell(trexio_file)
    ao_normalization = trexio.read_ao_normalization(trexio_file)


    # MOs
    # ---

    mo_type = trexio.read_mo_type(trexio_file)
    mo_num = trexio.read_mo_num(trexio_file)
    mo_coefficient = trexio.read_mo_coefficient(trexio_file)

    return

