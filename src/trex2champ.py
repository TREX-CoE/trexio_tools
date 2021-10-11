#!/usr/bin/env python3

"""
Trexio to CHAMP input converter
"""

__author__ = "Ravindra Shinde, Evgeny Posenitskiy"
__copyright__ = "Copyright 2021, The TREX Project"
__license__ = "BSD"
__version__ = "1.0.0"
__maintainer__ = "Ravindra Shinde"
__email__ = "r.l.shinde@utwente.nl"
__status__ = "Development"


import sys
import os
import numpy as np


try:
    import trexio
except:
    print("Error: The TREXIO Python library is not installed")
    sys.exit(1)

try:
    import resultsFile
except:
    print("Error: The resultsFile Python library is not installed")
    sys.exit(1)

def run(filename,  gamessfile, back_end=trexio.TREXIO_HDF5):

    trexio_file = trexio.File(filename, mode='r',back_end=trexio.TREXIO_HDF5)


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

    # Write the .xyz file containing cartesial coordinates (Bohr) of nuclei
    write_champ_file_geometry(filename, nucleus_num, nucleus_label, nucleus_coord)

    # ECP
    # ------

    ecp_z_core = trexio.read_ecp_z_core(trexio_file)
    ecp_lmax_plus_1 = trexio.read_ecp_lmax_plus_1(trexio_file)
    # ecp_local_n = trexio.read_ecp_local_n(trexio_file)
    ecp_local_num_n_max = trexio.read_ecp_local_num_n_max(trexio_file)
    ecp_local_exponent = trexio.read_ecp_local_exponent(trexio_file)
    ecp_local_coef = trexio.read_ecp_local_coef(trexio_file)
    ecp_local_power = trexio.read_ecp_local_power(trexio_file)

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
    mo_symmetry = trexio.read_mo_symmetry(trexio_file)

    # Write the .sym file containing symmetry information of MOs
    write_champ_file_symmetry(filename, mo_num, mo_symmetry)

    # Write the .orb / .lcao file containing orbital information of MOs
    write_champ_file_orbitals(filename, mo_num, ao_num, mo_coefficient)


    ###### NOTE ######
    # The following portion is written only to test few functionalities
    # It will be replaced by the data stored by trexio library.
    file = resultsFile.getFile(gamessfile)
    print(len(file.det_coefficients[0]))
    print(file.det_coefficients)
    print("CSF")
    print(len(file.csf_coefficients))
    print(file.csf_coefficients[0])

    [print(file.pseudo[i]) for i in range(len(file.pseudo))]

    write_champ_file_ecp(filename, nucleus_num, nucleus_label, file.pseudo)


    # Write the .orb / .lcao file containing orbital information of MOs
    #write_champ_file_determinants(filename, )



    return


## Champ v2.0 format input files

# Geometry
def write_champ_file_geometry(filename, nucleus_num, nucleus_label, nucleus_coord):
    """Writes the geometry data from the quantum
    chemistry calculation to a champ v2.0 format file.

    Returns:
        None as a function value
    """

    if filename is not None:
        if isinstance(filename, str):
            ## Write down a geometry file in the new champ v2.0 format
            filename_geometry = os.path.splitext("champ_v2_" + filename)[0]+'_geom.xyz'
            with open(filename_geometry, 'w') as file:

                file.write("{} \n".format(nucleus_num))
                # header line printed below
                file.write("# Converted from the trexio file using trex2champ converter https://github.com/TREX-CoE/trexio_tools \n")

                for element in range(nucleus_num):
                   file.write("{:5s} {: 0.6f} {: 0.6f} {: 0.6f} \n".format(nucleus_label[element], nucleus_coord[element][0], nucleus_coord[element][1], nucleus_coord[element][2]))

                file.write("\n")
            file.close()
        else:
            raise ValueError
    # If filename is None, return a string representation of the output.
    else:
        return None

# Symmetry
def write_champ_file_symmetry(filename,mo_num, mo_symmetry):
    """Writes the symmetry information of molecular orbitals from the quantum
    chemistry calculation to the new champ v2.0 input file format.

    Returns:
        None as a function value
    """

    if filename is not None:
        if isinstance(filename, str):
            ## Write down a symmetry file in the new champ v2.0 format
            filename_symmetry = os.path.splitext("champ_v2_" + filename)[0]+'_symmetry.sym'
            with open(filename_symmetry, 'w') as file:

                values, counts = np.unique(mo_symmetry, return_counts=True)
                # point group symmetry independent line printed below
                file.write("sym_labels " + str(len(counts)) + " " + str(mo_num)+"\n")

                irrep_string = ""
                irrep_correspondence = {}
                for i, val in enumerate(values):
                    irrep_correspondence[val] = i+1
                    irrep_string += " " + str(i+1) + " " + str(val)

                if all(irreps in mo_symmetry for irreps in values):
                    file.write(f"{irrep_string} \n")   # This defines the rule

                    for item in mo_symmetry:
                        for key, val in irrep_correspondence.items():
                            if item == key:
                                file.write(str(val)+" ")
                    file.write("\n")
                file.write("end\n")
            file.close()

        else:
            raise ValueError
    # If filename is None, return a string representation of the output.
    else:
        return None


# Orbitals / LCAO infomation

def write_champ_file_orbitals(filename, mo_num, ao_num, mo_coefficient):
    """Writes the molecular orbitals coefficients from the quantum
    chemistry calculation / trexio file to champ v2.0 input file format.

    Returns:
        None as a function value
    """

    if filename is not None:
        if isinstance(filename, str):
            ## Write down an orbitals file in the new champ v2.0 format
            filename_orbitals = os.path.splitext("champ_v2_" + filename)[0]+'_orbitals.orb'
            with open(filename_orbitals, 'w') as file:

                # header line printed below
                file.write("# File created using the trex2champ converter https://github.com/TREX-CoE/trexio_tools  \n")
                file.write("lcao " + str(mo_num) + " " + str(ao_num) + " 1 " + "\n" )
                np.savetxt(file, mo_coefficient)
                file.write("end\n")
            file.close()
        else:
            raise ValueError
    # If filename is None, return a string representation of the output.
    else:
        return None


# ECP / Pseudopotential files
def write_champ_file_ecp(filename, nucleus_num, nucleus_label, pseudo):
    """Writes the Gaussian - effective core potential / pseudopotential data from
    the quantum chemistry calculation to a champ v2.0 format file.

    Returns:
        None as a function value
    """

    if filename is not None:
        if isinstance(filename, str):
            unique_elements, indices = np.unique(nucleus_label, return_index=True)
            for i in range(len(unique_elements)):
                # Write down an ECP file in the new champ v2.0 format for each nucleus
                filename_ecp = "BFD." + 'gauss_ecp.dat.' + unique_elements[i]
                with open(filename_ecp, 'w') as file:
                    file.write("BFD {:s} pseudo \n".format(unique_elements[i]))

                    lmax_plus_one = pseudo[i].get("lmax") + 1
                    file.write("{} \n".format(lmax_plus_one))

                    # Write down the pseudopotential data
                    components = len(pseudo[i].get("1"))
                    file.write("{} \n".format(components))

                    for j in range(components):
                        file.write( "{} {} {} \n" .format(pseudo[i].get("1")[j][0], pseudo[i].get("1")[j][1] , pseudo[i].get("1")[j][2]))

                    components = len(pseudo[i].get("0"))
                    file.write("{} \n".format(components))

                    for j in range(components):
                        file.write( "{} {} {} \n" .format(pseudo[i].get("0")[j][0], pseudo[i].get("0")[j][1] , pseudo[i].get("0")[j][2]))


                file.close()
        else:
            raise ValueError
    # If filename is None, return a string representation of the output.
    else:
        return None




# # ECP / Pseudopotential files using the trexio file
# def write_champ_file_ecp(filename, nucleus_num, nucleus_label, ecp_z_core, ecp_local_num_n_max, ecp_local_exponent, ecp_local_coef, ecp_local_power, ecp_lmax_plus_1):
#     """Writes the Gaussian - effective core potential / pseudopotential data from
#     the quantum chemistry calculation to a champ v2.0 format file.

#     Returns:
#         None as a function value
#     """

#     if filename is not None:
#         if isinstance(filename, str):
#             unique_elements, indices = np.unique(nucleus_label, return_index=True)
#             for i in range(len(unique_elements)):
#                 # Write down an ECP file in the new champ v2.0 format for each nucleus
#                 filename_ecp = "BFD." + 'gauss_ecp.dat.' + unique_elements[i]
#                 with open(filename_ecp, 'w') as file:
#                     file.write("BFD {:s} pseudo \n".format(unique_elements[i]))
#                     file.write("{:d} \n".format(ecp_local_num_n_max))

#                     flattened_ecp_local_coef = ecp_local_coef.flatten()
#                     flattened_ecp_local_power = ecp_local_power.flatten()
#                     flattened_ecp_local_exponent = ecp_local_exponent.flatten()

#                     for j in range(len(indices)):
#                         file.write("{:.8f} ".format(flattened_ecp_local_coef[j*nucleus_num]))
#                         file.write("{} ".format(flattened_ecp_local_power[j*nucleus_num]+2))
#                         file.write("{:.8f} \n".format(flattened_ecp_local_exponent[j*nucleus_num]))

#                     # file.write("{} ".format(ecp_local_coef))
#                     # file.write("{} \n".format(ecp_local_power))
#                     # file.write("{} \n".format(ecp_local_exponent))
#                     # file.write("{} \n".format(ecp_lmax_plus_1))

#                 file.close()

#             # ## Write down a geometry file in the new champ v2.0 format
#             # filename_ecp = os.path.splitext("champ_v2_" + filename)[0]+'_geom.xyz'
#             # with open(filename_ecp, 'w') as file:

#             #     file.write("{} \n".format(nucleus_num))
#             #     # header line printed below
#             #     file.write("# Converted from the trexio file using trex2champ converter https://github.com/TREX-CoE/trexio_tools \n")

#             #     for element in range(nucleus_num):
#             #        file.write("{:5s} {: 0.6f} {: 0.6f} {: 0.6f} \n".format(nucleus_label[element], nucleus_coord[element][0], nucleus_coord[element][1], nucleus_coord[element][2]))

#             #     file.write("\n")
#             # file.close()
#         else:
#             raise ValueError
#     # If filename is None, return a string representation of the output.
#     else:
#         return None