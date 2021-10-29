#!/usr/bin/env python3
#   trex2champ is a tool which allows to read output files of quantum
#   chemistry codes (GAMESS and trexio files) and write input files for
#   CHAMP in V2.0 format.
#
# Copyright (c) 2021, TREX Center of Excellence
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#   Ravindra Shinde
#   University of Twente
#   Enschede, The Netherlands
#   r.l.shinde@utwente.nl


__author__ = "Ravindra Shinde, Evgeny Posenitskiy"
__copyright__ = "Copyright 2021, The TREX Project"
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

    ecp_num = trexio.read_ecp_num(trexio_file)
    ecp_z_core = trexio.read_ecp_z_core(trexio_file)
    ecp_max_ang_mom_plus_1 = trexio.read_ecp_max_ang_mom_plus_1(trexio_file)
    ecp_ang_mom = trexio.read_ecp_ang_mom(trexio_file)
    ecp_nucleus_index = trexio.read_ecp_nucleus_index(trexio_file)
    ecp_exponent = trexio.read_ecp_exponent(trexio_file)
    ecp_coefficient = trexio.read_ecp_coefficient(trexio_file)
    ecp_power = trexio.read_ecp_power(trexio_file)

    write_champ_file_ecp_trexio(filename, nucleus_num, nucleus_label, ecp_num, ecp_z_core, ecp_max_ang_mom_plus_1, ecp_ang_mom, ecp_nucleus_index, ecp_exponent, ecp_coefficient, ecp_power)

    # Basis

    basis_type = trexio.read_basis_type(trexio_file)
    basis_shell_num = trexio.read_basis_shell_num(trexio_file)
    basis_prim_num = trexio.read_basis_prim_num(trexio_file)
    basis_nucleus_index = trexio.read_basis_nucleus_index(trexio_file)
    basis_shell_ang_mom = trexio.read_basis_shell_ang_mom(trexio_file)
    basis_shell_factor = trexio.read_basis_shell_factor(trexio_file)
    basis_shell_index = trexio.read_basis_shell_index(trexio_file)
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

    write_champ_file_determinants(filename, file)

    write_champ_file_ecp(filename, nucleus_num, nucleus_label, file.pseudo)

    return


## Champ v2.0 format input files

def write_champ_file_determinants(filename, file):
    """Writes the determinant data from the quantum
    chemistry calculation to a champ v2.0 format file.

    Returns:
        None as a function value
    """
    det_coeff = file.det_coefficients
    csf_coeff = file.csf_coefficients
    num_csf = len(csf_coeff[0])
    num_states = file.num_states
    num_dets = len(det_coeff[0])
    num_alpha = len(file.determinants[0].get("alpha"))
    num_beta = len(file.determinants[0].get("beta"))


    if filename is not None:
        if isinstance(filename, str):
            ## Write down a determinant file in the new champ v2.0 format
            filename_determinant = os.path.splitext("champ_v2_" + filename)[0]+'_determinants.det'
            with open(filename_determinant, 'w') as f:
                # header line printed below
                f.write("# Determinants, CSF, and CSF mapping from the GAMESS output / TREXIO file. \n")
                f.write("# Converted from the trexio file using trex2champ converter https://github.com/TREX-CoE/trexio_tools \n")
                f.write("determinants {} {} \n".format(num_dets, num_states))

                # print the determinant coefficients
                for det in range(num_dets):
                    f.write("{:.8f} ".format(det_coeff[0][det]))
                f.write("\n")

                # print the determinant orbital mapping
                for det in range(num_dets):
                    for num in range(num_alpha):
                        f.write("{:4d} ".format(file.determinants[det].get("alpha")[num]+1))
                    f.write("  ")
                    for num in range(num_beta):
                        f.write("{:4d} ".format(file.determinants[det].get("beta")[num]+1))
                    f.write("\n")
                f.write("end \n")

                # print the CSF coefficients
                f.write("csf {} {} \n".format(num_csf, num_states))
                for state in range(num_states):
                    for ccsf in range(num_csf):
                        f.write("{:.8f} ".format(csf_coeff[state][ccsf]))
                    f.write("\n")
                f.write("end \n")

                f.write("\n")
            f.close()
        else:
            raise ValueError
    # If filename is None, return a string representation of the output.
    else:
        return None




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
            # Find the pseudos for unique elements
            ind = next((index for (index, d) in enumerate(pseudo) if d["atom"] == indices[2]), None)
            # if pseudo["atom"] == int(indices[2]):

            for ind in indices:
                atom_index = pseudo[ind].get("atom")

                # Write down an ECP file in the new champ v2.0 format for each nucleus
                filename_ecp = "BFD." + 'gauss_ecp.dat.' + nucleus_label[ind]

                with open(filename_ecp, 'w') as file:
                    file.write("BFD {:s} pseudo \n".format(nucleus_label[ind]))

                    lmax_plus_one = pseudo[ind].get("lmax") + 1
                    file.write("{} \n".format(lmax_plus_one))

                    # Write down the pseudopotential data
                    if pseudo[ind].get("zcore") >= 2:
                        components = len(pseudo[ind].get("1"))
                        file.write("{} \n".format(components))

                    if pseudo[ind].get("zcore") >= 2:
                        for j in range(components):
                            file.write( "{:.8f} {:.8f} {:.8f} \n" .format(pseudo[ind].get("1")[j][0], pseudo[ind].get("1")[j][1] , pseudo[ind].get("1")[j][2]))

                    if pseudo[ind].get("zcore") > 0 or pseudo[ind].get("lmax") >= 0:
                        components = len(pseudo[ind].get("0"))
                        file.write("{} \n".format(components))

                    for j in range(components):
                        file.write( "{:.8f} {:.8f} {:.8f} \n" .format(pseudo[ind].get("0")[j][0], pseudo[ind].get("0")[j][1] , pseudo[ind].get("0")[j][2]))

                file.close()
        else:
            raise ValueError
    # If filename is None, return a string representation of the output.
    else:
        return None




# ECP / Pseudopotential files using the trexio file
def write_champ_file_ecp_trexio(filename, nucleus_num, nucleus_label, ecp_num, ecp_z_core, ecp_max_ang_mom_plus_1, ecp_ang_mom, ecp_nucleus_index, ecp_exponent, ecp_coefficient, ecp_power):
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
                filename_ecp = "trexBFD." + 'gauss_ecp.dat.' + unique_elements[i]
                with open(filename_ecp, 'w') as file:
                    file.write("BFD {:s} pseudo \n".format(unique_elements[i]))

                    dict_ecp={}
                    # get the indices of the ecp data for each atom
                    for ind, val in enumerate(ecp_nucleus_index):
                        if val == indices[i]:
                            dict_ecp[ind] = [ecp_ang_mom[ind], ecp_coefficient[ind], ecp_power[ind]+2, ecp_exponent[ind]]

                    ecp_array =  np.array(list(dict_ecp.values()))
                    ecp_array = ecp_array[np.argsort(ecp_array[:,0])]

                    sorted_list = np.sort(ecp_array[:,0])[::-1]

                    np.savetxt(file, [len(np.unique(sorted_list))], fmt='%d')
                    # loop over ang mom for a given atom
                    for l in np.sort(np.unique(sorted_list))[::-1]:
                        # loop and if condition to choose the correct components
                        for x in np.unique(np.sort(ecp_array[:,0])[::-1]):
                            if ecp_array[int(x):,0:][0][0] == l:
                                count = np.count_nonzero(sorted_list == l)
                                np.savetxt(file, [count], fmt='%d')
                                np.savetxt(file, ecp_array[int(x):count+int(x),1:], fmt='%.8f')

                file.close()

        else:
            raise ValueError
    # If filename is None, return a string representation of the output.
    else:
        return None