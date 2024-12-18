def orca_to_trexio(
        orca_json: str = "orca.json",
        filename: str = "trexio.hdf5",
        back_end: str = "hdf5"
):
    # load python packages
    import os
    import json
    import numpy as np
    import scipy.special

    # ## ORCA -> TREX-IO
    # - how to install trexio
    # - pip install trexio

    # import trexio
    import trexio

    # Logger
    from logging import getLogger
    logger = getLogger("orca-trexio").getChild(__name__)

    logger.info(f"orca_json = {orca_json}")
    logger.info(f"trexio_filename = {filename}")
    logger.info("Conversion starts...")


    with open(orca_json, 'r') as f:
        data = json.load(f)

    # trexio back end handling
    if back_end.lower() == "hdf5":
        trexio_back_end = trexio.TREXIO_HDF5
    elif back_end.lower() == "text":
        trexio_back_end = trexio.TREXIO_TEXT
    else:
        raise NotImplementedError(f"{back_end} back-end is not supported.")

    trexio_file = trexio.File(filename, mode='w', back_end=trexio_back_end)

    natoms = len(data["Molecule"]["Atoms"])
    #print(f"Natoms={natoms}")
    coord = []
    chemical_symbol_list = []
    atom_charges_list = []
    ecp_charge_list = []
    ecp_charge = 0
    elec_num = 0
    total_charge = data["Molecule"]["Charge"]
    multiplicity = data["Molecule"]["Multiplicity"]
    orbital_labels = data["Molecule"]["MolecularOrbitals"]["OrbitalLabels"]
    has_ecp = []
    for i in range(natoms):
        atom = data["Molecule"]["Atoms"][i]
        coord.append(atom["Coords"])
        chemical_symbol_list.append(atom["ElementLabel"])
        elec_num += atom["NuclearCharge"]
        try:
            atom["ECPs"]
            has_ecp.append(True)
            N_core = atom["ECPs"]["N_core"]
            ecp_charge_list.append(N_core)
            ecp_charge += N_core
        except:
            has_ecp.append(False)
        atom_charges_list.append(atom["NuclearCharge"])
    elec_num = elec_num - total_charge - ecp_charge

    # Check coordinate type
    coord_type = data["Molecule"]["CoordinateUnits"]
    coord = np.array(coord)
    if coord_type == "Angs":
        # Convert to Bohrs
        for i in range(natoms):
            coord[i] = 1.8897259886 * coord[i]

    # Assuming multiplicity = (elec_up - elec_dn) + 1
    electron_up_num = int((elec_num + multiplicity - 1)//2)
    electron_dn_num = int(elec_num - electron_up_num)

    ##########################################
    # Structure info
    ##########################################
    trexio.write_electron_up_num(trexio_file, electron_up_num)
    trexio.write_electron_dn_num(trexio_file, electron_dn_num)
    trexio.write_nucleus_num(trexio_file, len(coord))
    trexio.write_nucleus_coord(trexio_file, coord)
    trexio.write_nucleus_charge(trexio_file, atom_charges_list)
    trexio.write_nucleus_label(trexio_file, chemical_symbol_list)
    ##########################################
    # basis set info
    ##########################################
    # check the orders of the spherical atomic basis in orca!!
    # L, ml
    # pz, px, py                    = (0,+1,-1)
    # dz2, dxz, dyz, dx2y2, dxy     = (0,+1,-1,+2,-2)
    # gto.spheric_labels(mol, fmt="%d, %s, %s, %s")
    # ORCA ordering                      -- TREXIO ordering
    # -----------------------------------------------------
    # for s -> s                         -- trexio: (0)
    # for p -> pz, px, py                -- trexio: (-1,0,1)
    # for d -> dz2, dxz, dyz, dx2y2, dxy -- trexio: (-2, -1, 0, 1, 2)
    # for l -> m=(0 -1 +1 -2 +2 ... -l +l) -- (-l, ..., 0, ..., +l)
    nucleus_num = natoms
    atom_nshells = []
    atom_shell_ids = []
    bas_angular = []
    bas_nprim = []
    bas_ctr_coeff = []
    bas_exp = []
    basis_shell_num = 0
    for i in range(nucleus_num):
        atom = data["Molecule"]["Atoms"][i]
        nshells = len(atom["Basis"])
        atom_nshells.append(nshells)
        shell_ids = []
        for k in range(nshells):
            shell_ids.append(i)
            bas_angular.append(atom["Basis"][k]["Shell"])
            bas_nprim.append(len(atom["Basis"][k]["Exponents"]))
            bas_exp.append(atom["Basis"][k]["Exponents"])
            bas_ctr_coeff.append(atom["Basis"][k]["Coefficients"])
        atom_shell_ids.append(shell_ids)

    try:
        S_matrix = np.array(data["Molecule"]["S-Matrix"])
        T_matrix = np.array(data["Molecule"]["T-Matrix"])
        H_matrix = np.array(data["Molecule"]["H-Matrix"])
        readS = True
    except:
        readS = False
    dict_ang_mom = dict()
    dict_ang_mom['s'] = 0
    dict_ang_mom['p'] = 1
    dict_ang_mom['d'] = 2
    dict_ang_mom['f'] = 3
    dict_ang_mom['g'] = 4
    dict_ang_mom['h'] = 5
    dict_ang_mom['i'] = 6

    basis_type = "Gaussian"
    basis_shell_num = int(np.sum([atom_nshells[i] for i in range(nucleus_num)]))
    nucleus_index = []
    for i in range(nucleus_num):
        for _ in range(len(atom_shell_ids[i])):
            nucleus_index.append(i)
    shell_ang_mom = [dict_ang_mom[bas_angular[i]] for i in range(basis_shell_num)]
    basis_prim_num = int(np.sum([bas_nprim[i] for i in range(basis_shell_num)]))

    basis_exponent = []
    basis_coefficient = []
    for i in range(basis_shell_num):
        for bas_exp_i in bas_exp[i]:
            basis_exponent.append(float(bas_exp_i))
        for bas_ctr_coeff_i in bas_ctr_coeff[i]:
            basis_coefficient.append(float(bas_ctr_coeff_i))

    basis_shell_index = []
    for i in range(basis_shell_num):
        for _ in range(len(bas_exp[i])):
            basis_shell_index.append(i)

    # normalization factors
    basis_shell_factor = [1.0 for _ in range(basis_shell_num)]  # 1.0 in ORCA

    # power of r is always zero for Gaussian functions
    basis_r_power = [0.0 for _ in range(basis_shell_num) ]

    def my_factorial2(n):
        if n < 0:
            return(1)
        else:
            return(scipy.special.factorial2(n))


    def gto_norm(alpha, ax, ay, az):
        val = ((alpha + alpha)/np.pi)**(3/4)*((4*alpha)**((ax + ay + az)/2))/((my_factorial2(2*ax - 1) * \
                                                                        my_factorial2(2*ay - 1) * \
                                                                        my_factorial2(2*az - 1))**(1/2))
        return(val)

    # gto_norm(l, expnt) => l is angmom, expnt is exponent
    # Note!! Here, the normalization factor of the spherical part
    # are not included. The normalization factor is computed according
    # to Eq.8 of the following paper
    # H.B.S and M.J.F, Int. J. Quant.  Chem., 54(1995), 83-87.
    basis_prim_factor = []
    for prim_i in range(basis_prim_num):
        coeff = basis_coefficient[prim_i]
        expnt = basis_exponent[prim_i]
        l_num = shell_ang_mom[basis_shell_index[prim_i]]
        basis_prim_factor.append(
            gto_norm(expnt, l_num, 0, 0)
        )

    ##########################################
    # basis set info
    ##########################################
    trexio.write_basis_type(trexio_file, basis_type)  #
    trexio.write_basis_shell_num(trexio_file, basis_shell_num)  #
    trexio.write_basis_prim_num(trexio_file, basis_prim_num)  #
    trexio.write_basis_nucleus_index(trexio_file, nucleus_index)  #
    trexio.write_basis_shell_ang_mom(trexio_file, shell_ang_mom)  #
    trexio.write_basis_shell_factor(trexio_file, basis_shell_factor)  #
    trexio.write_basis_r_power(trexio_file, basis_r_power)  #
    trexio.write_basis_shell_index(trexio_file, basis_shell_index)  #
    trexio.write_basis_exponent(trexio_file, basis_exponent)  #
    trexio.write_basis_coefficient(trexio_file, basis_coefficient)  #
    trexio.write_basis_prim_factor(trexio_file, basis_prim_factor)  #
    ##########################################
    # ao info
    ##########################################
    # to be fixed!! for Victor case mol.cart is false, but the basis seems cartesian...
    cart = False
    if cart:
        ao_cartesian = 1
    else:
        ao_cartesian = 0  # spherical basis representation
    ao_shell = []
    for i, ang_mom in enumerate(shell_ang_mom):
        for _ in range(2 * ang_mom + 1):
            ao_shell.append(i)
    ao_num = len(ao_shell)

    # 1.0 in pyscf (because spherical)
    ao_normalization = [1.0 for _ in range(ao_num)]

    ##########################################
    # ao info
    ##########################################
    trexio.write_ao_cartesian(trexio_file, ao_cartesian)  #
    trexio.write_ao_num(trexio_file, ao_num)  #
    trexio.write_ao_shell(trexio_file, ao_shell)  #
    trexio.write_ao_normalization(trexio_file, ao_normalization)  #
    if readS:
        trexio.write_ao_1e_int_overlap(trexio_file, S_matrix)
        trexio.write_ao_1e_int_kinetic(trexio_file, T_matrix)
        trexio.write_ao_1e_int_potential_n_e(trexio_file, H_matrix)

    ##########################################
    # mo info
    ##########################################
    mo_type = "MO"

    mo_occupation_read = []
    mo_energy_read     = []
    mo_coeff_read      = []
    for k in data['Molecule']['MolecularOrbitals']['MOs']:
        mo_occupation_read.append(k['Occupancy'])
        mo_coeff_read.append(k['MOCoefficients'])
        mo_energy_read.append(k['OrbitalEnergy'])

    # check if the pySCF calculation is Restricted or Unrestricted
    # Restricted -> RHF,RKS,ROHF,OROKS
    # Unrestricted -> UHF,UKS

    if len(mo_energy_read) == 2:
        if isinstance(mo_energy_read[0], float):
            spin_restricted = True
        else:
            spin_restricted = False
    else:
        spin_restricted = True

    # the followins are given to TREXIO file lager if spin_restricted == False,
    mo_coefficient_all = []
    mo_occupation_all = []
    mo_energy_all = []
    mo_spin_all = []

    # mo read part starts both for alpha and beta spins
    for ns, spin in enumerate([0, 1]):

        if spin_restricted:
            mo_occupation = mo_occupation_read
            mo_energy = mo_energy_read
            mo_coeff = mo_coeff_read
            if spin == 1:  # 0 is alpha(up), 1 is beta(dn)
                logger.info("This is spin-restricted calculation.")
                logger.info("Skip the MO conversion step for beta MOs.")
                break
        else:
            logger.info(
                f"MO conversion step for {spin}-spin MOs. 0 is alpha(up), 1 is beta(dn)."
            )
            mo_occupation = mo_occupation_read[ns]
            mo_energy = mo_energy_read[ns]
            mo_coeff = mo_coeff_read[ns]

        mo_num = len(mo_coeff[0])

        mo_spin_all += [spin for _ in range(mo_num)]


        logger.debug(mo_num)
        logger.debug(len(mo_coeff))
        logger.debug(mo_occupation)
        logger.debug(mo_energy)
        # logger.info(mo_coeff)

        # check if MOs are descending order with respect to "mo occ"
        # this is usually true, but not always true for
        # RO (restricted open-shell) calculations.
        order_bool = all(
            [
                True if mo_occupation[i] >= mo_occupation[i + 1] else False
                for i in range(len(mo_occupation) - 1)
            ]
        )
        logger.info(f"MO occupations are in the descending order ? -> {order_bool}")
        if not order_bool:
            logger.warning("MO occupations are not in the descending order!!")
            logger.warning("RO (restricted open-shell) calculations?")
            logger.warning("Reordering MOs...")
            # reordering MOs.
            # descending order (mo occ)
            reo_moocc_index = np.argsort(mo_occupation)[::-1]
            mo_occupation_o = [mo_occupation[l_num] for l_num in reo_moocc_index]
            mo_energy_o = [mo_energy[l_num] for l_num in reo_moocc_index]
            mo_coeff_o = [mo_coeff[l_num] for l_num in reo_moocc_index]
            # descending order (mo energy)
            mo_coeff = []
            mo_occupation = []
            mo_energy = []
            set_mo_occupation = sorted(list(set(mo_occupation_o)), reverse=True)
            for mo_occ in set_mo_occupation:
                mo_re_index = [
                    i for i, mo in enumerate(mo_occupation_o) if mo == mo_occ
                ]
                mo_occupation_t = [mo_occupation_o[l_num] for l_num in mo_re_index]
                mo_energy_t = [mo_energy_o[l_num] for l_num in mo_re_index]
                mo_coeff_t = [mo_coeff_o[l_num] for l_num in mo_re_index]
                reo_ene_index = np.argsort(mo_energy_t)
                mo_occupation += [mo_occupation_t[l_num] for l_num in reo_ene_index]
                mo_energy += [mo_energy_t[l_num] for l_num in reo_ene_index]
                mo_coeff += [mo_coeff_t[l_num] for l_num in reo_ene_index]

        logger.debug("--mo_num--")
        logger.debug(mo_num)
        logger.debug("--len(mo_coeff)--")
        logger.debug(len(mo_coeff))
        logger.debug("--mo_occupation--")
        logger.debug(mo_occupation)
        logger.debug("--mo_energy--")
        logger.debug(mo_energy)
        # logger.debug(mo_coeff)

        # saved mo_occ and mo_energy
        mo_occupation_all += list(mo_occupation)
        mo_energy_all += list(mo_energy)

        # permutation_matrix = []  # for ao and mo swaps, used later

        # molecular coefficient reordering
        # TREX-IO employs (m=-l,..., 0, ..., +l) for spherical basis
        mo_coefficient = []

        for mo_i in range(mo_num):
            mo = mo_coeff[mo_i]
            mo_coeff_buffer = []

            perm_list = []
            perm_n = 0
            for ao_i, ao_c in enumerate(mo):

                # initialization
                if ao_i == 0:
                    mo_coeff_for_reord = []
                    current_ang_mom = -1

                # read ang_mom (i.e., angular momentum of the shell)
                bas_i = ao_shell[ao_i]
                ang_mom = shell_ang_mom[bas_i]

                previous_ang_mom = current_ang_mom
                current_ang_mom = ang_mom

                # set multiplicity
                multiplicity = 2 * ang_mom + 1
                # print(f"multiplicity = {multiplicity}")

                # check if the buffer is null, when ang_mom changes
                if previous_ang_mom != current_ang_mom:
                    assert len(mo_coeff_for_reord) == 0

                if current_ang_mom == 0:  # s shell
                    # print("s shell/no permutation is needed.")
                    # print("(pyscf notation): s(l=0)")
                    # print("(trexio notation): s(l=0)")
                    reorder_index = [0]

                elif current_ang_mom == 1:  # p shell

                    # print("p shell/permutation is needed.")
                    # print("(pyscf notation): px(l=+1), py(l=-1), pz(l=0)")
                    # print("(trexio notation): pz(l=0), px(l=+1), py(l=-1)")
                    reorder_index = [2, 0, 1]

                elif current_ang_mom >= 2:  # > d shell

                    # print("> d shell/permutation is needed.")
                    # print(
                    #    "(pyscf) e.g., f3,-3(l=-3), f3,-2(l=-2), f3,-1(l=-1), \
                    #        f3,0(l=0), f3,+1(l=+1), f3,+2(l=+2), f3,+3(l=+3)"
                    # )
                    # print(
                    #    "(trexio) e.g, f3,0(l=0), f3,+1(l=+1), f3,-1(l=-1), \
                    #        f3,+2(l=+2), f3,-2(l=-2), f3,+3(l=+3), f3,-3(l=-3)"
                    # )
                    l0_index = int((multiplicity - 1) / 2)
                    reorder_index = [l0_index]
                    for i in range(1, int((multiplicity - 1) / 2) + 1):
                        reorder_index.append(l0_index + i)
                        reorder_index.append(l0_index - i)

                else:
                    raise ValueError("A wrong value was set to current_ang_mom.")

                mo_coeff_for_reord.append(ao_c)

                # write MOs!!
                if len(mo_coeff_for_reord) == multiplicity:
                    # print("--write MOs!!--")
                    mo_coeff_buffer += [
                        mo_coeff_for_reord[i] for i in reorder_index
                    ]

                    # reset buffer
                    mo_coeff_for_reord = []

                    # print("--write perm_list")
                    perm_list += list(np.array(reorder_index) + perm_n)
                    perm_n = perm_n + len(reorder_index)

            mo_coefficient.append(mo)
            # permutation_matrix.append(perm_list)

        mo_coefficient_all += mo_coefficient

    # MOs read part end both for alpha and beta spins[l]
    logger.debug("len(mo_coefficient_all)")
    logger.debug(len(mo_coefficient_all))
    logger.debug("len(mo_occupation_all)")
    logger.debug(len(mo_occupation_all))
    logger.debug("len(mo_spin_all)")
    logger.debug(len(mo_spin_all))

    # Conversion from Python complex -> real, complex separately.
    # force WF complex
    force_wf_complex = False
    if force_wf_complex:
        complex_flag = True
    # check if the MOs have imag.!
    else:
        imag_flags = []
        for mo in mo_coefficient_all:
            imag_flags += list(np.isreal(list(np.real_if_close(mo, tol=100))))
        # print(imag_flags)
        if all(imag_flags):
            complex_flag = False
        else:
            complex_flag = True

    if complex_flag:
        logger.info("The WF is complex")
        mo_coefficient_real = []
        mo_coefficient_imag = []

        for mo__ in mo_coefficient_all:
            mo_real_b = []
            mo_imag_b = []
            for coeff in mo__:
                mo_real_b.append(coeff.real)
                mo_imag_b.append(coeff.imag)
            mo_coefficient_real.append(mo_real_b)
            mo_coefficient_imag.append(mo_imag_b)

    else:
        logger.info("The WF is real")
        mo_coefficient_real = [list(np.array(mo).real) for mo in mo_coefficient_all]

    logger.debug("--MOs Done--")
    ##########################################
    # mo info
    ##########################################
    trexio.write_mo_type(trexio_file, mo_type)  #

    if complex_flag:
        trexio.write_mo_num(trexio_file, len(mo_coefficient_real))  #
        trexio.write_mo_coefficient(trexio_file, mo_coefficient_real)  #
        trexio.write_mo_coefficient_im(trexio_file, mo_coefficient_imag)  #
    else:
        trexio.write_mo_num(trexio_file, len(mo_coefficient_real))  #
        trexio.write_mo_coefficient(trexio_file, mo_coefficient_real)  #

    trexio.write_mo_occupation(trexio_file, mo_occupation_all)  #

    trexio.write_mo_spin(trexio_file, mo_spin_all)  #
    ##########################################
    # ECP
    ##########################################
    # internal format of ORCA
    # See Manual 5.0.4 pg : 497
    """
    -------------
    atom:
      nelec  (number of core electrons)
      lmax (max. angular momentum for Ul to indicate |l><l|)
      [shells]
        [Exp]
        (exp_1)
        (exp_2)
        ...
        [Coef]
        (coef_1)
        (coef_2)
        ...
        [Radial power r^p]
        (p_1)
        (p_2)
        ...
    ------------
    """

    #ecp_num = 0
    #ecp_max_ang_mom_plus_1 = []
    #ecp_z_core = []
    #ecp_nucleus_index = []
    #ecp_ang_mom = []
    #ecp_coefficient = []
    #ecp_exponent = []
    #ecp_power = []
    #for i in range(nucleus_num):
    #    if has_ecp[i]:
    #        ecps = data["Molecule"]["Atoms"][i]["ECPs"]
    #        ecp_z_core.append(ecps["N_core"])
    #        ecp_max_ang_mom_plus_1.append(dict_ang_mom[ecps["lmax"]])
    #        ecp_num += len(ecps["potential"])
    #        ecp_nucleus_index.append(i)
    #        for k in range(len(ecps["potential"])):
    #            ecpi = ecps["potential"][k]["ecp"]
    #            ecp_ang_mom.extend(dict_ang_mom[ecps["potential"][k]["Shell"]])
    #            ecp_exponent.extend(ecpi[0])
    #            ecp_coefficient.extend(ecpi[1])
    #            ecp_power.extend(ecpi[2])


    #    else:
    #        # special case!! H and He.
    #        # For the sake of clarity, here I put a dummy coefficient (0.0)
    #        # for the ul-s part here.
    #        ecp_num += 1
    #        ecp_nucleus_index.append(i)
    #        ecp_ang_mom.append(0)
    #        ecp_coefficient.append(0.0)
    #        ecp_exponent.append(1.0)
    #        ecp_power.append(0)

    ## write to the trex file
    #trexio.write_ecp_num(trexio_file, ecp_num)
    #trexio.write_ecp_max_ang_mom_plus_1(trexio_file, ecp_max_ang_mom_plus_1)
    #trexio.write_ecp_z_core(trexio_file, ecp_z_core)
    #trexio.write_ecp_nucleus_index(trexio_file, ecp_nucleus_index)
    #trexio.write_ecp_ang_mom(trexio_file, ecp_ang_mom)
    #trexio.write_ecp_coefficient(trexio_file, ecp_coefficient)
    #trexio.write_ecp_exponent(trexio_file, ecp_exponent)
    #trexio.write_ecp_power(trexio_file, ecp_power)


    # close the TREX-IO file
    trexio_file.close()

    logger.info("Conversion to TREXIO is done.")

def cli():
    import argparse
    from logging import getLogger, StreamHandler, Formatter

    log_level = "INFO"
    logger = getLogger("orca-trexio")
    logger.setLevel(log_level)
    stream_handler = StreamHandler()
    stream_handler.setLevel(log_level)
    handler_format = Formatter("%(message)s")
    stream_handler.setFormatter(handler_format)
    logger.addHandler(stream_handler)

    # define the parser
    parser = argparse.ArgumentParser(
        epilog="From orca json file to TREXIO file",
        usage="python orca_to_trexio.py -c \
            orca_json -o trexio_filename",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-c",
        "--orca_json",
        help="orca json",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-o",
        "--trexio_filename",
        help="trexio filename",
        type=str,
        default="trexio.hdf5",
    )
    parser.add_argument(
        "-b",
        "--back_end",
        help="trexio I/O back-end",
        type=str,
        default="hdf5",
    )

    # parse the input values
    args = parser.parse_args()
    # parsed_parameter_dict = vars(args)

    orca_to_trexio(
        orca_json=args.orca_json,
        filename=args.trexio_filename,
        back_end=args.back_end
    )


if __name__ == "__main__":
    cli()
