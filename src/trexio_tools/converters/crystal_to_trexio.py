# CRYSTAL23 output file -> TREXIO hdf5 file
# author: Kosuke Nakano
# maintainer: Kosuke Nakano
# email: "kousuke_1123@icloud.com"

# load python packages
import os
import re
import numpy as np

# Logger
from logging import getLogger

logger = getLogger("crystal-trexio").getChild(__name__)


def extract_matched_line(keyword, lines):
    matched_lines = [
        (index, line) for index, line in enumerate(lines) if re.match(keyword, line)
    ]
    if len(matched_lines) == 0:
        logger.error(f"The keyword line = {keyword} is not found.")
        return -1, None
    elif len(matched_lines) > 1:
        logger.error(f"The keyword line = {keyword} is found more than once.")
        raise ValueError
    else:
        # logger.error(f'The keyword line is found.')
        index = int(matched_lines[0][0])
        line = lines[index]
        return index, line


def crystal_to_trexio(
    crystal_output: str = "crystal.out",
    trexio_filename: str = "trexio.hdf5",
    back_end: str = "hdf5",
    spin: int = 1,  # 2S+1
):
    """CRYSTAL to TREXIO converter."""
    # import trexio
    import trexio

    logger.info(f"crystal_output = {crystal_output}")
    logger.info(f"trexio_filename = {trexio_filename}")
    logger.info("Conversion starts...")

    # crystal instances
    with open(crystal_output, "r") as f:
        lines = f.readlines()

    # restricted or unrestricted
    keyword = r".*CRYSTAL\s*-\s*SCF\s*-\s*TYPE\s*OF\s*CALCULATION.*"
    calc_type_index, calc_type_line = extract_matched_line(keyword, lines)
    calc_type_keyword = calc_type_line.split(":")[-1]
    if re.match(r".*\sRESTRICTED\sCLOSED\sSHELL.*", calc_type_keyword):
        calc_type = "RHF"
    elif re.match(r".*\sRESTRICTED\sOPEN\sSHELL.*", calc_type_keyword):
        calc_type = "ROHF"
    elif re.match(r".*\sUNRESTRICTED\sOPEN\sSHELL.*", calc_type_keyword):
        calc_type = "UHF"
    else:
        raise NotImplementedError

    logger.info(f"calc_type = {calc_type}")

    # PBC != 0 is not implemented yet.
    if calc_type == "RHF" or calc_type == "ROHF":
        spin_restricted = True
    elif calc_type == "UHF":
        spin_restricted = False
    else:
        raise ValueError

    # PBC
    if any([re.match(r"\s*CRYSTAL\s*CALCULATION.*", line) for line in lines]):
        pbc_flag = True
    else:
        pbc_flag = False

    logger.info(f"PBC flag = {pbc_flag}")

    # PBC != 0 is not implemented yet.
    if not pbc_flag:
        logger.error("This converter supports only 3D PBC calculations.")
        raise NotImplementedError

    # twist_average info

    keyword = r".*\s*SHRINK.*\s*FACT.*MONKH.*\s*"
    k_factor_index, k_factor_line = extract_matched_line(keyword, lines)
    k_factor_x, k_factor_y, k_factor_z = (int(s) for s in k_factor_line.split()[2:5])
    logger.info(f"k_factor={k_factor_x}, {k_factor_y}, {k_factor_z}")

    keyword = r".*NUMBER\s*OF\s*K\s*POINTS\s*IN\s*THE\s*IBZ\s*.*"
    irrep_k_index, irrep_k_line = extract_matched_line(keyword, lines)
    irrep_k_num = int(irrep_k_line.split()[-1])
    logger.info(f"irrep_k_num={irrep_k_num}")

    irrep_k_num_remaining = irrep_k_num
    k_complex_flag_list = []
    k_list = []

    keyword = r".*\s*K\s*POINTS\s*COORDINATES\s*.*"
    k_grid_index, _ = extract_matched_line(keyword, lines)

    for line in lines[k_grid_index + 1 :]:
        num_k_in_the_line = line.count("R") + line.count("C")  # R is real, C is complex
        if num_k_in_the_line == 0:
            break
        pattern = r"([R,C])\(\s*(\d+\s*\d*\s*\d*\s*)\)"
        matches = re.findall(pattern, line)

        for match in matches:
            complex_chr = str(match[0])
            logger.info(f"complex_chr={complex_chr}")
            if complex_chr == "R":
                k_complex_flag_list.append(False)
            elif complex_chr == "C":
                k_complex_flag_list.append(True)
            else:
                logger.error(f"complex_chr={complex_chr} is not expected.")
                raise ValueError
            kx, ky, kz = match[1].split()
            k_list.append(
                (int(kx) / k_factor_x, int(ky) / k_factor_y, int(kz) / k_factor_z)
            )
        irrep_k_num_remaining -= num_k_in_the_line

    if num_k_in_the_line != 0:
        logger.error(
            "The number of read k points is inconsistent with the number of irrep. k-num in the output."
        )
        raise ValueError

    if irrep_k_num == 1:
        twist_average = False
        logger.info("Single-k calculation")

    else:
        twist_average = True
        logger.info("Twisted-average calculation")
        logger.info("Separate TREXIO files are generated")
        logger.info(
            "The correspondence between the index and k is written in kp_info.dat"
        )
        with open("kp_info.dat", "w") as f:
            f.write("# k_index, kx, ky, kz\n")
        logger.warning(
            f"WF at each k point is saved in a separate file kXXXX_{trexio_filename}"
        )

    # set complex flag
    if irrep_k_num == 1 and all([k_i == 0.0 for k_i in k_list[0]]):
        logger.info("k = gamma point")
        logger.info("The generated WF will be real.")
        wf_complex_flag = False
    elif not any(k_complex_flag_list):
        logger.info("All WFs are real.")
        logger.info("The generated WFs will be real.")
        wf_complex_flag = False
    else:
        logger.info("At least, one WF is complex. All WFs are treated as complex")
        logger.info("The generated WFs will be complex.")
        wf_complex_flag = True

    # k_weight info
    keyword = r".*\s*WEIGHT\s*OF\s*K\s*POINTS\s*-\s*MONKHORST\s*.*"
    k_weight_index, k_weight_line = extract_matched_line(keyword, lines)

    if k_weight_index == -1:
        logger.error("The weights of k points are not printed.")
        logger.error("The CRYSTAL code does not print the information by default.")
        logger.error("Please insert KWEIGHTS option.")
        raise ValueError

    k_weight_list = []
    for line in lines[k_weight_index + 1 :]:
        if re.match(r"\s*\n", line):
            break
        else:
            k_weight_list += list(map(float, line.split()))

    # check consistency between the number of read k weights and k points.
    if len(k_list) != len(k_weight_list):
        logger.error(
            "The number of k points ({len(k_list)}) is inconsistent with that of k weights ({len(k_weight_list)})"
        )
        raise ValueError

    # generate complete k lists. This is needed because CRYSTAL always employs
    # the time-reversal symmetry, i.e., \Psi(k, alpha) = \Psi(-k, beta).
    # Since CRYSTAL employs a Gamma-Centered k-grid, the number of TRIM should be 8.
    # In other words, the number of BZ sampling is (# read k points - 8) * 2 [inversion] + 8 [TRIM point].
    # Here, the code assumes that a user does put SYMMREMO in a CRYSTAL input (no symmetry is used.)

    # check if the first k is Gamma point
    if not all([k_i == 0.0 for k_i in k_list[0]]):
        logger.error("The first k point is not Gamma. Not expected.")
        raise NotImplementedError

    # convert weights to integers (the reference value is that at gamma)
    k_float_weight_list = [
        float(k_weight / k_weight_list[0]) for k_weight in k_weight_list
    ]
    if not all(
        [float(round(k_weight, 2)).is_integer() for k_weight in k_float_weight_list]
    ):
        logger.error("Some k points are irrational.")
        logger.error(k_float_weight_list)
        raise ValueError
    k_int_weight_list = [int(k_weight) for k_weight in k_float_weight_list]

    """
    # the code expects that the weight is 1 (TRIM) or 2 (not).
    if not all([k_weight == 1 or k_weight == 2 for k_weight in k_int_weight_list]):
        logger.error(
            "A symmetry is imposed in this calculation, which is not supported yet."
        )
        logger.error("Plase use SYMMREMO to remove symmetries.")
        raise NotImplementedError
    """

    # generate k list for generating TREXIO files
    k_in_list = []
    k_group_index_list = []
    k_in_complex_flag_list = []

    for k_group_index, (k_in, k_weight, k_complex_flag) in enumerate(
        zip(k_list, k_int_weight_list, k_complex_flag_list)
    ):
        if not len(k_in) == 3:  # 3d variable
            logger.error("The dimension of k_vec is not 3!")
            raise ValueError
        for _ in range(k_weight):
            k_group_index_list.append(k_group_index)
            k_in_list.append([k_in[0], k_in[1], k_in[2]])
            k_in_complex_flag_list.append(k_complex_flag)

    # each k WF is stored as a separate file!!
    # for an open-boundary calculation, and a single-k one,
    # k_index is a dummy variable
    for k_file_index, (k_group_index, k_vec, k_complex_flag) in enumerate(
        zip(k_group_index_list, k_in_list, k_in_complex_flag_list)
    ):
        logger.info(f"kpt in CRYSTAL ={k_vec}")
        logger.info(f"k_complex_flag={k_complex_flag}")
        logger.info(f"wf_complex_flag={wf_complex_flag}")

        if not k_complex_flag and wf_complex_flag:
            logger.warning(
                "WF at this k point is real, but it is treated as a complex WF."
            )

        # set a filename
        if twist_average:
            filename = os.path.join(
                os.path.dirname(trexio_filename),
                f"k{k_file_index}_" + os.path.basename(trexio_filename),
            )
            logger.info(f"filename={filename}")
            with open("kp_info.dat", "a") as f:
                f.write(f"{k_file_index} {k_vec[0]} {k_vec[1]} {k_vec[2]}\n")
        else:
            filename = trexio_filename

        if os.path.exists(filename):
            logger.warning(
                f"TREXIO file {filename} already exists and will be removed before conversion."
            )
            if back_end.lower() == "hdf5":
                os.remove(filename)
            else:
                raise NotImplementedError(
                    f"Please remove the {filename} directory manually."
                )

        # trexio back end handling
        if back_end.lower() == "hdf5":
            trexio_back_end = trexio.TREXIO_HDF5
        elif back_end.lower() == "text":
            trexio_back_end = trexio.TREXIO_TEXT
        else:
            raise NotImplementedError(f"{back_end} back-end is not supported.")

        # trexio file
        trexio_file = trexio.File(filename, mode="w", back_end=trexio_back_end)

        # PBC info.
        keyword = ".*LATTICE\s*PARAMETERS\s*.*ANGSTROMS\s*AND\s*DEGREES.*\s*-\s*BOHR"
        bohr_info_index, bohr_info_line = extract_matched_line(keyword, lines)
        bohr_in_crystal = float(bohr_info_line.split()[-2])
        angstrom_to_bohr = 1.0 / bohr_in_crystal

        keyword = r".*DIRECT\s*LATTICE\s*VECTORS\s*CARTESIAN\s*COMPONENTS.*"
        lattice_vec_info_starting_index, _ = extract_matched_line(keyword, lines)
        a = (
            np.array(lines[lattice_vec_info_starting_index + 2].split(), dtype=float)
            * angstrom_to_bohr
        )
        b = (
            np.array(lines[lattice_vec_info_starting_index + 3].split(), dtype=float)
            * angstrom_to_bohr
        )
        c = (
            np.array(lines[lattice_vec_info_starting_index + 4].split(), dtype=float)
            * angstrom_to_bohr
        )

        ##########################################
        # PBC info (TREXIO)
        ##########################################
        trexio.write_pbc_periodic(trexio_file, pbc_flag)
        trexio.write_cell_a(trexio_file, a)
        trexio.write_cell_b(trexio_file, b)
        trexio.write_cell_c(trexio_file, c)
        trexio.write_pbc_k_point(trexio_file, k_vec)

        # structure info.
        # if restricted closed shell ok. N_up = N_dn
        # if restricted open shell, users should specify an integar number (i.e. 2S+1) in the input
        # if unrestricted open shell, the same as restricted open shell
        keyword = r".*N.*\s*OF\s*ATOMS\s*PER\s*CELL.*"
        nucleus_num_index, nucleus_num_line = extract_matched_line(keyword, lines)
        nucleus_num = int(nucleus_num_line.split()[5])

        keyword = r".*N.*\s*OF\s*ELECTRONS\s*PER\s*CELL.*"
        electron_tot_num_index, electron_tot_num_line = extract_matched_line(
            keyword, lines
        )
        electron_tot_num = int(electron_tot_num_line.split()[5])

        logger.warning(f"The input spin(2S+1)={spin}.")
        diff_ele = spin - 1
        if (electron_tot_num - diff_ele) % 2 != 0:
            logger.error(
                f"electron_tot_num={electron_tot_num} is incompatible with input spin."
            )
            raise ValueError

        electron_up_num = int((electron_tot_num - diff_ele) / 2 + diff_ele)
        electron_dn_num = int((electron_tot_num - diff_ele) / 2)

        logger.info(f"nucleus_num = {nucleus_num}")
        logger.info(f"electron_up_num = {electron_up_num}")
        logger.info(f"electron_dn_num = {electron_dn_num}")

        # number of shells
        keyword = r".*NUMBER*\s*OF\s*SHELLS.*"
        num_shell_output_index, num_shell_output_line = extract_matched_line(
            keyword, lines
        )
        num_shell_output = int(num_shell_output_line.split()[3])

        # number of AOs
        keyword = r".*NUMBER*\s*OF\s*AO.*"
        num_ao_output_index, num_ao_output_line = extract_matched_line(keyword, lines)
        num_ao_output = int(num_ao_output_line.split()[3])

        atom_charges_list = [0.0 for i in range(nucleus_num)]  # tentative

        keyword = r".*CARTESIAN\s*COORDINATES\s*-\s*PRIMITIVE\s*CELL.*"
        geom_info_index, geom_info_line = extract_matched_line(keyword, lines)

        atomic_number_list = []
        chemical_symbol_list = []
        coords_list = []

        for line in lines[geom_info_index + 4 : geom_info_index + 4 + nucleus_num]:
            atomic_number = int(line.split()[1])
            chemical_symbol = str(line.split()[2])
            chemical_symbol = chemical_symbol[0].upper() + chemical_symbol[1:].lower()
            coords = [
                float(a) * angstrom_to_bohr for a in line.split()[3:6]
            ]  # unit is bohr

            atomic_number_list.append(atomic_number)
            chemical_symbol_list.append(chemical_symbol)
            coords_list.append(coords)  # this is angstrom, it's ok?

        coords_list_np = np.array(coords_list)

        ##########################################
        # Structure info (TREXIO)
        ##########################################
        trexio.write_electron_up_num(trexio_file, electron_up_num)
        trexio.write_electron_dn_num(trexio_file, electron_dn_num)
        trexio.write_nucleus_num(trexio_file, nucleus_num)
        trexio.write_nucleus_charge(trexio_file, atom_charges_list)
        trexio.write_nucleus_label(trexio_file, chemical_symbol_list)
        trexio.write_nucleus_coord(trexio_file, coords_list_np)

        ##########################################
        # ECP
        ##########################################
        # internal format of CRYSTAL
        # the convention of the exponent is the same as in the TREX-IO format!

        # PSEUDO POTENTIAL
        if any([int(atomic_number) > 200 for atomic_number in atomic_number_list]):
            pp_calc = True
        else:
            pp_calc = False

        if pp_calc:
            keyword = r".*PSEUDOPOTENTIAL\s*INFORMATION.*"
            pp_info_index, pp_info_line = extract_matched_line(keyword, lines)

            pp_dict = {}  # key: atomic number

            for line in lines[pp_info_index + 3 :]:
                if re.match(r"\s*\*+", line):
                    break
                elif re.match(r"\n+", line):
                    continue
                elif re.match(r"\s*TYPE\s*.*", line):
                    continue
                elif re.match(r"\s*ATOMIC\sNUMBER\s*", line):
                    line_ = line.replace(",", "")
                    atomic_number = int(line_.split()[2])
                    z_core = atomic_number - int(float(line_.split()[5]))
                    pp_dict[atomic_number] = {
                        "ecp_num": -1,
                        "ecp_max_ang_mom_plus_1": -1,
                        "ecp_z_core": z_core,
                        "ecp_ang_mom": [],
                        "ecp_exponent": [],
                        "ecp_coefficient": [],
                        "ecp_power": [],
                    }
                else:
                    if re.match(r"\s*.*\sTMS\s*", line):
                        if re.match(r"\s*W0\s*TMS\s*", line):
                            ecp_ang_mom = -1
                        else:
                            m = re.match(r"\s*P(\d)\s*TMS\s*", line)
                            ecp_ang_mom = int(m.groups()[0])
                        shift = 2
                    else:
                        ecp_ang_mom = ecp_ang_mom
                        shift = 0

                    num_ecp_at_line = int((len(line.split()) - shift) / 3)

                    for l in range(num_ecp_at_line):
                        ecp_exponent = float(line.split()[shift + 3 * l + 0])
                        ecp_coefficient = float(line.split()[shift + 3 * l + 1])
                        ecp_power = int(line.split()[shift + 3 * l + 2])

                        pp_dict[atomic_number]["ecp_ang_mom"].append(ecp_ang_mom)
                        pp_dict[atomic_number]["ecp_exponent"].append(ecp_exponent)
                        pp_dict[atomic_number]["ecp_coefficient"].append(
                            ecp_coefficient
                        )
                        pp_dict[atomic_number]["ecp_power"].append(ecp_power)

            for atomic_number, item in pp_dict.items():
                ecp_max_ang_mom = np.max(item["ecp_ang_mom"])
                item["ecp_num"] = len(item["ecp_ang_mom"])
                item["ecp_max_ang_mom_plus_1"] = ecp_max_ang_mom + 1
                # replace -1 in pp_dict[atomic_number]["ecp_ang_mom"] with ecp_max_ang_mom + 1
                for kk in range(len(item["ecp_ang_mom"])):
                    if item["ecp_ang_mom"][kk] == -1:
                        item["ecp_ang_mom"][kk] = ecp_max_ang_mom + 1

            ecp_num = 0

            ecp_max_ang_mom_plus_1 = []
            ecp_z_core = []

            ecp_nucleus_index = []
            ecp_ang_mom = []
            ecp_coefficient = []
            ecp_exponent = []
            ecp_power = []

            for nucleus_i, atomic_number_label in enumerate(atomic_number_list):
                atomic_number = int(atomic_number_label) - 200
                ecp_max_ang_mom_plus_1.append(
                    pp_dict[atomic_number]["ecp_max_ang_mom_plus_1"]
                )
                ecp_z_core.append(pp_dict[atomic_number]["ecp_z_core"])

                ecp_num += pp_dict[atomic_number]["ecp_num"]
                for ind_i in range(pp_dict[atomic_number]["ecp_num"]):
                    ecp_nucleus_index.append(nucleus_i)
                    ecp_ang_mom.append(pp_dict[atomic_number]["ecp_ang_mom"][ind_i])
                    ecp_coefficient.append(
                        pp_dict[atomic_number]["ecp_coefficient"][ind_i]
                    )
                    ecp_exponent.append(pp_dict[atomic_number]["ecp_exponent"][ind_i])
                    ecp_power.append(pp_dict[atomic_number]["ecp_power"][ind_i])

            # logger.debug(ecp_nucleus_index)
            # logger.debug('ecp_max_ang_mom_plus_1')
            # logger.debug(ecp_max_ang_mom_plus_1)
            # logger.debug('ecp_ang_mom')
            # logger.debug(ecp_ang_mom)
            # logger.debug('ecp_coefficient')
            # logger.debug(ecp_coefficient)
            # logger.debug('ecp_exponent')
            # logger.debug(ecp_exponent)
            # logger.debug('ecp_power')
            # logger.debug(ecp_power)

            ##########################################
            # PSEUDO POTENTIAL (TREXIO)
            ##########################################
            # write to the trex file
            trexio.write_ecp_num(trexio_file, ecp_num)
            trexio.write_ecp_max_ang_mom_plus_1(trexio_file, ecp_max_ang_mom_plus_1)
            trexio.write_ecp_z_core(trexio_file, ecp_z_core)
            trexio.write_ecp_nucleus_index(trexio_file, ecp_nucleus_index)
            trexio.write_ecp_ang_mom(trexio_file, ecp_ang_mom)
            trexio.write_ecp_coefficient(trexio_file, ecp_coefficient)
            trexio.write_ecp_exponent(trexio_file, ecp_exponent)
            trexio.write_ecp_power(trexio_file, ecp_power)

        ##########################################
        # basis set info
        ##########################################
        # See page 27 of the manual for the orders of the spherical atomic basis in CRYSTAL
        # for s -> s
        # for p -> px, py, pz
        # for d,f,g -> 0, +-1, +-2, ...
        # Spherical harmonic is employed in the CRYSTAL code.

        basis_type = "Gaussian"

        keyword = r".*LOCAL\s*ATOMIC\s*FUNCTIONS\s*BASIS\s*SET.*"
        basis_info_index, basis_info_line = extract_matched_line(keyword, lines)

        atom_line_keyword = (
            r"\s+\d+\s+[a-z,A-Z]+\s+[+-]*\d+\.\d+\s+[+-]*\d+\.\d+\s+[+-]*\d+\.\d+.*"
        )

        basis_shell_num = 0
        basis_prim_num = 0

        nucleus_index = []
        shell_ang_mom = []
        basis_shell_index = []
        basis_exponent = []
        basis_coefficient = []

        basis_info_read = True

        for line in lines[basis_info_index + 4 :]:
            if re.match(".*INFORMATION.*", line):
                if not basis_info_read:
                    p_nucleus_index = [
                        i for i, x in enumerate(nucleus_index) if x == nucleus_i - 1
                    ]
                    for i_shell in p_nucleus_index:
                        nucleus_index.append(nucleus_i)
                        shell_ang_mom.append(shell_ang_mom[i_shell])
                        basis_shell_num += 1

                        p_prim_index = [
                            i for i, x in enumerate(basis_shell_index) if x == i_shell
                        ]

                        for i_prim in p_prim_index:
                            basis_prim_num += 1
                            basis_shell_index.append(basis_shell_num - 1)
                            basis_exponent.append(basis_exponent[i_prim])
                            basis_coefficient.append(basis_coefficient[i_prim])
                break
            if re.match(atom_line_keyword, line):
                if basis_info_read:
                    nucleus_i = int(line.split()[0]) - 1
                    basis_info_read = False
                else:
                    nucleus_i = int(line.split()[0]) - 1
                    p_nucleus_index = [
                        i for i, x in enumerate(nucleus_index) if x == nucleus_i - 2
                    ]
                    for i_shell in p_nucleus_index:
                        nucleus_index.append(nucleus_i - 1)
                        shell_ang_mom.append(shell_ang_mom[i_shell])
                        basis_shell_num += 1

                        p_prim_index = [
                            i for i, x in enumerate(basis_shell_index) if x == i_shell
                        ]

                        for i_prim in p_prim_index:
                            basis_prim_num += 1
                            basis_shell_index.append(basis_shell_num - 1)
                            basis_exponent.append(basis_exponent[i_prim])
                            basis_coefficient.append(basis_coefficient[i_prim])

                    basis_info_read = False
            else:
                basis_info_read = True
                m = re.match(r"\s+.*([S,P,D,F,G])\s+.*", line)
                if m:
                    orb_type = str(m.groups()[0])
                    nucleus_index.append(nucleus_i)
                    basis_shell_num += 1

                    if orb_type == "S":
                        shell_ang_mom.append(0)
                    elif orb_type == "P":
                        shell_ang_mom.append(1)
                    elif orb_type == "D":
                        shell_ang_mom.append(2)
                    elif orb_type == "F":
                        shell_ang_mom.append(3)
                    elif orb_type == "G":
                        shell_ang_mom.append(4)

                else:
                    basis_prim_num += 1
                    basis_shell_index.append(basis_shell_num - 1)

                    if orb_type == "S":
                        coeff_index = 1
                    elif orb_type == "P":
                        coeff_index = 2
                    elif orb_type == "D":
                        coeff_index = 3
                    elif orb_type == "F":
                        coeff_index = 3
                    elif orb_type == "G":
                        coeff_index = 3

                    mod_line = re.sub(r"(\d)(?=-)", r"\1 ", line)
                    basis_exponent.append(float(mod_line.split()[0]))
                    basis_coefficient.append(float(mod_line.split()[coeff_index]))

        # logger.debug(nucleus_index)
        # logger.debug(basis_shell_index)
        # logger.debug(basis_coefficient)
        # logger.debug(basis_exponent)

        # check if num_shell_output == basis_shell_num?
        if num_shell_output != basis_shell_num:
            logger.error(
                f"num of shell in the output is {num_shell_output}, while read one is {basis_shell_num}"
            )
            raise ValueError

        # normalization factors
        basis_shell_factor = [1.0 for _ in range(basis_shell_num)]  # 1.0 in CRYSTAL
        basis_prim_factor = [1.0 for _ in range(basis_shell_num)]  # tentative
        # Note!! Here, the normalization factors are not computed.

        ##########################################
        # ao info
        ##########################################
        ao_cartesian = 0  # spherical basis representation in CRYSTAL
        ao_shell = []
        for i, ang_mom in enumerate(shell_ang_mom):
            for _ in range(2 * ang_mom + 1):
                ao_shell.append(i)
        ao_num = len(ao_shell)

        # 1.0 in CRYSTAL (because spherical)
        ao_normalization = [1.0 for _ in range(ao_num)]

        logger.info(f"ao_num={ao_num}")

        # check if num_ao_output == basis_ao_num?
        if num_ao_output != ao_num:
            logger.error(
                f"num of AO in the output is {num_ao_output}, while the read one is {ao_num}"
            )
            raise ValueError

        ##########################################
        # mo info
        ##########################################
        mo_type = "MO"

        # start reading MOs
        mo_read_flag = False

        keyword = r".*FINAL\s*EIGENVECTORS.*"
        mo_info_index, mo_info_line = extract_matched_line(keyword, lines)
        if mo_info_index == -1:
            logger.error("The eigenvalues (i.e., KS orbital info.) are not printed.")
            logger.error("The CRYSTAL code does not print the information by default.")
            logger.warning("Please use the SETPRINT option with 67 -5YY,")
            logger.warning("where YY is the number of k points.")
            logger.warning("Please refer to the CRYSTAL manual for the detail.")
            raise ValueError
        mo_coeff_read = [[], []]  # alpha, beta

        ao_index_b_list = []
        mo_coeff_b_list = []
        mo_num_to_be_read = 0

        if spin_restricted:
            read_spin = 0  # alpha
        else:
            read_spin = 0  # alpha as default, but change according to headers

        for ll, line in enumerate(lines[mo_info_index:]):
            # logger.debug(line)
            if re.match(r"\s*\n", line):
                continue
            if re.match(r"\s*ALPHA\s*ELECTRONS\s*", line):  # header
                read_spin = 0  # alpha
                continue
            if re.match(r"\s*BETA\s*ELECTRONS\s*", line):  # header
                read_spin = 1  # beta
                continue
            if re.match(r".*EIGENVECTORS\s*IN\s*FORTRAN\s*UNIT.*", line):
                break

            m = re.match(r"\s*(\d+)\s\((\s*\d+\s*\d+\s*\d+)\).*", line)
            if m:
                k_index_read = int(m.groups()[0]) - 1
                if k_group_index == k_index_read:
                    # logger.debug(f"k_index_read={k_index_read}")
                    # logger.debug(f"k_group_index={k_group_index}")
                    mo_read_flag = True
                else:
                    mo_read_flag = False
            elif mo_read_flag:
                if re.match(r"\s*(\d+\s+)*\d+$", line):
                    mo_num_to_be_read = int(line.split()[-1]) - int(line.split()[0]) + 1
                    # logger.debug(f"line={line}")
                    # logger.debug(f"mo_num_to_be_read = {mo_num_to_be_read}")
                else:
                    ao_index = int(line.split()[0]) - 1
                    mo_coeff_b = list(map(float, line.split()[1:]))

                    ao_index_b_list.append(ao_index)
                    mo_coeff_b_list.append(mo_coeff_b)

                    if ao_index == ao_num - 1:
                        # logger.debug('storing and resetting mo_coeff_b')
                        for i_read_mo in range(mo_num_to_be_read):
                            if (
                                not k_complex_flag and not wf_complex_flag
                            ):  # real wf -> real wf
                                mo_coeff_read[read_spin].append(
                                    [
                                        mo_coeff_b[i_read_mo]
                                        for mo_coeff_b in mo_coeff_b_list
                                    ]
                                )
                            elif (
                                not k_complex_flag and wf_complex_flag
                            ):  # real wf -> complex wf
                                mo_coeff_read[read_spin].append(
                                    [
                                        complex(mo_coeff_b[i_read_mo], 0)
                                        for mo_coeff_b in mo_coeff_b_list
                                    ]
                                )
                            else:  # complex wf -> complex wf
                                mo_coeff_read[read_spin].append(
                                    [
                                        complex(
                                            mo_coeff_b[2 * i_read_mo],
                                            mo_coeff_b[2 * i_read_mo + 1],
                                        )
                                        for mo_coeff_b in mo_coeff_b_list
                                    ]
                                )

                        ao_index_b_list = []
                        mo_coeff_b_list = []

        # sanity check
        if spin_restricted:
            spin_check_list = {0}
        else:
            spin_check_list = {0, 1}
        for ns in spin_check_list:
            if not all([len(mo_coeff) == ao_num for mo_coeff in (mo_coeff_read[ns])]):
                logger.error(
                    "the number of read AO coeff. is inconsistent with the read ao_num."
                )
                raise ValueError

        # set occupations
        if spin_restricted:
            if spin == 1:
                mo_occupation_read_up = [2.0 for _ in range(electron_up_num)] + [
                    0.0 for _ in range(len(mo_coeff_read[0]) - electron_up_num)
                ]
                mo_occupation_read_dn = []
            else:
                mo_occupation_read_up = (
                    [2.0 for _ in range(electron_dn_num)]
                    + [1.0 for _ in range(electron_up_num - electron_dn_num)]
                    + [0.0 for _ in range(len(mo_coeff_read[0]) - electron_up_num)]
                )
                mo_occupation_read_dn = []
        else:
            mo_occupation_read_up = [1.0 for _ in range(electron_up_num)] + [
                0.0 for _ in range(len(mo_coeff_read[0]) - electron_up_num)
            ]
            mo_occupation_read_dn = [1.0 for _ in range(electron_dn_num)] + [
                0.0 for _ in range(len(mo_coeff_read[1]) - electron_dn_num)
            ]

        mo_occupation_read = [mo_occupation_read_up, mo_occupation_read_dn]

        # start reading energy (tentative)
        mo_energy_read_up = [0.0 for _ in range(len(mo_coeff_read[0]))]
        mo_energy_read_dn = [0.0 for _ in range(len(mo_coeff_read[0]))]
        mo_energy_read = [mo_energy_read_up, mo_energy_read_dn]

        # the followings are finally given to the TREXIO file
        mo_coefficient_all = []
        mo_spin_all = []
        mo_occupation_all = []
        mo_energy_all = []

        # mo read part starts both for alpha and beta spins
        for ns, spin in enumerate([0, 1]):
            if spin_restricted:
                if spin == 1:  # 0 is alpha(up), 1 is beta(dn)
                    logger.info("This is spin-restricted calculation.")
                    logger.info("Skip the MO conversion step for beta MOs.")
                    continue
            else:
                logger.info(
                    f"MO conversion step for {spin}-spin MOs. [0 is alpha(up), 1 is beta(dn)]."
                )

            mo_occupation = mo_occupation_read[ns]
            mo_energy = mo_energy_read[ns]
            mo_coeff = mo_coeff_read[ns]
            mo_num = len(mo_coeff_read[ns])

            # saved mo_occ and mo_energy
            mo_occupation_all += list(mo_occupation)
            mo_energy_all += list(mo_energy)
            mo_spin_all += [spin for _ in range(mo_num)]

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
                        # print("(CRYSTAL notation): s(l=0)")
                        # print("(TREX-IO notation): s(l=0)")
                        reorder_index = [0]

                    elif current_ang_mom == 1:  # p shell
                        # print("p shell/permutation is needed.")
                        # print("(CRYSTAL notation): px(l=+1), py(l=-1), pz(l=0)")
                        # print("(TREX-IO notation): pz(l=0), px(l=+1), py(l=-1)")
                        reorder_index = [2, 0, 1]

                    elif current_ang_mom >= 2:  # > d shell
                        # print("> d shell/no permutation is needed.")
                        # print(
                        #    "(CRYSTAL) f3,0(l=0),
                        #               f3,+1(l=+1), f3,-1(l=-1),
                        #               f3,+2(l=+2), f3,-2(l=-2),
                        #               f3,+3(l=+3), f3,-3(l=-3)
                        # )
                        # print(
                        #    "(TREX-IO) f3,0(l=0),
                        #               f3,+1(l=+1), f3,-1(l=-1),
                        #               f3,+2(l=+2), f3,-2(l=-2),
                        #               f3,+3(l=+3), f3,-3(l=-3)
                        # )
                        reorder_index = [i for i in range(multiplicity)]

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

                mo_coefficient.append(mo_coeff_buffer)
                # permutation_matrix.append(perm_list)

            mo_coefficient_all += mo_coefficient

        # MOs read part end both for alpha and beta spins[l]
        # logger.debug("len(mo_coefficient_all)")
        # logger.debug(len(mo_coefficient_all))
        # logger.debug("len(mo_occupation_all)")
        # logger.debug(len(mo_occupation_all))
        # logger.debug("len(mo_spin_all)")
        # logger.debug(len(mo_spin_all))

        if spin_restricted:
            logger.info(f"number of MOs = {mo_spin_all.count(0)}")
        else:
            logger.info(f"number of MOs for up spin = {mo_spin_all.count(0)}")
            logger.info(f"number of MOs for dn spin = {mo_spin_all.count(1)}")

        if wf_complex_flag:
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

        logger.info("--MOs Done--")

        ##########################################
        # basis set info
        ##########################################
        trexio.write_basis_type(trexio_file, basis_type)  #
        trexio.write_basis_shell_num(trexio_file, basis_shell_num)  #
        trexio.write_basis_prim_num(trexio_file, basis_prim_num)  #
        trexio.write_basis_nucleus_index(trexio_file, nucleus_index)  #
        trexio.write_basis_shell_ang_mom(trexio_file, shell_ang_mom)  #
        trexio.write_basis_shell_factor(trexio_file, basis_shell_factor)  #
        trexio.write_basis_shell_index(trexio_file, basis_shell_index)  #
        trexio.write_basis_exponent(trexio_file, basis_exponent)  #
        trexio.write_basis_coefficient(trexio_file, basis_coefficient)  #
        trexio.write_basis_prim_factor(trexio_file, basis_prim_factor)  #

        ##########################################
        # ao info
        ##########################################
        trexio.write_ao_cartesian(trexio_file, ao_cartesian)  #
        trexio.write_ao_num(trexio_file, ao_num)  #
        trexio.write_ao_shell(trexio_file, ao_shell)  #
        trexio.write_ao_normalization(trexio_file, ao_normalization)  #

        ##########################################
        # mo info
        ##########################################
        trexio.write_mo_type(trexio_file, mo_type)  #

        if wf_complex_flag:
            trexio.write_mo_num(trexio_file, len(mo_coefficient_real))  #
            trexio.write_mo_coefficient(trexio_file, mo_coefficient_real)  #
            trexio.write_mo_coefficient_im(trexio_file, mo_coefficient_imag)  #
        else:
            trexio.write_mo_num(trexio_file, len(mo_coefficient_real))  #
            trexio.write_mo_coefficient(trexio_file, mo_coefficient_real)  #

        trexio.write_mo_occupation(trexio_file, mo_occupation_all)  #

        trexio.write_mo_spin(trexio_file, mo_spin_all)  #

        # close the TREX-IO file
        trexio_file.close()

    logger.info("Conversion to TREXIO is done.")


def cli():
    import argparse
    from logging import getLogger, StreamHandler, Formatter

    log_level = "INFO"
    logger = getLogger("crystal-trexio")
    logger.setLevel(log_level)
    stream_handler = StreamHandler()
    stream_handler.setLevel(log_level)
    handler_format = Formatter("%(message)s")
    stream_handler.setFormatter(handler_format)
    logger.addHandler(stream_handler)

    # define the parser
    parser = argparse.ArgumentParser(
        epilog="From CRYSTAL output file to TREXIO file",
        usage="python crystal_to_trexio.py -c \
            crystal_output -o trexio_filename",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-c",
        "--crystal_output",
        help="crystal output",
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
    parser.add_argument(
        "-s",
        "--spin",
        help="spin(2S+1)",
        type=int,
        default=1,
    )

    # parse the input values
    args = parser.parse_args()
    # parsed_parameter_dict = vars(args)

    crystal_to_trexio(
        crystal_output=args.crystal_output,
        trexio_filename=args.trexio_filename,
        back_end=args.back_end,
        spin=args.spin,
    )


if __name__ == "__main__":
    cli()
