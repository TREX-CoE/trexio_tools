import math
import os
import re

import numpy as np
import trexio

try:
    import h5py
except ImportError:
    h5py = None

def _trexio_backend(back_end):
    key = str(back_end).strip().lower()
    if key == "hdf5":
        return trexio.TREXIO_HDF5
    if key == "text":
        return trexio.TREXIO_TEXT

    raise NotImplementedError(f"{back_end} back-end is not supported.")


def _factorial2(n):
    if n <= 0:
        return 1.0

    value = 1.0
    while n > 0:
        value *= n
        n -= 2
    return value


def _primitive_norm(alpha, ang_mom):
    return ((2.0 * alpha / math.pi) ** 0.75) * ((4.0 * alpha) ** (0.5 * ang_mom)) / math.sqrt(_factorial2(2 * ang_mom - 1))


def _shell_symbol(ang_mom):
    symbols = "spdfghijkl"
    if ang_mom >= len(symbols):
        raise NotImplementedError(f"Angular momentum {ang_mom} is not supported.")

    return symbols[ang_mom]


def _trexio_m_order(magnetic_quantum_number):
    if magnetic_quantum_number == 0:
        return 0
    if magnetic_quantum_number > 0:
        return 2 * magnetic_quantum_number - 1
    return 2 * abs(magnetic_quantum_number)


def _parse_ao_label(label):
    match = re.match(r"^\s*(\d+)\s+\S+\s+(\d+)([spdfghijkl])([+-]?\d*)\s*$", label)
    if match is None:
        raise NotImplementedError(f"Unable to parse VeloxChem AO label: {label!r}")

    atom_index = int(match.group(1)) - 1
    shell_key = f"{int(match.group(2))}{match.group(3)}"
    magnetic_quantum_number = match.group(4)
    magnetic_quantum_number = 0 if magnetic_quantum_number in ("", "+", "-") else int(magnetic_quantum_number)
    return atom_index, shell_key, magnetic_quantum_number


def _checkpoint_candidates(vlx_h5):
    root, extension = os.path.splitext(vlx_h5)
    candidates = [vlx_h5]

    if vlx_h5.endswith("_scf.h5"):
        candidates.append(vlx_h5[:-7])
    else:
        candidates.append(f"{root}_scf{extension or '.h5'}")
        candidates.append(f"{vlx_h5}_scf.h5")

    seen = []
    for candidate in candidates:
        if candidate not in seen:
            seen.append(candidate)
    return seen


def _results_candidates(vlx_h5):
    candidates = [vlx_h5]

    if vlx_h5.endswith("_scf.h5"):
        candidates.append(vlx_h5.replace("_scf.h5", ".h5"))
    else:
        candidates.append(f"{vlx_h5}.h5")

    seen = []
    for candidate in candidates:
        if candidate not in seen:
            seen.append(candidate)
    return seen


def _first_success(readers):
    last_error = None
    for reader in readers:
        try:
            return reader()
        except Exception as exc:
            last_error = exc

    if last_error is not None:
        raise last_error

    raise RuntimeError("No readers were provided.")


def _read_plain_scf_group(path):
    if h5py is None:
        raise ImportError("h5py is required to read plain VeloxChem HDF5 files.")

    with h5py.File(path, "r") as handle:
        if "scf" not in handle:
            raise KeyError("No scf group found in VeloxChem HDF5 file.")

        group = handle["scf"]
        raw = {}
        for key in group.keys():
            value = group[key][()]
            if isinstance(value, np.ndarray) and value.shape == (1,):
                value = value[0]
            if isinstance(value, (bytes, np.bytes_)):
                value = value.decode("utf-8")
            raw[key] = value
        return raw


def _load_veloxchem_objects(vlx_h5):
    from veloxchem import MolecularOrbitals
    from veloxchem.molecularorbitals import molorb
    from veloxchem.resultsio import read_molecule_and_basis, read_results

    checkpoint_candidates = _checkpoint_candidates(vlx_h5)

    molecule, basis = _first_success([
        lambda path=path: read_molecule_and_basis(path)
        for path in checkpoint_candidates
        if os.path.exists(path)
    ])

    scf_results = {}
    for candidate in _results_candidates(vlx_h5):
        if not os.path.exists(candidate):
            continue
        try:
            scf_results = read_results(candidate, "scf")
            break
        except Exception:
            try:
                scf_results = _read_plain_scf_group(candidate)
                break
            except Exception:
                continue

    mo_readers = [
        lambda path=path: MolecularOrbitals.read_hdf5(path)
        for path in checkpoint_candidates
        if os.path.exists(path)
    ]

    if scf_results:
        def _read_mos_from_scf_results(results=scf_results):
            if "C_alpha" not in results or "E_alpha" not in results or "occ_alpha" not in results:
                raise KeyError("SCF results do not contain alpha orbital data.")

            orbitals = [np.asarray(results["C_alpha"], dtype=np.float64)]
            energies = [np.asarray(results["E_alpha"], dtype=np.float64)]
            occupations = [np.asarray(results["occ_alpha"], dtype=np.float64)]

            scf_type = str(results.get("scf_type", "restricted")).lower()
            if scf_type == "unrestricted":
                orbitals.append(np.asarray(results["C_beta"], dtype=np.float64))
                energies.append(np.asarray(results["E_beta"], dtype=np.float64))
                occupations.append(np.asarray(results["occ_beta"], dtype=np.float64))
                orbital_type = molorb.unrest
            elif scf_type == "restricted_openshell":
                occupations.append(np.asarray(results["occ_beta"], dtype=np.float64))
                orbital_type = molorb.restopen
            else:
                orbital_type = molorb.rest

            return MolecularOrbitals(orbitals, energies, occupations, orbital_type)

        mo_readers.append(_read_mos_from_scf_results)

    mos = _first_success(mo_readers)

    return molecule, basis, mos, scf_results


def _build_basis_data(molecule, basis):
    shell_to_atom = list(basis.atomic_indices())
    basis_functions = basis.basis_functions()

    if len(shell_to_atom) != len(basis_functions):
        raise ValueError("Inconsistent VeloxChem basis: shell count does not match shell mapping.")

    shell_ang_mom = []
    shell_index = []
    exponent = []
    coefficient = []
    prim_factor = []
    shell_factor = []
    shell_keys = {}
    shell_counts = {}

    for shell_id, (atom_index, shell) in enumerate(zip(shell_to_atom, basis_functions)):
        ang_mom = shell.get_angular_momentum()
        shell_ang_mom.append(ang_mom)
        shell_factor.append(1.0)

        per_atom_key = (atom_index, ang_mom)
        shell_counts[per_atom_key] = shell_counts.get(per_atom_key, 0) + 1
        shell_keys[(atom_index, f"{shell_counts[per_atom_key]}{_shell_symbol(ang_mom)}")] = shell_id

        exponents = list(shell.get_exponents())
        contracted_norms = list(shell.get_normalization_factors())

        if len(exponents) != len(contracted_norms):
            raise ValueError("Inconsistent VeloxChem basis: primitive data lengths differ.")

        for alpha, contracted_norm in zip(exponents, contracted_norms):
            primitive_norm = _primitive_norm(alpha, ang_mom)
            shell_index.append(shell_id)
            exponent.append(alpha)
            prim_factor.append(primitive_norm)
            coefficient.append(contracted_norm / primitive_norm)

    ao_map = basis.get_ao_basis_map(molecule)
    shell_entries = {shell_id: [] for shell_id in range(len(basis_functions))}
    for ao_index, label in enumerate(ao_map):
        atom_index, shell_key, magnetic_quantum_number = _parse_ao_label(label)
        shell_id = shell_keys[(atom_index, shell_key)]
        shell_entries[shell_id].append((ao_index, magnetic_quantum_number))

    ao_permutation = []
    ao_shell = []
    for shell_id in range(len(basis_functions)):
        entries = sorted(shell_entries[shell_id], key=lambda item: _trexio_m_order(item[1]))
        ao_permutation.extend(index for index, _ in entries)
        ao_shell.extend(shell_id for _ in entries)

    return {
        "shell_num": len(basis_functions),
        "prim_num": len(exponent),
        "nucleus_index": np.asarray(shell_to_atom, dtype=np.int64),
        "shell_ang_mom": np.asarray(shell_ang_mom, dtype=np.int32),
        "shell_factor": np.asarray(shell_factor, dtype=np.float64),
        "r_power": np.zeros(len(basis_functions), dtype=np.int32),
        "shell_index": np.asarray(shell_index, dtype=np.int64),
        "exponent": np.asarray(exponent, dtype=np.float64),
        "coefficient": np.asarray(coefficient, dtype=np.float64),
        "prim_factor": np.asarray(prim_factor, dtype=np.float64),
        "ao_num": len(ao_permutation),
        "ao_shell": np.asarray(ao_shell, dtype=np.int64),
        "ao_normalization": np.ones(len(ao_permutation), dtype=np.float64),
        "ao_permutation": np.asarray(ao_permutation, dtype=np.int64),
    }


def _build_mo_data(mos, ao_permutation):
    orbitals_type = str(mos.get_orbitals_type())
    alpha_orbitals = mos.alpha_to_numpy()[ao_permutation, :].T
    alpha_energies = mos.ea_to_numpy()
    alpha_occ = mos.occa_to_numpy()

    if orbitals_type.endswith("unrest"):
        beta_orbitals = mos.beta_to_numpy()[ao_permutation, :].T
        beta_energies = mos.eb_to_numpy()
        beta_occ = mos.occb_to_numpy()
        return {
            "coefficient": np.vstack((alpha_orbitals, beta_orbitals)),
            "energy": np.concatenate((alpha_energies, beta_energies)),
            "occupation": np.concatenate((alpha_occ, beta_occ)),
            "spin": np.concatenate((np.zeros(alpha_orbitals.shape[0], dtype=np.int32), np.ones(beta_orbitals.shape[0], dtype=np.int32))),
        }

    if orbitals_type.endswith("restopen"):
        return {
            "coefficient": alpha_orbitals,
            "energy": alpha_energies,
            "occupation": alpha_occ + mos.occb_to_numpy(),
            "spin": None,
        }

    return {
        "coefficient": alpha_orbitals,
        "energy": alpha_energies,
        "occupation": alpha_occ,
        "spin": None,
    }


def vlx_to_trexio(vlx_h5="vlx.h5", filename="trexio.hdf5", back_end="hdf5"):
    try:
        molecule, basis, mos, scf_results = _load_veloxchem_objects(vlx_h5)
    except ImportError as exc:
        raise ImportError("VeloxChem is required for the VLX -> TREXIO converter.") from exc

    if basis.has_ecp():
        raise NotImplementedError("VLX -> TREXIO conversion does not yet support ECP basis sets.")

    basis_data = _build_basis_data(molecule, basis)
    mo_data = _build_mo_data(mos, basis_data["ao_permutation"])

    with trexio.File(filename, mode="w", back_end=_trexio_backend(back_end)) as trexio_file:
        trexio.write_metadata_code_num(trexio_file, 1)
        trexio.write_metadata_code(trexio_file, ["VeloxChem"])

        trexio.write_nucleus_num(trexio_file, molecule.number_of_atoms())
        trexio.write_nucleus_charge(trexio_file, molecule.get_element_ids())
        trexio.write_nucleus_coord(trexio_file, molecule.get_coordinates_in_bohr())
        trexio.write_nucleus_label(trexio_file, molecule.get_labels())
        try:
            trexio.write_nucleus_repulsion(trexio_file, molecule.nuclear_repulsion_energy())
        except Exception:
            pass

        trexio.write_electron_num(trexio_file, molecule.number_of_electrons())
        trexio.write_electron_up_num(trexio_file, molecule.number_of_alpha_electrons())
        trexio.write_electron_dn_num(trexio_file, molecule.number_of_beta_electrons())

        trexio.write_basis_type(trexio_file, "Gaussian")
        trexio.write_basis_shell_num(trexio_file, int(basis_data["shell_num"]))
        trexio.write_basis_prim_num(trexio_file, int(basis_data["prim_num"]))
        trexio.write_basis_nucleus_index(trexio_file, basis_data["nucleus_index"])
        trexio.write_basis_shell_ang_mom(trexio_file, basis_data["shell_ang_mom"])
        trexio.write_basis_shell_factor(trexio_file, basis_data["shell_factor"])
        trexio.write_basis_r_power(trexio_file, basis_data["r_power"])
        trexio.write_basis_shell_index(trexio_file, basis_data["shell_index"])
        trexio.write_basis_exponent(trexio_file, basis_data["exponent"])
        trexio.write_basis_coefficient(trexio_file, basis_data["coefficient"])
        trexio.write_basis_prim_factor(trexio_file, basis_data["prim_factor"])

        trexio.write_ao_cartesian(trexio_file, 0)
        trexio.write_ao_num(trexio_file, int(basis_data["ao_num"]))
        trexio.write_ao_shell(trexio_file, basis_data["ao_shell"])
        trexio.write_ao_normalization(trexio_file, basis_data["ao_normalization"])

        trexio.write_mo_num(trexio_file, int(mo_data["coefficient"].shape[0]))
        trexio.write_mo_type(trexio_file, "SCF")
        trexio.write_mo_coefficient(trexio_file, np.asarray(mo_data["coefficient"], dtype=np.float64))
        trexio.write_mo_energy(trexio_file, np.asarray(mo_data["energy"], dtype=np.float64))
        trexio.write_mo_occupation(trexio_file, np.asarray(mo_data["occupation"], dtype=np.float64))
        if mo_data["spin"] is not None:
            trexio.write_mo_spin(trexio_file, mo_data["spin"])

        overlap = scf_results.get("S")
        if overlap is not None:
            overlap = np.asarray(overlap, dtype=np.float64)
            order = basis_data["ao_permutation"]
            trexio.write_ao_1e_int_overlap(trexio_file, overlap[np.ix_(order, order)])

        scf_energy = scf_results.get("scf_energy")
        if scf_energy is not None:
            trexio.write_state_num(trexio_file, 1)
            trexio.write_state_id(trexio_file, 0)
            trexio.write_state_energy(trexio_file, float(scf_energy))
            trexio.write_state_current_label(trexio_file, "State 0")
            trexio.write_state_label(trexio_file, ["State 0"])
            trexio.write_state_file_name(trexio_file, [filename])