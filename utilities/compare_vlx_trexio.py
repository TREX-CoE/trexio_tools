#!/usr/bin/env python3

import argparse
import sys

import numpy as np
import trexio

from trexio_tools.converters.vlx_to_trexio import _build_basis_data
from trexio_tools.converters.vlx_to_trexio import _build_mo_data
from trexio_tools.converters.vlx_to_trexio import _load_veloxchem_objects


def _max_abs_diff(left, right):
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    if left_array.shape != right_array.shape:
        raise ValueError(f"shape mismatch: {left_array.shape} != {right_array.shape}")
    if left_array.size == 0:
        return 0.0
    return float(np.max(np.abs(left_array - right_array)))


def _compare_scalar(name, reference, actual, tolerance, failures):
    if isinstance(reference, str) or isinstance(actual, str):
        ok = reference == actual
        diff_text = "match" if ok else f"ref={reference!r} actual={actual!r}"
    else:
        diff = abs(float(reference) - float(actual))
        ok = diff <= tolerance
        diff_text = f"abs_diff={diff:.3e}"

    status = "OK" if ok else "FAIL"
    print(f"[{status}] {name}: {diff_text}")
    if not ok:
        failures.append(name)


def _compare_array(name, reference, actual, tolerance, failures):
    try:
        diff = _max_abs_diff(reference, actual)
        ok = diff <= tolerance
        detail = f"shape={np.asarray(reference).shape} max_abs_diff={diff:.3e}"
    except ValueError as exc:
        ok = False
        detail = str(exc)

    status = "OK" if ok else "FAIL"
    print(f"[{status}] {name}: {detail}")
    if not ok:
        failures.append(name)


def _read_trexio_data(path):
    with trexio.File(path, mode="r", back_end=trexio.TREXIO_AUTO) as handle:
        data = {
            "nucleus_num": trexio.read_nucleus_num(handle),
            "nucleus_charge": trexio.read_nucleus_charge(handle),
            "nucleus_coord": trexio.read_nucleus_coord(handle),
            "electron_num": trexio.read_electron_num(handle),
            "electron_up_num": trexio.read_electron_up_num(handle),
            "electron_dn_num": trexio.read_electron_dn_num(handle),
            "basis_shell_num": trexio.read_basis_shell_num(handle),
            "basis_prim_num": trexio.read_basis_prim_num(handle),
            "basis_nucleus_index": trexio.read_basis_nucleus_index(handle),
            "basis_shell_ang_mom": trexio.read_basis_shell_ang_mom(handle),
            "basis_shell_factor": trexio.read_basis_shell_factor(handle),
            "basis_r_power": trexio.read_basis_r_power(handle),
            "basis_shell_index": trexio.read_basis_shell_index(handle),
            "basis_exponent": trexio.read_basis_exponent(handle),
            "basis_coefficient": trexio.read_basis_coefficient(handle),
            "basis_prim_factor": trexio.read_basis_prim_factor(handle),
            "ao_num": trexio.read_ao_num(handle),
            "ao_shell": trexio.read_ao_shell(handle),
            "ao_normalization": trexio.read_ao_normalization(handle),
            "mo_num": trexio.read_mo_num(handle),
            "mo_coefficient": trexio.read_mo_coefficient(handle),
            "mo_energy": trexio.read_mo_energy(handle),
            "mo_occupation": trexio.read_mo_occupation(handle),
        }

        if trexio.has_mo_spin(handle):
            data["mo_spin"] = trexio.read_mo_spin(handle)
        else:
            data["mo_spin"] = None

        if trexio.has_ao_1e_int_overlap(handle):
            data["ao_overlap"] = trexio.read_ao_1e_int_overlap(handle)
        else:
            data["ao_overlap"] = None

        if trexio.has_state_energy(handle):
            data["state_energy"] = trexio.read_state_energy(handle)
        else:
            data["state_energy"] = None

        return data


def main():
    parser = argparse.ArgumentParser(description="Compare a VeloxChem HDF5 file to a converted TREXIO file.")
    parser.add_argument("vlx_h5", help="Input VeloxChem HDF5 file")
    parser.add_argument("trexio_file", help="Converted TREXIO file")
    parser.add_argument("--tolerance", type=float, default=1.0e-10, help="Absolute tolerance for floating-point comparisons")
    args = parser.parse_args()

    try:
        molecule, basis, mos, scf_results = _load_veloxchem_objects(args.vlx_h5)
    except ImportError:
        print("VeloxChem is required to compare VLX inputs.", file=sys.stderr)
        return 1

    basis_data = _build_basis_data(molecule, basis)
    mo_data = _build_mo_data(mos, basis_data["ao_permutation"])
    trexio_data = _read_trexio_data(args.trexio_file)

    failures = []

    _compare_scalar("nucleus_num", molecule.number_of_atoms(), trexio_data["nucleus_num"], 0.0, failures)
    _compare_array("nucleus_charge", molecule.get_element_ids(), trexio_data["nucleus_charge"], 0.0, failures)
    _compare_array("nucleus_coord", molecule.get_coordinates_in_bohr(), trexio_data["nucleus_coord"], args.tolerance, failures)

    _compare_scalar("electron_num", molecule.number_of_electrons(), trexio_data["electron_num"], 0.0, failures)
    _compare_scalar("electron_up_num", molecule.number_of_alpha_electrons(), trexio_data["electron_up_num"], 0.0, failures)
    _compare_scalar("electron_dn_num", molecule.number_of_beta_electrons(), trexio_data["electron_dn_num"], 0.0, failures)

    _compare_scalar("basis_shell_num", basis_data["shell_num"], trexio_data["basis_shell_num"], 0.0, failures)
    _compare_scalar("basis_prim_num", basis_data["prim_num"], trexio_data["basis_prim_num"], 0.0, failures)
    _compare_array("basis_nucleus_index", basis_data["nucleus_index"], trexio_data["basis_nucleus_index"], 0.0, failures)
    _compare_array("basis_shell_ang_mom", basis_data["shell_ang_mom"], trexio_data["basis_shell_ang_mom"], 0.0, failures)
    _compare_array("basis_shell_factor", basis_data["shell_factor"], trexio_data["basis_shell_factor"], args.tolerance, failures)
    _compare_array("basis_r_power", basis_data["r_power"], trexio_data["basis_r_power"], 0.0, failures)
    _compare_array("basis_shell_index", basis_data["shell_index"], trexio_data["basis_shell_index"], 0.0, failures)
    _compare_array("basis_exponent", basis_data["exponent"], trexio_data["basis_exponent"], args.tolerance, failures)
    _compare_array("basis_coefficient", basis_data["coefficient"], trexio_data["basis_coefficient"], args.tolerance, failures)
    _compare_array("basis_prim_factor", basis_data["prim_factor"], trexio_data["basis_prim_factor"], args.tolerance, failures)

    _compare_scalar("ao_num", basis_data["ao_num"], trexio_data["ao_num"], 0.0, failures)
    _compare_array("ao_shell", basis_data["ao_shell"], trexio_data["ao_shell"], 0.0, failures)
    _compare_array("ao_normalization", basis_data["ao_normalization"], trexio_data["ao_normalization"], args.tolerance, failures)

    _compare_scalar("mo_num", mo_data["coefficient"].shape[0], trexio_data["mo_num"], 0.0, failures)
    _compare_array("mo_coefficient", mo_data["coefficient"], trexio_data["mo_coefficient"], args.tolerance, failures)
    _compare_array("mo_energy", mo_data["energy"], trexio_data["mo_energy"], args.tolerance, failures)
    _compare_array("mo_occupation", mo_data["occupation"], trexio_data["mo_occupation"], args.tolerance, failures)

    expected_spin = mo_data["spin"]
    actual_spin = trexio_data["mo_spin"]
    if expected_spin is None and actual_spin is None:
        print("[OK] mo_spin: absent in both files")
    elif expected_spin is None or actual_spin is None:
        print("[FAIL] mo_spin: present in only one file")
        failures.append("mo_spin")
    else:
        _compare_array("mo_spin", expected_spin, actual_spin, 0.0, failures)

    expected_overlap = scf_results.get("S")
    actual_overlap = trexio_data["ao_overlap"]
    if expected_overlap is None and actual_overlap is None:
        print("[OK] ao_overlap: absent in both files")
    elif expected_overlap is None or actual_overlap is None:
        print("[FAIL] ao_overlap: present in only one file")
        failures.append("ao_overlap")
    else:
        order = basis_data["ao_permutation"]
        _compare_array("ao_overlap", np.asarray(expected_overlap)[np.ix_(order, order)], actual_overlap, args.tolerance, failures)

    expected_state_energy = scf_results.get("scf_energy")
    actual_state_energy = trexio_data["state_energy"]
    if expected_state_energy is None and actual_state_energy is None:
        print("[OK] state_energy: absent in both files")
    elif expected_state_energy is None or actual_state_energy is None:
        print("[FAIL] state_energy: present in only one file")
        failures.append("state_energy")
    else:
        actual_value = np.asarray(actual_state_energy).reshape(-1)[0]
        _compare_scalar("state_energy", expected_state_energy, actual_value, args.tolerance, failures)

    if failures:
        print("\nMismatches:")
        for name in failures:
            print(f" - {name}")
        return 1

    print("\nAll compared quantities match within tolerance.")
    return 0


if __name__ == "__main__":
    sys.exit(main())