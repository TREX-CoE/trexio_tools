#!/usr/bin/env python3

import trexio
import numpy as np
from . import nucleus

def read(trexio_file):
    r = {}

    r["nucleus"]            =  nucleus.read(trexio_file)

    r["type"]               =  trexio.read_basis_type(trexio_file)
    r["shell_num"]          =  trexio.read_basis_shell_num(trexio_file)
    r["prim_num"]           =  trexio.read_basis_prim_num(trexio_file)
    r["nucleus_index"]      =  trexio.read_basis_nucleus_index(trexio_file)
    r["shell_ang_mom"]      =  trexio.read_basis_shell_ang_mom(trexio_file)
    r["shell_factor"]       =  trexio.read_basis_shell_factor(trexio_file)
    r["shell_index"]        =  trexio.read_basis_shell_index(trexio_file)
    r["exponent"]           =  trexio.read_basis_exponent(trexio_file)
    r["coefficient"]        =  trexio.read_basis_coefficient(trexio_file)
    r["prim_factor"]        =  trexio.read_basis_prim_factor(trexio_file)

    return r


def convert_to_old(basis: dict) -> dict:
    """Convert the new basis set format into the old one (<2.0.0)."""

    basis_old = {}
    # The quantities below did not change in v.2.0.0
    basis_old["type"]               =  basis["type"]
    basis_old["prim_num"]           =  basis["prim_num"]  
    basis_old["shell_ang_mom"]      =  basis["shell_ang_mom"]
    basis_old["shell_factor"]       =  basis["shell_factor"]
    basis_old["exponent"]           =  basis["exponent"]
    basis_old["coefficient"]        =  basis["coefficient"]
    basis_old["prim_factor"]        =  basis["prim_factor"] 
    # basis_num has been renamed into basis_shell_num
    basis_old["num"]                =  basis["shell_num"]
    # The per-nucleus and per-shell lists below have to be reconstructed from the 
    # `nucleus_index` and `shell_index` maps, respectively, introduced in v.2.0.0
    # Save the data in the old format (index of the first shell and No of shells per atom)
    l1, l2 = map_to_lists(basis["nucleus_index"], basis["nucleus"]["num"])
    basis_old["nucleus_index"] = l1 
    basis_old["nucleus_shell_num"] = l2
    # Same for primitives per shell
    l3, l4 = map_to_lists(basis["shell_index"], basis["shell_num"])
    basis_old["shell_prim_index"] = l3 
    basis_old["shell_prim_num"] = l4

    return basis_old


def map_to_lists(map: list, dim: int) -> tuple:
    """Convert long map into two short ones (with index and number of elements per instance (e.g. atom), respectively)."""

    from collections import Counter

    index_per_instance = []
    instances_done = []
    for i, inst in enumerate(map):
        if not inst in instances_done:
            index_per_instance.append(i)
            instances_done.append(inst)

    n_per_instance = Counter(map)
    num_per_instance = [ n_per_instance[j] for j in range(dim) ]

    if(len(index_per_instance) != len(num_per_instance)):
        raise Exception(f"Inconsistent dimensions of output arrays: {len(index_per_instance)} != {len(num_per_instance)}")
    #print(map, "\n", index_per_instance, "\n", num_per_instance)

    return (index_per_instance, num_per_instance)


def lists_to_map(indices: list, numbers: list) -> list:
    """Convert two lists (with index and number of elements per instance like atom) into one big mapping list."""

    if(len(indices) != len(numbers)):
        raise Exception(f"Inconsistent dimensions of input arrays: {len(indices)} != {len(numbers)}")

    map = []
    for i, _ in enumerate(indices):
        for _ in range(numbers[i]):
            map.append(i)
    #print(indices, "\n", numbers, "\n", map)

    return map
