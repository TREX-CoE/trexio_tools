#!/usr/bin/env python3

"""
    Script to numerically calculate the overlap integrals for numerical
    orbitals of the FHIaims-type

    Written by Johannes Günzl, TU Dresden 2023
"""

import trexio
import numpy as np
from scipy.special import sph_harm, factorial

from . import basis as trexio_basis

ao_exponents_l_cart = []
#    [[0, 0, 0]],
#    [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
#    [[2, 0, 0], [1, 1, 0], [1, 0, 1], [0, 2, 0], [0, 1, 1], [0, 0, 2]] etc

# Generate l-exponents for cartesian orbitals
for l in range(10):
    sublist = []
    for x in range(l + 1):
        for y in range(l - x + 1):
            sublist.append([x, y, l - x - y])
    sublist.reverse()
    ao_exponents_l_cart.append(sublist)


def read(trexio_file):
    r = {}

    r["basis"]     = trexio_basis.read(trexio_file)
    r["num"]       = trexio.read_ao_num(trexio_file)
    r["shell"]     = trexio.read_ao_shell(trexio_file)
    r["factor"]    = trexio.read_ao_normalization(trexio_file)
    r["cartesian"]    = trexio.read_ao_cartesian(trexio_file)

    # Since this varibale is not used for spherical harmonics, this is fine
    ao_exponents_l = ao_exponents_l_cart

    ao_num = r["num"]
    basis = r["basis"]
    ao_shell = r["shell"]
    shell_ang_mom = basis["shell_ang_mom"]

    ao_exponents = np.zeros((ao_num, 3), dtype=float)
    same_shell_cnt = 0
    last_shell = -1

    for i in range(ao_num):
        curr_shell = ao_shell[i]
        if curr_shell != last_shell:
            same_shell_cnt = -1
            last_shell = curr_shell
        same_shell_cnt += 1
        l = shell_ang_mom[curr_shell]
        ao_exponents[i] = ao_exponents_l[l][same_shell_cnt]

    r["ao_exponents"] = ao_exponents

    return r

def shell_to_ao(ao, ao_ind, r, shell_rad, m):
    ao_shell = ao["shell"]
    basis = ao["basis"]
    shell_ang_mom = basis["shell_ang_mom"]
    nucleus_index     = basis["nucleus_index"]
    nucleus_coord     = basis["nucleus"]["coord"]
    ao_norms     = ao["factor"]

    l = shell_ang_mom[ao_shell[ao_ind]]
    nuc_coords = nucleus_coord[nucleus_index[ao_shell[ao_ind]]]

    dx = r[0] - nuc_coords[0]
    dy = r[1] - nuc_coords[1]
    dz = r[2] - nuc_coords[2]

    dr = np.sqrt(dx**2 + dy**2 + dz**2)
    if ao["cartesian"] == 1:
        exps = ao["ao_exponents"][ao_ind]
        angle_part = dx**exps[0] * dy**exps[1] * dz**exps[2] * dr**-l
        ret = shell_rad * angle_part * ao_norms[ao_ind]
    else:
        theta = np.arccos(dz/dr)
        phi = np.sign(dy)*np.arccos(dx/(dx**2+dy**2)**(0.5))

        # Convert from Gamess convetion to FHI-aims convention (each radial
        # function is normalized with an s-orbital) since it is more
        # convenient here
        l_factor = [1, 1, 2, 2, 8, 8, 16, 16, 128, 128, 256]
        l_factor = np.sqrt(np.array([
            (2*l + 1) / l_factor[l]**2 for l in range(11)
            ]) * (1/np.pi))/2

        shell_rad /= l_factor[l]

        angle_part = sph_harm(np.abs(m), l, phi, theta)
        if m == 0:
            angle_part = np.real(angle_part)
        if m < 0:
            angle_part = np.sqrt(2)*(-1)**m*np.imag(angle_part)
        if m > 0:
            angle_part = np.sqrt(2)*(-1)**m*np.real(angle_part)

        # ao_norm is handled by scipy
        ret = shell_rad * angle_part

    return ret

def value(ao,r):
    """
    Evaluates all the basis functions at R=(x,y,z)
    """

    basis           = ao["basis"]
    ao_num          = ao["num"]

    nucleus         = basis["nucleus"]
    coord           = nucleus["coord"]
    nucleus_num     = nucleus["num"]

    basis_num       = basis["shell_num"]
    shell_ang_mom   = basis["shell_ang_mom"]

    nucleus_index   = basis["nucleus_index"]

    nao_grid_start   = basis["nao_grid_start"]
    nao_grid_size    = basis["nao_grid_size"]
    nao_grid_r       = basis["nao_grid_radius"]
    interpolator    = basis["interpolator"]
    shell_factor    = basis["shell_factor"]
    norm            = ao["factor"]
    ao_shell        = ao["shell"]

    amplitudes = trexio.evaluate_nao_radial_all(nucleus_index, coord, 
        nao_grid_start, nao_grid_size, nao_grid_r, interpolator, shell_factor, r)

    ao_amp = np.zeros(ao_num, dtype=float)
    last_shell = ao_shell[0]
    n = 0 # Index of current ao within current shell
    for a in range(ao_num):
        curr_shell = ao_shell[a]
        if curr_shell != last_shell:
            last_shell = curr_shell
            n = 0

        m = (n+1)//2 * (2*(n % 2) - 1)
        ao_amp[a] = shell_to_ao(ao, a, r, amplitudes[curr_shell], m)
        n += 1

    #print(r, ao_amp)

    return ao_amp

