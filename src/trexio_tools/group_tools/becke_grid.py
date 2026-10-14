#!/usr/bin/env python3
"""
Becke multi-center integration grid for molecular integrals.

Implements:
- Mura-Knowles radial grid with log mapping
- Product angular grid (Gauss-Legendre × uniform phi)
- Becke partitioning for multi-center integration

References:
    A.D. Becke, J. Chem. Phys. 88, 2547 (1988)
    M.E. Mura, P.J. Knowles, J. Chem. Phys. 104, 9848 (1996)
"""

import numpy as np


# Bragg-Slater radii (in Bohr) for elements H(1) through Xe(54).
# Used as the radial scaling parameter R_m for each atom.
_BRAGG_SLATER_RADII = {
     1: 0.661,   2: 0.567,   3: 2.740,   4: 1.984,   5: 1.606,
     6: 1.417,   7: 1.228,   8: 1.134,   9: 1.039,  10: 0.945,
    11: 3.402,  12: 2.835,  13: 2.362,  14: 2.079,  15: 1.890,
    16: 1.795,  17: 1.701,  18: 1.606,  19: 4.158,  20: 3.402,
    21: 2.835,  22: 2.646,  23: 2.551,  24: 2.457,  25: 2.457,
    26: 2.457,  27: 2.362,  28: 2.362,  29: 2.362,  30: 2.362,
    31: 2.362,  32: 2.173,  33: 2.079,  34: 2.079,  35: 1.984,
    36: 1.890,  37: 4.441,  38: 3.780,  39: 3.213,  40: 2.835,
    41: 2.646,  42: 2.551,  43: 2.457,  44: 2.362,  45: 2.362,
    46: 2.362,  47: 2.362,  48: 2.646,  49: 2.646,  50: 2.457,
    51: 2.362,  52: 2.362,  53: 2.268,  54: 2.173,
}

_DEFAULT_RADIUS = 2.0  # Default radius in Bohr for unknown elements


def _get_radius(charge):
    """Get Bragg-Slater radius for a given nuclear charge."""
    z = int(round(charge))
    return _BRAGG_SLATER_RADII.get(z, _DEFAULT_RADIUS)


def _angular_grid(n_theta):
    """
    Generate angular grid using product Gauss-Legendre × uniform quadrature.

    Parameters
    ----------
    n_theta : int
        Number of Gauss-Legendre points for cos(theta).

    Returns
    -------
    directions : ndarray, shape (n_angular, 3)
        Unit vectors on the sphere.
    weights : ndarray, shape (n_angular,)
        Angular quadrature weights (integrate to 4*pi for f=1).
    """
    cos_theta, w_theta = np.polynomial.legendre.leggauss(n_theta)

    n_phi = 2 * n_theta
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    w_phi = 2 * np.pi / n_phi

    sin_theta = np.sqrt(1 - cos_theta**2)

    # Build product grid using broadcasting
    # shape: (n_theta, n_phi)
    x = np.outer(sin_theta, np.cos(phi))
    y = np.outer(sin_theta, np.sin(phi))
    z = np.outer(cos_theta, np.ones(n_phi))
    w = np.outer(w_theta, np.ones(n_phi)) * w_phi

    directions = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    weights = w.ravel()

    return directions, weights


def _radial_grid(n_radial, R_m=1.0):
    """
    Generate radial grid using the Mura-Knowles log mapping.

    The mapping is: r = -R_m * ln(1 - x^3), x = i/(n+1)

    Reference:
        M.E. Mura, P.J. Knowles, J. Chem. Phys. 104, 9848 (1996)

    Parameters
    ----------
    n_radial : int
        Number of radial grid points.
    R_m : float
        Radial scaling parameter (typically Bragg-Slater radius).

    Returns
    -------
    radii : ndarray, shape (n_radial,)
        Radial grid points.
    weights : ndarray, shape (n_radial,)
        Radial quadrature weights (including r^2 Jacobian).
    """
    i = np.arange(1, n_radial + 1)
    x = i / (n_radial + 1)

    r = -R_m * np.log(1 - x**3)
    dr_dx = 3 * R_m * x**2 / (1 - x**3)

    # Include r^2 from spherical coordinates, the Jacobian, and the
    # simple quadrature weight 1/(n+1)
    weights = r**2 * dr_dx / (n_radial + 1)

    return r, weights


def _becke_partition_weights(atom_points, nuclear_coords, atom_index):
    """
    Compute Becke partition weights for a set of points belonging to atom atom_index.

    Parameters
    ----------
    atom_points : ndarray, shape (n_points, 3)
        Grid points centered on atom atom_index.
    nuclear_coords : ndarray, shape (n_atoms, 3)
        Nuclear coordinates.
    atom_index : int
        Index of the atom these points belong to.

    Returns
    -------
    partition_weights : ndarray, shape (n_points,)
        Becke partition weights for atom atom_index at each point.
    """
    n_atoms = len(nuclear_coords)
    n_points = len(atom_points)

    if n_atoms == 1:
        return np.ones(n_points)

    # Distances from each point to each nucleus: shape (n_points, n_atoms)
    dist = np.array([np.linalg.norm(atom_points - nuclear_coords[a], axis=1)
                     for a in range(n_atoms)]).T

    # Pre-compute inter-nuclear distances
    R_kl = np.zeros((n_atoms, n_atoms))
    for k in range(n_atoms):
        for l in range(k + 1, n_atoms):
            d = np.linalg.norm(nuclear_coords[k] - nuclear_coords[l])
            R_kl[k, l] = d
            R_kl[l, k] = d

    # Compute the raw Becke partition for each atom: shape (n_points, n_atoms)
    P = np.ones((n_points, n_atoms))

    for k in range(n_atoms):
        for l in range(n_atoms):
            if k == l:
                continue
            mu = (dist[:, k] - dist[:, l]) / R_kl[k, l]
            # Apply Becke's smoothing function 3 times
            for _ in range(3):
                mu = 1.5 * mu - 0.5 * mu**3
            s = 0.5 * (1 - mu)
            P[:, k] *= s

    P_sum = np.sum(P, axis=1)
    # Avoid division by zero
    mask = P_sum > 0
    result = np.zeros(n_points)
    result[mask] = P[:, atom_index][mask] / P_sum[mask]

    return result


def becke_grid(nuclear_coords, nuclear_charges=None, n_radial=75, n_angular=15):
    """
    Generate a Becke multi-center integration grid.

    Parameters
    ----------
    nuclear_coords : ndarray, shape (n_atoms, 3)
        Nuclear coordinates in Bohr.
    nuclear_charges : array_like, shape (n_atoms,), optional
        Nuclear charges (used to select Bragg-Slater radii).
        If None, a default radius is used for all atoms.
    n_radial : int
        Number of radial grid points per atom.
    n_angular : int
        Number of Gauss-Legendre theta points; total angular points
        per radial shell = n_angular × 2*n_angular.

    Returns
    -------
    points : ndarray, shape (n_total, 3)
        Grid points.
    weights : ndarray, shape (n_total,)
        Integration weights.
    """
    n_atoms = len(nuclear_coords)
    nuclear_coords = np.asarray(nuclear_coords)

    # Angular grid (same for all atoms)
    ang_dirs, w_ang = _angular_grid(n_angular)
    n_ang = len(ang_dirs)

    all_points = []
    all_weights = []

    for k in range(n_atoms):
        # Determine radial scaling
        if nuclear_charges is not None:
            R_m = _get_radius(nuclear_charges[k])
        else:
            R_m = _DEFAULT_RADIUS

        # Radial grid for this atom
        radii, w_rad = _radial_grid(n_radial, R_m)

        # Build atom-centered grid: shape (n_radial * n_ang, 3)
        # For each radius, place angular points
        atom_pts = np.empty((n_radial * n_ang, 3))
        atom_wts = np.empty(n_radial * n_ang)

        for i_r in range(n_radial):
            start = i_r * n_ang
            end = start + n_ang
            atom_pts[start:end] = nuclear_coords[k] + radii[i_r] * ang_dirs
            atom_wts[start:end] = w_rad[i_r] * w_ang

        # Apply Becke partition
        partition = _becke_partition_weights(atom_pts, nuclear_coords, k)
        atom_wts *= partition

        all_points.append(atom_pts)
        all_weights.append(atom_wts)

    points = np.concatenate(all_points, axis=0)
    weights = np.concatenate(all_weights, axis=0)

    return points, weights
