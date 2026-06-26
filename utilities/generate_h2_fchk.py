#!/usr/bin/env python3
"""Generate a real RHF Gaussian .fchk fixture for CI.

Light H2 system with a soft custom basis (s, sp, cartesian d) so that the
numerical grid in `trexio check-mos` resolves every function. MO coefficients
come from an actual PySCF RHF calculation, expressed in the unit-normalized
cartesian AO basis and Gaussian component ordering used by .fchk files.
"""
import sys
import numpy as np
from pyscf import gto, scf

BOND = 1.4  # bohr

# Per-H shells defined once as (l, [(exp, coef), ...]); the s@1.5 and p@1.5
# shells share an exponent and are written as a single Gaussian SP shell.
SHELLS = [
    (0, [(3.4, 0.4), (1.2, 0.6)]),
    (0, [(1.5, 1.0)]),
    (1, [(1.5, 1.0)]),
    (2, [(1.3, 1.0)]),
]
# Gaussian-component-order -> PySCF-component-order for a cartesian d shell.
# Gaussian: xx,yy,zz,xy,xz,yz   PySCF: xx,xy,xz,yy,yz,zz
D_G2P = [0, 3, 5, 1, 2, 4]


def pyscf_basis():
    return {'H': [[l] + [[e, c] for e, c in prims] for l, prims in SHELLS]}


def main(path):
    mol = gto.M(atom=f'H 0 0 0; H 0 0 {BOND}', unit='Bohr',
                basis=pyscf_basis(), cart=True)
    mf = scf.RHF(mol)
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("SCF did not converge; refusing to emit an "
                           "inconsistent fchk fixture.")

    S = mol.intor('int1e_ovlp')
    C = mf.mo_coeff                         # (nao, nmo), pyscf cart AO order
    scale = np.sqrt(np.diag(S))             # pyscf AO -> unit-normalized AO
    C_unit = C * scale[:, None]
    nao, nmo = C.shape

    # pyscf AO order per atom: s(2p), s, px,py,pz, d(6). Permute to fchk order.
    per_atom = 1 + 1 + 3 + 6
    perm = []
    for a in range(mol.natm):
        base = a * per_atom
        perm += [base, base + 1, base + 2, base + 3, base + 4]
        perm += [base + 5 + k for k in D_G2P]
    C_fchk = C_unit[perm, :]

    # FCHK shell description (per atom: S 2-prim, SP 1-prim, D 1-prim).
    shell_types, prim_per_shell, shell_atom = [], [], []
    exps, scoef, pcoef = [], [], []
    for a in range(mol.natm):
        shell_types += [0, -1, 2]
        prim_per_shell += [2, 1, 1]
        shell_atom += [a + 1, a + 1, a + 1]
        for (e, c) in SHELLS[0][1]:          # S shell
            exps.append(e); scoef.append(c); pcoef.append(0.0)
        exps.append(SHELLS[1][1][0][0])      # SP shell (shared exponent)
        scoef.append(SHELLS[1][1][0][1])     # s contraction coefficient
        pcoef.append(SHELLS[2][1][0][1])     # p contraction coefficient
        exps.append(SHELLS[3][1][0][0])      # D shell
        scoef.append(SHELLS[3][1][0][1]); pcoef.append(0.0)

    mo_flat = C_fchk.T.reshape(-1).tolist()
    energies = mf.mo_energy.tolist()

    def si(n, v): return f"{n:<40}   I     {v:>12}\n"
    def ai(n, v):
        out = f"{n:<40}   I   N={len(v):>12}\n"
        for i in range(0, len(v), 6):
            out += "".join(f"{x:>12}" for x in v[i:i + 6]) + "\n"
        return out
    def ar(n, v):
        out = f"{n:<40}   R   N={len(v):>12}\n"
        for i in range(0, len(v), 5):
            out += "".join(f"{x:>16.8E}" for x in v[i:i + 5]) + "\n"
        return out

    txt = "H2 soft-basis RHF fixture (s/sp/cart-d)\n"
    txt += "SP        RHF                                                         Custom\n"
    txt += si("Number of atoms", mol.natm)
    txt += si("Charge", mol.charge)
    txt += si("Multiplicity", mol.spin + 1)
    txt += si("Number of electrons", mol.nelectron)
    txt += si("Number of alpha electrons", mol.nelec[0])
    txt += si("Number of beta electrons", mol.nelec[1])
    txt += si("Number of basis functions", nao)
    txt += ai("Atomic numbers", [1] * mol.natm)
    txt += ar("Nuclear charges", [1.0] * mol.natm)
    txt += ar("Current cartesian coordinates", mol.atom_coords().reshape(-1).tolist())
    txt += ai("Shell types", shell_types)
    txt += ai("Number of primitives per shell", prim_per_shell)
    txt += ai("Shell to atom map", shell_atom)
    txt += ar("Primitive exponents", exps)
    txt += ar("Contraction coefficients", scoef)
    txt += ar("P(S=P) Contraction coefficients", pcoef)
    txt += ar("Alpha Orbital Energies", energies)
    txt += ar("Alpha MO coefficients", mo_flat)
    with open(path, "w") as fh:
        fh.write(txt)
    print("wrote", path, "nao", nao, "nmo", nmo, "E_scf", mf.e_tot)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(f"usage: {sys.argv[0]} OUTPUT.fchk")
    main(sys.argv[1])
