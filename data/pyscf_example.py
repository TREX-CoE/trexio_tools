#!/usr/bin/env python
# coding: utf-8

# module information
__author__ = "Kosuke Nakano"
__copyright__ = "Copyright 2021, Kosuke Nakano (SISSA/JAIST)"
__version__ = "0.0.1"
__maintainer__ = "Kosuke Nakano"
__email__ = "kousuke_1123@icloud.com"
__date__ = "18. May. 2022"

# # pySCF -> pyscf checkpoint file (Water molecule, Diamond with single-k and grid-k)

#Logger
from logging import config, getLogger, StreamHandler, Formatter
log_level="INFO"
logger = getLogger("pyscf-trexio")
logger.setLevel(log_level)
stream_handler = StreamHandler()
stream_handler.setLevel(log_level)
#handler_format = Formatter('%(name)s - %(levelname)s - %(lineno)d - %(message)s')
handler_format = Formatter('%(message)s')
stream_handler.setFormatter(handler_format)
logger.addHandler(stream_handler)

# load python packages
import os, sys
import numpy as np

# load ASE modules
from ase.io import read

# load pyscf packages
from pyscf import gto, scf, mp, tools
from pyscf.pbc import gto as gto_pbc
from pyscf.pbc import dft as pbcdft
from pyscf.pbc import scf as pbcscf
 
"""
#open boundary condition
structure_file="water.xyz"
checkpoint_file="water.chk"
pyscf_output="out_water_pyscf"
charge=0
spin=0
basis="ccecp-ccpvtz"
ecp='ccecp'
scf_method="DFT"  # HF or DFT
dft_xc="LDA_X,LDA_C_PZ" # XC for DFT
"""

"""
#periodic boundary condition (single-k)
structure_file="diamond.cif"
checkpoint_file="diamond_single_k.chk"
pyscf_output="out_diamond_pyscf"
charge=0
spin=0
basis="ccecp-ccpvtz"
exp_to_discard=0.10
ecp="ccecp"
scf_method="DFT"  # HF or DFT
dft_xc="LDA_X,LDA_C_PZ" # XC for DFT
twist_average=False
kpt=[0,0,0]
"""

#periodic boundary condition (k-grid)
structure_file="diamond.cif"
checkpoint_file="diamond_k_grid.chk"
pyscf_output="out_diamond_pyscf"
charge=0
spin=0
basis="ccecp-ccpvtz"
exp_to_discard=0.10
ecp="ccecp"
scf_method="DFT"  # HF or DFT
dft_xc="LDA_X,LDA_C_PZ" # XC for DFT
twist_average=True
kpt_grid=[2,2,2]

###########
logger.info(f"structure file = {structure_file}")

atom=read(structure_file)
pbc_flag=atom.get_cell().any()

if pbc_flag:
    logger.info("Periodic System")
else:
    logger.info("Molecule")

#PBC case
if pbc_flag:
    cell=gto_pbc.M()
    cell.from_ase(atom)

    cell.verbose = 5
    cell.output = pyscf_output
    cell.charge = charge
    cell.spin = spin
    cell.symmetry = False
    a=cell.a
    cell.a=np.array([a[0], a[1], a[2]]) # otherwise, we cannot dump a
    # basis set
    cell.basis = basis
    cell.exp_to_discard=exp_to_discard

    # define ecp
    if ecp is not None: cell.ecp = ecp
    
    cell.build(cart=False)
    
    # calc type setting
    logger.info(f"scf_method = {scf_method}")  # HF/DFT
    
    if scf_method == "HF":
        # HF calculation
        if cell.spin == 0:
            logger.info("HF kernel=RHF")
            if twist_average:
                logger.info("twist_average=True")
                kpt_grid = cell.make_kpts(kpt_grid)
                mf = pbcscf.khf.kRHF(cell, kpt_grid)
                mf = mf.newton()
            else:
                logger.info("twist_average=False")
                mf = pbcscf.hf.RHF(cell, kpt=kpt)
                mf = mf.newton()
            
        else:
            logger.info("HF kernel=ROHF")
            if twist_average:
                logger.info("twist_average=True")
                kpt_grid = cell.make_kpts(kpt_grid)
                mf = pbcscf.krohf.KROHF(cell, kpt_grid)
                mf = mf.newton()
            else:
                logger.info("twist_average=False")
                mf = pbcscf.rohf.ROHF(cell, kpt=kpt)
                mf = mf.newton()
        
        mf.chkfile = checkpoint_file
        
    elif scf_method == "DFT":
        # DFT calculation
        if cell.spin == 0:
            logger.info("DFT kernel=RKS")
            if twist_average:
                logger.info("twist_average=True")
                kpt_grid = cell.make_kpts(kpt_grid)
                mf = pbcdft.krks.KRKS(cell, kpt_grid)
                mf = mf.newton()
                #print(dir(mf))
                #sys.exit()
            else:
                logger.info("twist_average=False")
                mf = pbcdft.rks.RKS(cell, kpt=kpt)
                mf = mf.newton()
        else:
            logger.info("DFT kernel=ROKS")
            if twist_average:
                logger.info("twist_average=True")
                kpt_grid = cell.make_kpts(kpt_grid)
                mf = pbcdft.kroks.KROKS(cell, kpt_grid)
                mf = mf.newton()
            else:
                logger.info("twist_average=False")
                mf = pbcdft.roks.ROKS(cell, kpt=kpt)
                mf = mf.newton()
        
        mf.chkfile = checkpoint_file
        mf.xc = dft_xc
    else:
        raise NotImplementedError
    
    total_energy = mf.kernel()

#Open system
else:
    chemical_symbols=atom.get_chemical_symbols()
    positions=atom.get_positions()
    mol_string=""
    for chemical_symbol, position in zip(chemical_symbols, positions):
        mol_string+="{:s} {:.10f} {:.10f} {:.10f} \n".format(chemical_symbol, position[0], position[1], position[2])
    # build a molecule
    mol = gto.Mole()
    mol.atom = mol_string
    mol.verbose = 5
    mol.output = pyscf_output
    mol.unit = 'A' # angstrom
    mol.charge = charge
    mol.spin = spin
    mol.symmetry = False

    # basis set
    mol.basis = basis

    # define ecp
    if ecp is not None: mol.ecp = ecp

    # molecular build
    mol.build(cart=False)  # cart = False => use spherical basis!!

    # calc type setting
    logger.info(f"scf_method = {scf_method}")  # HF/DFT
    
    if scf_method == "HF":
        # HF calculation
        if mol.spin == 0:
            logger.info("HF kernel=RHF")
            mf = scf.RHF(mol)
            mf.chkfile = checkpoint_file
        else:
            logger.info("HF kernel=ROHF")
            mf = scf.ROHF(mol)
            mf.chkfile = checkpoint_file
    
    elif scf_method == "DFT":
        # DFT calculation
        if mol.spin == 0:
            logger.info("DFT kernel=RKS")
            mf = scf.KS(mol).density_fit()
            mf.chkfile = checkpoint_file
        else:
            logger.info("DFT kernel=ROKS")
            mf = scf.ROKS(mol)
            mf.chkfile = checkpoint_file
        mf.xc = dft_xc
    else:
        raise NotImplementedError
    
    total_energy = mf.kernel()

# Molecular Orbitals and occupations
logger.info("MOs-HF/DFT")
logger.info(mf.mo_coeff)  # HF/DFT coeff
logger.info(mf.mo_occ)  # HF/DFT occ
logger.info(mf.mo_energy)  # HF/DFT energy
# Notice!! The mo_i-th molecular orbital is NOT mo_coeff[mo_i], but mo_coeff[:,mo_i] !!

# HF/DFT energy
logger.info(f"Total HF/DFT energy = {total_energy}")
logger.info("HF/DFT calculation is done.")
logger.info("PySCF calculation is done.")
logger.info(f"checkpoint file = {checkpoint_file}")