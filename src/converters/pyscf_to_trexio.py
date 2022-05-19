#!/usr/bin/env python
# coding: utf-8

# module information
__author__ = "Kosuke Nakano"
__copyright__ = "Copyright 2021, Kosuke Nakano (SISSA/JAIST)"
__version__ = "0.0.1"
__maintainer__ = "Kosuke Nakano"
__email__ = "kousuke_1123@icloud.com"
__date__ = "18. May. 2022"

# # pySCF checkpoint file -> TREX-IO

#Logger
from logging import config, getLogger, StreamHandler, Formatter
logger=getLogger('pyscf-trexio').getChild(__name__)

# load python packages
import os, sys
import numpy as np

def pyscf_checkfile_to_trexio(pyscf_checkfile, trexio_filename="trexio.hdf5"):

    # ## pySCF -> TREX-IO
    # - how to install trexio
    # - pip install trexio

    # import trexio
    import trexio

    # - how to install pyscf
    # - pip install pyscf
    
    # import pyscf
    import pyscf.scf
    from pyscf.pbc import scf as pbc_scf
    
    logger.info(f"pyscf_checkfile = {pyscf_checkfile}")
    logger.info(f"trexio_filename = {trexio_filename}")
    logger.info(f"Conversion starts...")

    # pyscf instances
    mol=pyscf.scf.chkfile.load_mol(pyscf_checkfile)
    mf=pyscf.scf.chkfile.load(pyscf_checkfile, "scf")
    
    # PBC info
    try:
        mol.a
        pbc_flag=True
    except AttributeError:
        pbc_flag=False
    logger.info(f"PBC flag = {pbc_flag}")
    
    # twist_average info
    if pbc_flag:
        try:
            k = mf['kpt']
            twist_average=False
            logger.info("Single-k calculation")
            k_list=[k]
        except KeyError:
            twist_average=True
            #logger.error("Twisted-average = True is not implemented")
            #raise NotImplementedError
            logger.error("Twisted-average calculation")
            logger.info("Separated TREXIO files are generated")
            logger.info("The Correspondence between the index and k is written in kp_info.dat")
            with open("kp_info.dat", "w") as f:
                f.write("# k_index, kx, ky, kz\n")
            k_list=mf['kpts']
        finally:
            mol=pbc_scf.chkfile.load_cell(pyscf_checkfile)
    else:
        twist_average=False
        k_list=[None]
    
    # if pbc_flag == true, check if ecp or pseudo
    if pbc_flag:
        if len(mol._pseudo) > 0:
            logger.error("TREXIO does not support 'pseudo' format. Plz. use 'ecp'")
            raise NotImplementedError
    
    # each k WF is stored as a separate file!!
    # for an open-boundary calculation, and a single-k one, 
    # k_index is a dummy variable
    for k_index, k_vec in enumerate(k_list):
        # set a filename
        if twist_average:
            logger.info(f"kpt={k_vec}")
            filename = f"k{k_index}_" + trexio_filename
            with open("kp_info.dat", "a") as f:
                f.write(f"{k_index} {k_vec[0]} {k_vec[1]} {k_vec[2]}\n")
        else:
            filename = trexio_filename
        if os.path.exists(filename): os.remove(filename)
        trexio_file = trexio.File(filename, mode='w', back_end=trexio.TREXIO_HDF5)
    
        ##########################################
        # PBC info
        ##########################################
        if pbc_flag:
            a=mol.a[0]; b=mol.a[1]; c=mol.a[2]
            k_point=k_vec
            periodic=True
        else:
            periodic=False
            
        # pbc and cell info
        trexio.write_pbc_periodic(trexio_file, periodic)
        if pbc_flag:
            trexio.write_cell_a(trexio_file, a)
            trexio.write_cell_b(trexio_file, b)
            trexio.write_cell_c(trexio_file, c)
            trexio.write_pbc_k_point(trexio_file, k_point)
        
        # structure info.
        electron_up_num, electron_dn_num=mol.nelec
        nucleus_num=mol.natm
        atom_charges_list=[mol.atom_charge(i) for i in range(mol.natm)]
        atom_nelec_core_list=[mol.atom_nelec_core(i) for i in range(mol.natm)]
        atomic_number_list=[mol.atom_charge(i) + mol.atom_nelec_core(i) for i in range(mol.natm)]
        chemical_symbol_list=[mol.atom_pure_symbol(i) for i in range(mol.natm)]
        coords_np=mol.atom_coords(unit='Bohr')
    
        ##########################################
        # Structure info
        ##########################################
        trexio.write_electron_up_num(trexio_file, electron_up_num)
        trexio.write_electron_dn_num(trexio_file, electron_dn_num)
        trexio.write_nucleus_num(trexio_file, nucleus_num)
        trexio.write_nucleus_charge(trexio_file, atom_charges_list)
        trexio.write_nucleus_label(trexio_file, chemical_symbol_list)
        trexio.write_nucleus_coord(trexio_file, coords_np)
    
        ##########################################
        # basis set info
        ##########################################
        # check the orders of the spherical atomic basis in pyscf!!
        # gto.spheric_labels(mol, fmt="%d, %s, %s, %s")
        # for s -> s
        # for p -> px, py, pz
        # for l >= d -> m=(-l ... 0 ... +l)
    
        basis_type="gto"
        basis_shell_num=int(np.sum([mol.atom_nshells(i) for i in range(nucleus_num)]))
        nucleus_index=[]
        for i in range(nucleus_num):
            for _ in range(len(mol.atom_shell_ids(i))):
                nucleus_index.append(i)
        shell_ang_mom=[mol.bas_angular(i) for i in range(basis_shell_num)]
        basis_prim_num=int(np.sum([mol.bas_nprim(i) for i in range(basis_shell_num)]))
    
        basis_exponent=[]
        basis_coefficient=[]
        for i in range(basis_shell_num):
            for bas_exp in mol.bas_exp(i):
                basis_exponent.append(float(bas_exp))
            for bas_ctr_coeff in mol.bas_ctr_coeff(i):
                basis_coefficient.append(float(bas_ctr_coeff))
    
        basis_shell_index=[]
        for i in range(basis_shell_num):
            for _ in range(len(mol.bas_exp(i))):
                basis_shell_index.append(i)
    
        # normalization factors
        basis_shell_factor = [1.0 for _ in range(basis_shell_num)] # 1.0 in pySCF
    
        # gto_norm(l, expnt) => l is angmom, expnt is exponent
        # Note!! Here, the normalization factor of the spherical part are not included.
        # The normalization factor is computed according to Eq.8 of the following paper
        # H. B. Schlegel and M. J. Frisch, Int. J. Quant.  Chem., 54(1995), 83-87.
        basis_prim_factor=[]
        for prim_i in range(basis_prim_num):
            coeff=basis_coefficient[prim_i]
            expnt=basis_exponent[prim_i]
            l=shell_ang_mom[basis_shell_index[prim_i]]
            basis_prim_factor.append(mol.gto_norm(l, expnt)/np.sqrt(4*np.pi)*np.sqrt(2*l+1))
    
        ##########################################
        # ao info
        ##########################################
        ao_cartesian = 0 # spherical basis representation
        ao_shell=[]
        for i, ang_mom in enumerate(shell_ang_mom):
            for _ in range(2*ang_mom + 1):
                ao_shell.append(i)
        ao_num=len(ao_shell)
    
        # 1.0 in pyscf (because spherical)
        ao_normalization = [1.0 for _ in range(ao_num)]
    
        ##########################################
        # mo info
        ##########################################
        mo_type="MO"
        
        if twist_average:
            mo_num=len(mf['mo_coeff'][k_index])
            mo_occupation=mf['mo_occ'][k_index]
            mo_energy=mf['mo_energy'][k_index]
            mo_coeff=mf['mo_coeff'][k_index]
        else:
            mo_num=len(mf['mo_coeff'])
            mo_occupation=mf['mo_occ']
            mo_energy=mf['mo_energy']
            mo_coeff=mf['mo_coeff']
        
        permutation_matrix=[] # for ao and mo swaps, used later
    
        # molecular orbital reordering
        # TREX-IO employs (m=-l,..., 0, ..., +l) for spherical basis
        mo_coefficient=[]
    
        for mo_i in range(mo_num):
            mo=mo_coeff[:,mo_i]
            mo_coeff_buffer=[]
    
            perm_list=[]
            perm_n=0
            for ao_i, ao_c in enumerate(mo):
    
                # initialization
                if ao_i==0:
                    mo_coeff_for_reord=[]
                    current_ang_mom=-1
    
                # read ang_mom (i.e., angular momentum of the shell)
                bas_i=ao_shell[ao_i]
                ang_mom=shell_ang_mom[bas_i]
    
                previous_ang_mom=current_ang_mom
                current_ang_mom=ang_mom
    
                # set multiplicity
                multiplicity = 2 * ang_mom + 1
                logger.debug(f"multiplicity = {multiplicity}")
    
                # check if the buffer is null, when ang_mom changes
                if previous_ang_mom != current_ang_mom:
                    assert len(mo_coeff_for_reord) == 0
    
                if current_ang_mom==0: # s shell
                    logger.debug("s shell/no permutation is needed.")
                    logger.debug("(pyscf notation): s(l=0)")
                    logger.debug("(trexio notation): s(l=0)")
                    reorder_index=[0]
    
                elif current_ang_mom==1: # p shell
    
                    logger.debug("p shell/permutation is needed.")
                    logger.debug("(pyscf notation): px(l=+1), py(l=-1), pz(l=0)")
                    logger.debug("(trexio notation): pz(l=0), px(l=+1), py(l=-1)")
                    reorder_index=[2,0,1]
    
    
                elif current_ang_mom>=2: # > d shell
    
                    logger.debug("> d shell/permutation is needed.")
                    logger.debug("(pyscf notation): e.g., f3,-3(l=-3), f3,-2(l=-2), f3,-1(l=-1), f3,0(l=0), f3,+1(l=+1), f3,+2(l=+2), f3,+3(l=+3)")
                    logger.debug("(trexio  notation): e.g, f3,0(l=0), f3,+1(l=+1), f3,-1(l=-1), f3,+2(l=+2), f3,-2(l=-2), f3,+3(l=+3), f3,-3(l=-3)")
                    l0_index=int((multiplicity-1)/2)
                    reorder_index=[l0_index]
                    for i in range(1, int((multiplicity-1)/2)+1):
                        reorder_index.append(l0_index+i)
                        reorder_index.append(l0_index-i)
    
                else:
                    raise ValueError("A wrong value was set to current_ang_mom.")
    
                mo_coeff_for_reord.append(ao_c)
    
                # write MOs!!
                if len(mo_coeff_for_reord) == multiplicity:
                    logger.debug("--write MOs!!--")
                    mo_coeff_buffer+=[mo_coeff_for_reord[i] for i in reorder_index]
    
                    # reset buffer
                    mo_coeff_for_reord=[]
    
                    logger.debug("--write perm_list")
                    perm_list+=list(np.array(reorder_index)+perm_n)
                    perm_n=perm_n+len(reorder_index)
    
            mo_coefficient.append(mo_coeff_buffer)
            permutation_matrix.append(perm_list)
        
        # here, we should think about complex cases
        if pbc_flag:
            if type(mo_coefficient[0][0])==np.complex128:
                complex_flag=True
            else:
                complex_flag=False
        else:
            complex_flag=False
            
        if complex_flag:
            logger.info("The WF is complex")
            mo_coefficient_real=[]
            mo_coefficient_imag=[]
            
            for mo__ in mo_coefficient:
                mo_real_b=[]
                mo_imag_b=[]
                for coeff in mo__:
                    mo_real_b.append(coeff.real)
                    mo_imag_b.append(coeff.imag)
                mo_coefficient_real.append(mo_real_b)
                mo_coefficient_imag.append(mo_imag_b)
        
        else:
            logger.info("The WF is real")
            
        logger.debug("--MOs Done--")
    
        ##########################################
        # atomic orbital integrals
        ##########################################
    
        def row_column_swap(inp_matrix, perm_list):
            mat_org=inp_matrix
            mat_row_swap=np.array([mat_org[i] for i in perm_list])
            mat_row_swap_T=mat_row_swap.T
            mat_row_swap_col_swap=np.array([mat_row_swap_T[i] for i in perm_list])
            mat_inv=mat_row_swap_col_swap.T
    
            #for i in range(len(mat_org)):
            #    for j in range(len(mat_org)):
            #        assert np.round(mat_inv[i][j],10) == np.round(mat_inv[j][i],10)
    
            return mat_inv
    
        perm_list=permutation_matrix[0]
        
        if pbc_flag:
            #logger.warning("1b integral for pbc is at gamma! Generic k points will be implemented.")
            intor_int1e_ovlp=row_column_swap(mol.pbc_intor("int1e_ovlp"), perm_list)
            intor_int1e_nuc=row_column_swap(mol.pbc_intor("int1e_nuc"), perm_list)
            intor_int1e_kin=row_column_swap(mol.pbc_intor("int1e_kin"), perm_list)
        else:
            intor_int1e_ovlp=row_column_swap(mol.intor("int1e_ovlp"), perm_list)
            intor_int1e_nuc=row_column_swap(mol.intor("int1e_nuc"), perm_list)
            intor_int1e_kin=row_column_swap(mol.intor("int1e_kin"), perm_list)
    
        ##########################################
        # basis set info
        ##########################################
        trexio.write_basis_type(trexio_file, basis_type) #
        trexio.write_basis_shell_num(trexio_file, basis_shell_num) #
        trexio.write_basis_prim_num(trexio_file, basis_prim_num) #
        trexio.write_basis_nucleus_index(trexio_file, nucleus_index) #
        trexio.write_basis_shell_ang_mom(trexio_file, shell_ang_mom) #
        trexio.write_basis_shell_factor(trexio_file, basis_shell_factor) #
        trexio.write_basis_shell_index(trexio_file, basis_shell_index) #
        trexio.write_basis_exponent(trexio_file, basis_exponent) #
        trexio.write_basis_coefficient(trexio_file, basis_coefficient) #
        trexio.write_basis_prim_factor(trexio_file, basis_prim_factor) #
    
        ##########################################
        # ao info
        ##########################################
        trexio.write_ao_cartesian(trexio_file, ao_cartesian) #
        trexio.write_ao_num(trexio_file, ao_num) #
        trexio.write_ao_shell(trexio_file, ao_shell) #
        trexio.write_ao_normalization(trexio_file, ao_normalization) #
    
        ##########################################
        # mo info
        ##########################################
        trexio.write_mo_type(trexio_file, mo_type) #
        trexio.write_mo_num(trexio_file, mo_num) #
        trexio.write_mo_occupation(trexio_file, mo_occupation) #
        if complex_flag:
            trexio.write_mo_coefficient(trexio_file, mo_coefficient_real) #
            trexio.write_mo_coefficient_im(trexio_file, mo_coefficient_imag) # 
        else:
            trexio.write_mo_coefficient(trexio_file, mo_coefficient) #
        ##########################################
        # ao integrals
        ##########################################
        trexio.write_ao_1e_int_overlap(trexio_file,intor_int1e_ovlp)
        trexio.write_ao_1e_int_kinetic(trexio_file,intor_int1e_kin)
        trexio.write_ao_1e_int_potential_n_e(trexio_file,intor_int1e_nuc)
    
        ##########################################
        # ECP
        ##########################################
        # internal format of pyscf
        # https://pyscf.org/pyscf_api_docs/pyscf.gto.html?highlight=ecp#module-pyscf.gto.ecp
        """
        { atom: (nelec,  # core electrons
        ((l, # l=-1 for UL, l>=0 for Ul to indicate |l><l|
        (((exp_1, c_1), # for r^0
        (exp_2, c_2), …),
        
        ((exp_1, c_1), # for r^1
        (exp_2, c_2), …),
        
        ((exp_1, c_1), # for r^2
        …))))),
        
        …}
        """
    
        # Note! here, the smallest l for the local part is l=1(i.e., p).
        # As a default, nwchem does not have a redundant non-local term (i.e., coeff=0) for H and He.
        
        if len(mol._ecp) > 0:
            logger.info("ECP info. is stored in the file.")
            ecp_num=0
            ecp_max_ang_mom_plus_1=[]
            ecp_z_core=[]
            ecp_nucleus_index=[]
            ecp_ang_mom=[]
            ecp_coefficient=[]
            ecp_exponent=[]
            ecp_power=[]
        
            for nuc_index, chemical_symbol in enumerate(chemical_symbol_list):
                logger.debug(f"Chemical symbol is {chemical_symbol}")
                z_core, ecp_list = mol._ecp[chemical_symbol]
        
                #ecp zcore
                ecp_z_core.append(z_core)
        
                #max_ang_mom+1
                max_ang_mom = max([ecp[0] for ecp in ecp_list]) # this is lmax.
                if max_ang_mom == -1: # special case!! H and He. PySCF database does not define the ul-s part for them.
                    max_ang_mom = 0
                    max_ang_mom_plus_1 = 1
                else:
                    max_ang_mom_plus_1 = max_ang_mom + 1
                
                ecp_max_ang_mom_plus_1.append(max_ang_mom_plus_1)
        
                for ecp in ecp_list:
                    ang_mom=ecp[0]
                    if ang_mom==-1:
                        ang_mom=max_ang_mom_plus_1
                    for r, exp_coeff_list in enumerate(ecp[1]):
                        for exp_coeff in exp_coeff_list:
                            exp,coeff = exp_coeff
        
                            #store variables!!
                            ecp_num+=1
                            ecp_nucleus_index.append(nuc_index)
                            ecp_ang_mom.append(ang_mom)
                            ecp_coefficient.append(coeff)
                            ecp_exponent.append(exp)
                            ecp_power.append(r-2)
        
                # special case!! H and He.
                # For the sake of clarity, here I put a dummy coefficient (0.0) for the ul-s part here.
                ecp_num+=1
                ecp_nucleus_index.append(nuc_index)
                ecp_ang_mom.append(0)
                ecp_coefficient.append(0.0)
                ecp_exponent.append(1.0)
                ecp_power.append(0)
        
            # write to the trex file
            trexio.write_ecp_num(trexio_file, ecp_num) #
            trexio.write_ecp_max_ang_mom_plus_1(trexio_file, ecp_max_ang_mom_plus_1) #
            trexio.write_ecp_z_core(trexio_file, ecp_z_core) #
            trexio.write_ecp_nucleus_index(trexio_file, ecp_nucleus_index) #
            trexio.write_ecp_ang_mom(trexio_file, ecp_ang_mom) #
            trexio.write_ecp_coefficient(trexio_file, ecp_coefficient) #
            trexio.write_ecp_exponent(trexio_file, ecp_exponent) #
            trexio.write_ecp_power(trexio_file, ecp_power) #
        
        else:
            logger.info("No ECP info. is stored in the file.")
        
        # close the TREX-IO file
        trexio_file.close()
    
    logger.info(f"Conversion to {trexio_filename} is done.")

if __name__ == "__main__":
    import argparse
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
    
    # define the parser
    parser = argparse.ArgumentParser(epilog='From pyscf chk file to TREXIO file', usage='python pyscf_to_trexio.py -c pyscf_checkfile -o trexio_filename', formatter_class=argparse.RawDescriptionHelpFormatter)
    # Job type:
    parser.add_argument('-c', '--pyscf_checkfile', help=f'pyscf checkfile', type=str, required=True)
    # Job ID:
    parser.add_argument('-o', '--trexio_filename', help=f'trexio filename', type=str, default="trexio.hdf5")
    
    # parse the input values
    args = parser.parse_args()
    parsed_parameter_dict=vars(args)
    
    pyscf_checkfile_to_trexio(pyscf_checkfile=args.pyscf_checkfile, trexio_filename=args.trexio_filename)