
"""
Script to gather data from FHI-aims output files and load it into trexio
Implemented for non-periodic systems
Written by Johanens Günzl, TU Dresden 2023

Note that, since there are several input files, the input is given as the
path to the directory that contains these files.

The following files are always required:
    control.in
    geometry.in
    
    as well as the directive
        elsi_output_matrix eigenvectors
    in control.in

control.in must contain the following directives to load basis sets:
    output basis
    output h_s_matrices

control.in must contain the following keywords to load integrals:
    xc hf
    elsi_output_matrix overlap
    elsi_output_matrix hamiltonian
    calculate all eigenstates
    output_ks_coulomb_integral [full or qtmp]

    Within this project, the functionality to print atomic orbital four
    center integrals has been added, so the following keyword might not be
    available for other users
    output core_hamiltonian
    output_ao_coulomb_integral [full or qtmp]
""" 

"""
All data needs to be converted to atomic units!
FHI-aims used eV and Angstrom for their output where
    1 Ha = 27.2113845 eV
    1 a_0 = 0.52917721 Angstrom
(see src/constant.f90 in the code)
"""

try:
    import trexio
except ImportError:
    raise Exception("Unable to import trexio. Please check that the trexio \
                    Python package is properly installed.")

import struct
import sys
import os
import numpy as np
from scipy.linalg.lapack import dgtsv # For interpolation parameters
from scipy.linalg import block_diag # Block matrices for spin polarization
from scipy.sparse import csc_matrix, coo_matrix # Read ELSI matrices


hartree_to_ev = 27.2113845
bohr_to_angstrom = 0.52917721

class Species:
    def __init__(self, id_num, symbol, charge):
        self.id_num = id_num
        self.symbol = symbol
        self.charge = charge
        self.nao_grid_ids = []
        self.nao_grid_sizes = []
        # Species list is loaded from control.in bc it contains all info,
        # but (shell) order is determined by geometry
        self.geom_index = 10000 # -1 would disturb sorting for unused species

    def add_nao_grid(self, id_num, size):
        self.nao_grid_ids.append(id_num)
        self.nao_grid_sizes.append(size)

    def __repr__(self):
        return f"Species(id={self.id_num}, symbol={self.symbol})"

class Atom:
    def __init__(self, id_num, species, coords):
        self.id_num = id_num
        self.species = species
        self.coords = coords

class OrbitalShell:
    def __init__(self, id_num, atom_id, fn_type, n, l):
        self.id_num = id_num
        self.atom_id = atom_id
        self.fn_type = fn_type[:2].lower() # Needed to find the u(r) file
        self.n = n
        self.l = l
        self.nao_grid_id = -1 # Dummy values
        self.nao_grid_size = -1

    def __repr__(self):
        #return f"OS(id_num={self.id_num})"
        return f"OS(id_num={self.id_num},atomid_id={self.atom_id}," \
            + f"fn_type={self.fn_type},n={self.n},l={self.l})"

class AtomicOrbital:
    def __init__(self, id_num, shell, m, normalization):
        self.id_num = id_num # The position in the input order, i.e. input matrices etc
        self.shell = shell
        self.m = m
        self.normalization = normalization
        self.r_power = -1

    def __repr__(self):
        #return f"AtomicOrbital[id_num={self.id_num}, shell={self.shell},"\
        #    + f" m={self.m}, normalization={self.normalization}]"
        return f"AO[id={self.id_num}, shell={self.shell.id_num},"\
            + f" m={self.m}]"

    def __lt__(self, other):
        # Needed to sort the ao list according to their m
        # Only switch orbitals that differ only in their m
        if self.shell.atom_id != other.shell.atom_id:
            return self.shell.atom_id < other.shell.atom_id
        if self.shell.id_num != other.shell.id_num:
            return 0
        if abs(self.m) == abs(other.m):
            return self.m > other.m
        return abs(self.m) < abs(other.m)

class Context:
    def __init__(self, dirpath, aimsoutpath):
        self.dirpath = dirpath
        self.aimsoutpath = aimsoutpath

        # Everything is initialized to None so availability can be checked
        self.species = None
        self.system_charge = None
        self.spin_moment = None
        self.has_core_hole = False
        self.rohf = False

        self.atoms = None
        self.nuclear_charge = None

        self.aos = None
        self.elec_cnt = None
        self.elec_up = None
        self.elec_dn = None

        self.ao_num = None
        self.mo_num = None

        self.mo_occupation = None
        # Permutation from the FHI-aims orbital order to trexio
        self.matrix_indices = None
        # The signs of matrix entries depends on whether the angular part
        # of orbitals is calculated by r(elec) - r(nuc) or r(nuc) - r(elec).
        # To enforce r(elec) - r(nuc), some signs have to be flipped
        self.matrix_signs = None

    def ao_count(self):
        return len(self.aos)

    # Occupation number per orbital
    def nspin(self):
        if spin_moment == 0:
            return 2
        return 1
    
    def unrestricted(self):
        return self.spin_moment != 0 and not self.rohf

class SparseData:
    def __init__(self, trexfile, write_function, batch_size):
        self.trexfile = trexfile
        self.write_f = write_function
        
        self.batch_offset = 0
        self.batch_size = batch_size

        self.batch_indices = np.zeros((batch_size, 4), dtype=int)
        self.batch_vals = np.zeros(batch_size, dtype=float)
        self.index_in_batch = 0

    def write_batch(self):
        # Check if there is anything left to flush
        if self.index_in_batch == 0:
            return

        self.write_f(self.trexfile, self.batch_offset,
                     self.index_in_batch,
                     self.batch_indices, self.batch_vals)

        self.batch_offset += self.index_in_batch
        self.index_in_batch = 0

    def add(self, indices, val):
        self.batch_indices[self.index_in_batch] = indices
        self.batch_vals[self.index_in_batch] = val

        self.index_in_batch += 1

        if self.index_in_batch >= self.batch_size:
            self.write_batch()

# For debugging purposes
def print_matrix(mat, title, thresh=1e-10):
    if mat is None:
        return
    print(title)
    print(np.where(np.fabs(mat) > thresh, mat, 0))
    print()

def angular_letter(l):
    letters = "spdfghijklmno"
    return letters[l]

def create_cubic_spline(r_vals, phi_vals):
    f_grid = phi_vals
    data_cnt = len(r_vals)
    vector = np.zeros((data_cnt,))
    matrix_diag = np.zeros((data_cnt,))
    matrix_upper = np.ones((data_cnt - 1,))
    matrix_lower = np.ones((data_cnt - 1,))

    # First point
    vector[0] = 3*(f_grid[1] - f_grid[2])
    matrix_diag[0] = 2

    for i in range(1, data_cnt - 1):
        vector[i] = 3*(f_grid[i] - f_grid[i - 1])
        matrix_diag[i] = 4

    # Last grid point
    vector[-1] = 3 * (f_grid[-1] - f_grid[-2])
    matrix_diag[-1] = 2

    ret = dgtsv(matrix_lower, matrix_diag, matrix_upper, vector, 1, 1, 1, 1)
    if ret[4] != 0:
        print("Something went wrong in dgtsv")
        exit(-1)

    spline_params = np.zeros((data_cnt, 4), dtype=float)
    for i in range(data_cnt - 1):
        spline_params[i, 0] = phi_vals[i]
        spline_params[i, 1] = vector[i]
        spline_params[i, 2] = 3*(f_grid[i + 1] - f_grid[i]) - 2*vector[i] - vector[i + 1]
        spline_params[i, 3] = 2*(f_grid[i] - f_grid[i + 1]) + vector[i] + vector[i + 1]

    spline_params[data_cnt - 1, 0] = f_grid[data_cnt-1]
    spline_params[data_cnt - 1, 1] = vector[data_cnt-1]

    return spline_params

def read_control(control_path, context):
    if not os.path.isfile(control_path):
        return

    species_list = []
    system_charge = 0
    fixed_spin_moment = 0
    spin_type = "none"
    species_cnt = 0

    with open(control_path) as cfile:
        lines = cfile.readlines()

        species_next = True
        curr_species = ""

        for line in lines:
            data = line.split()
            if len(data) < 2 or data[0][0] == "#":
                continue
            if data[0] == "species":
                species_next = False
                curr_species = data[1]
            elif data[0] == "nucleus" and not species_next:
                charge = float(data[1])
                species_list.append(Species(species_cnt, \
                                    curr_species, charge))
                curr_species = ""
                species_cnt += 1
                species_next = True
            elif data[0] == "charge": # Total system charge, not per species
                system_charge = float(data[1])
            elif data[0] == "fixed_spin_moment":
                fixed_spin_moment = int(data[1]) # = 2S = N_up - N_down
            elif data[0] == "spin":
                spin_type = data[1] # collinear or none
            elif data[0] == "force_occupation_projector":
                context.has_core_hole = True

                spin = int(data[2])
                # aims can dynamically change the orbital index, so we can't just take it from here

    # aims also allows to set the spin in a different way, so inform user
    # to add redundant information if necessary
    if spin_type == "collinear" and fixed_spin_moment == 0:
        raise Exception("Spin type is collinear, but no fixed_spin_moment \
                        is set in control.in!")

    context.species = species_list
    context.system_charge = system_charge
    context.spin_moment = fixed_spin_moment

def find_species(species_list, symbol):
    for species in species_list:
        if species.symbol == symbol:
            return species

    return None

# Returns a list of atoms and their summed core charges
def import_geometry(trexfile, geom_path, context):
    if not os.path.isfile(geom_path):
        return

    species_list = context.species

    # geometry.in has been found; try to load it and store data directly
    # trexfile

    charges = []
    labels = []
    coords = []

    atoms = []

    with open(geom_path) as gfile:
        lines = gfile.readlines()
        # Species need to be sorted by order in geometry.in
        geom_index = 0

        for line in lines:
            data = line.split()
            if len(data) < 1:
                continue
            if data[0] == "atom" or data[0] == "empty":
                at_coords = [float(data[1]) / bohr_to_angstrom,
                             float(data[2]) / bohr_to_angstrom,
                             float(data[3]) / bohr_to_angstrom]
                coords.append(at_coords)
                species_label = data[4]
                species = find_species(species_list, species_label)
                if species == None:
                    raise Exception("Unknown species found in geometry.in: " + species)
                if species.geom_index == 10000:
                    species.geom_index = geom_index
                    geom_index += 1
                charge = 1.0*species.charge
                if data[0] == "empty":
                    species_label = "Empty"
                    charge = 0.0

                labels.append(species_label)
                charges.append(charge)
                atoms.append(Atom(len(atoms), species, at_coords))
            else:
                continue

            #print("Atom: ", coords[-1], charges[-1])

    nucleus_num = len(charges)

    # Calculate nuclear repulsion energy here
    e_nuc = 0.0
    for n0 in range(len(atoms)):
        a0 = atoms[n0]
        for n1 in range(n0 + 1, len(atoms)):
            a1 = atoms[n1]
            dist = np.sqrt((a0.coords[0] - a1.coords[0])**2
                         + (a0.coords[1] - a1.coords[1])**2
                         + (a0.coords[2] - a1.coords[2])**2)
            e_nuc += a0.species.charge * a1.species.charge / dist

    trexio.write_nucleus_num(trexfile, nucleus_num)
    trexio.write_nucleus_charge(trexfile, charges)
    trexio.write_nucleus_label(trexfile, labels)
    trexio.write_nucleus_coord(trexfile, coords)
    trexio.write_nucleus_repulsion(trexfile, e_nuc)

    nuclear_charge = 0
    for c in charges:
        nuclear_charge += c

    context.atoms = atoms
    context.nuclear_charge = nuclear_charge

def load_basis_set(trexfile, dirpath, context):
    atoms = context.atoms
    species_list = context.species
    aos = []
    shells = [] # Shells are defined by atom, n, l

    # Existence of file has been verified outside
    with open(dirpath + "/basis-indices.out") as ind_file:
        lines = ind_file.readlines()[2:] # Two header lines
        # To have the correct shell ids, start with lowest-index species
        for species in species_list:
            sp_lines = []
            shell_entries = 0 # How many m-entries are left for current shell
            for line in lines:
                # File is 1-indexed
                data = line.split()
                index = int(data[0]) - 1
                fn_type = data[1]
                atom_id = int(data[2]) - 1
                n = int(data[3])
                l = int(data[4])
                m = int(data[5])

                if atoms[atom_id].species != species:
                    continue

                # For n-Zeta sets, basis_indices lists equivalent data
                if shell_entries == 0:
                    # New shell has been started
                    shells.append(OrbitalShell(
                        len(shells), atom_id, fn_type, n, l
                    ))
                    shell_entries = 2*abs(m) + 1 # Always starts with -l

                normalization = 1
                aos.append(AtomicOrbital(index, shells[-1], m, normalization))
                shell_entries -= 1

    # aos need to be brought in the correct order (0, -1, 1 etc)
    aos.sort()

    # List of aos is done, shells can be loaded
    nao_grid_r = []
    nao_grid_phi = []
    nao_grid_grad = []
    nao_grid_lap = []
    nao_grid_start = [] # len = number of radial functions

    species_atoms = {}
    radial_cnt = 0

    shell_in_atom = 0
    last_atom = -1

    for shell in shells:
        # Need to get species id of atom
        curr_atom = atoms[shell.atom_id]
        species = curr_atom.species
        species_id = species.id_num

        if curr_atom != last_atom:
            shell_in_atom = -1
            last_atom = curr_atom

        shell_in_atom += 1

        # Only load this data once per species
        if species_id in species_atoms:
            if curr_atom != species_atoms[species_id]:
                # Still need to set nao_grid_id and size
                shell.nao_grid_id = species_list[species_id].nao_grid_ids[shell_in_atom]
                shell.nao_grid_size = species_list[species_id].nao_grid_sizes[shell_in_atom]
                continue
        else:
            species_atoms[species_id] = curr_atom

        filename = f"{shell.fn_type}_{species_id+1}_{radial_cnt+1:0>2}_" \
            f"{shell.n:0>2}_{angular_letter(shell.l)}.dat"
        
        # Load the actual data file
        if not os.path.isfile(dirpath + "/" + filename):
            raise Exception("Expected to find NAO data file " + filename \
                            + " but did not. This can be fixed by making sure that the species appear in control.in in the order they are in geometry.in")

        with open(dirpath + "/" + filename) as dfile:
            # First line contains number of data points == len(lines)
            lines = dfile.readlines()[1:]
            start = len(nao_grid_r)
            nao_grid_id = len(nao_grid_start)
            nao_grid_start.append(start)
            species_list[species_id].add_nao_grid(nao_grid_id, len(lines))
            shell.nao_grid_id = nao_grid_id
            shell.nao_grid_size = len(lines)
            for line in lines:
                data = line.split()
                r = float(data[0])
                phi = float(data[1])
                nao_grid_r.append(r)
                nao_grid_phi.append(phi)

        # Repeat for derivative
        filename = "drv_" + filename
        radial_cnt += 1
        
        if not os.path.isfile(dirpath + "/" + filename):
            raise Exception("Expected to find NAO derivative file " + filename \
                            + " but did not. Since the printing of the " \
                            "derivative is an addition within this project, " \
                            "please check your FHIaims version.")

        with open(dirpath + "/" + filename) as dfile:
            # First line contains number of data points == len(lines)
            lines = dfile.readlines()[1:]
            for line in lines:
                i = len(nao_grid_grad)
                data = line.split()
                deriv = float(data[1])
                nao_grid_grad.append(deriv)

        # Repeat for second derivative data
        filename = "kin" + filename[3:]

        # Load the actual data file
        if not os.path.isfile(dirpath + "/" + filename):
            raise Exception("Expected to find NAO data file " + filename \
                            + " but did not.")

        with open(dirpath + "/" + filename) as dfile:
            # First line contains number of data points == len(lines)
            lines = dfile.readlines()[1:]
            for line in lines:
                i = len(nao_grid_lap)
                data = line.split()
                kin = float(data[1])
                # The FHIaims kinetic spline is 
                # kin = -0.5 d**2/dr**2 + 0.5 l*(l+1) u/r**2
                r = nao_grid_r[i]
                deriv2 = shell.l * (shell.l + 1) * nao_grid_phi[i] / r**2 - 2*kin
                nao_grid_lap.append(deriv2)

    nao_grid_r = np.array(nao_grid_r)
    nao_grid_phi = np.array(nao_grid_phi)

    # Printed derivative is d u(r) / dr
    # It would appear logical to directly store d( u(r)/r ) / dr, 
    # which can be calculated via the chain rule, but storing d u(r) / dr
    # allows to calculate the gradients and Laplacian directly via the chain rule, so
    # this is more useful
    #if len(nao_grid_grad) != 0:
    #    nao_grid_grad = np.array(nao_grid_grad)
    #    if len(nao_grid_lap) != 0:
    #        nao_grid_lap = np.array(nao_grid_lap)
    #        nao_grid_lap = nao_grid_lap - 2*nao_grid_grad/nao_grid_r + 2*nao_grid_phi / nao_grid_r / nao_grid_r

    #    nao_grid_grad = nao_grid_grad - nao_grid_phi / nao_grid_r

    interp = np.zeros((0, 4), dtype=float)
    grad_interp = np.zeros((0, 4), dtype=float)
    lap_interp = np.zeros((0, 4), dtype=float)

    # Compute interpolation coefficients
    buffer_zero = np.zeros((1), dtype=float)
    for radial_at in range(radial_cnt):
        if radial_at < radial_cnt - 1:
            i0 = nao_grid_start[radial_at]
            i1 = nao_grid_start[radial_at + 1]
            r_sub = nao_grid_r[i0:i1]
            phi_sub = nao_grid_phi[i0:i1]
            grad_sub = nao_grid_grad[i0:i1]
            lap_sub = nao_grid_lap[i0:i1]
        else:
            i0 = nao_grid_start[radial_at]
            r_sub = nao_grid_r[i0:]
            phi_sub = nao_grid_phi[i0:]
            grad_sub = nao_grid_grad[i0:]
            lap_sub = nao_grid_lap[i0:]
        spline_params = create_cubic_spline(r_sub, phi_sub)
        grad_spline_params = create_cubic_spline(r_sub, grad_sub)
        lap_spline_params = create_cubic_spline(r_sub, lap_sub)
        interp = np.concatenate((interp, spline_params)) 
        grad_interp = np.concatenate((grad_interp, grad_spline_params))
        lap_interp = np.concatenate((lap_interp, lap_spline_params)) 

    # Scaffold starts are known for radials -> list for shells
    nucleus_index = [] # len = number of shells
    shell_start = []   # len = number of shells
    shell_size = []    # len = number of shells
    shell_ang_mom = [] # len = number of shells
    normalization = []

    shell_r_power = np.full(len(shells), -1, dtype=int)
    # Normalization factor for m=0
    l_factor = [1, 1, 2, 2, 8, 8, 16, 16, 128, 128, 256]
    l_factor = np.sqrt(np.array([
        (2*l + 1) / l_factor[l]**2 for l in range(11)
    ]) * (1/np.pi))/2

    m_factor = [
        [1],
        [1, 1, 1],
        [1, 2*3**0.5, 2*3**0.5, 3**0.5, 2*3**0.5],
        [1, 0.5*6**0.5, 0.5*6**0.5, 15**0.5, 2*15**0.5, 0.5*10**0.5, 0.5*10**0.5],
        [1, 2*10**0.5, 2*10**0.5, 2*5**0.5, 4*5**0.5, 2*70**0.5, 2*70**0.5, 35**0.5, 4*35**0.5]
        # Only implemented up to g
    ]

    # Factors between different values of m in a test-convenient form
    for shell in shells:
        shell_start.append(nao_grid_start[shell.nao_grid_id])
        shell_size.append(shell.nao_grid_size)
        nucleus_index.append(shell.atom_id)
        l = shell.l
        shell_ang_mom.append(shell.l)
        normalization.append(l_factor[l])


    ao_shells = []
    ao_norms = [] #np.ones(len(aos), dtype=float)

    # Convert to GAMESS convention

    for i in range(len(aos)):
        ao = aos[i]
        ao_shells.append(ao.shell.id_num)
        l = ao.shell.l
        m = ao.m

        n = 2 * np.abs(m) - (np.sign(m) + 1) // 2
        # Don't throw an error in case the user does not need them anyways
        if l < 5:
            ao_norms.append(m_factor[l][n])
        else:
            ao_norms.append(1)

    ao_norms = np.array(ao_norms) # Factor in normalization of radial functions

    # All data is accumulated, write to trexio file
    trexio.write_ao_cartesian(trexfile, 0)
    trexio.write_ao_num(trexfile, len(aos))
    trexio.write_ao_shell(trexfile, ao_shells)
    trexio.write_ao_normalization(trexfile, ao_norms)

    trexio.write_basis_type(trexfile, "Numerical")
    trexio.write_basis_shell_num(trexfile, len(shells))
    trexio.write_basis_nao_grid_num(trexfile, len(nao_grid_r))
    trexio.write_basis_interp_coeff_cnt(trexfile, 4)

    trexio.write_basis_nucleus_index(trexfile, nucleus_index)
    trexio.write_basis_shell_ang_mom(trexfile, shell_ang_mom)
    trexio.write_basis_r_power(trexfile, shell_r_power)
    trexio.write_basis_shell_factor(trexfile, normalization)

    trexio.write_basis_nao_grid_radius(trexfile, nao_grid_r)
    trexio.write_basis_nao_grid_phi(trexfile, nao_grid_phi)
    trexio.write_basis_nao_grid_grad(trexfile, nao_grid_grad)
    trexio.write_basis_nao_grid_lap(trexfile, nao_grid_lap)
    trexio.write_basis_nao_grid_start(trexfile, shell_start)
    trexio.write_basis_nao_grid_size(trexfile, shell_size)

    trexio.write_basis_interpolator_kind(trexfile, "Polynomial")
    trexio.write_basis_interpolator_phi(trexfile, interp)
    trexio.write_basis_interpolator_grad(trexfile, grad_interp)
    trexio.write_basis_interpolator_lap(trexfile, lap_interp)

    context.aos = aos

def get_occupation_and_class(trexfile, dirpath, context):
    # This is only tricky for core holes, but we cannot infer with
    # certainty which mo is unoccupied without the output file
    
    # Try to find the output file
    filename = context.aimsoutpath
    readable = os.path.isfile(filename)
    mo_num = context.mo_num
    ao_num = context.ao_num
    occupation = np.zeros(mo_num, dtype=int)
    if readable:
        with open(filename) as file:
            try:
                lines = file.readlines()
                header = "  Writing Kohn-Sham eigenvalues.\n"
                last_header = 0

                for iline, line in enumerate(lines):
                    if line == header:
                        last_header = iline

                if context.unrestricted():
                    i_up0 = last_header + 5
                    for iline in range(0, mo_num // 2):
                        line = lines[i_up0 + iline]
                        occ = int(float(line.split()[1]))
                        occupation[iline] = occ

                    i_dn0 = last_header + 9 + mo_num // 2
                    for iline in range(0, mo_num // 2):
                        line = lines[i_dn0 + iline]
                        occ = int(float(line.split()[1]))
                        occupation[mo_num // 2 + iline] = occ
                else:
                    i0 = last_header + 3
                    for iline in range(0, mo_num):
                        line = lines[i0 + iline]
                        occ = int(float(line.split()[1]))
                        occupation[iline] = occ
            except:
                # For large systems, not all virtual orbitals are printed
                pass

    else:
        # Don't print the warning if this is just a ground state
        if context.has_core_hole:
            print(f"File \"%s\" could not be found. It should be the FHI-aims output file and is needed to load the MO occupation. Please provide it or set the occupation manually in the trexio file.", filename)

        if context.spin_moment == 0:
            # RHF
            for i in range(context.elec_cnt // 2):
                occupation[i] = 2
        else:
            for i in range(context.elec_up):
                occupation[i] = 1
            for i in range(context.elec_dn):
                occupation[i + context.mo_num // 2] = 1

    trexio.write_mo_occupation(trexfile, occupation)

    # Since aims is an all-electron code, marking electrons as core for
    # a frozen core treatment makes little sense; the only purpose of
    # this part is thus to mark core holes as inactive

    orb_class = []

    spin_counted = 0 # Counted electrons for current spin
    # Standard is for RHF
    spin_max = context.elec_cnt # Number of electrons for current spin
    if context.unrestricted():
        spin_max = context.elec_up

    for iorb in range(mo_num):
        occ = occupation[iorb]
        if occ == 0:
            if spin_counted < spin_max:
                # This should be a core hole
                orb_class.append("inactive core")
            else:
                orb_class.append("active")
        else:
            orb_class.append("active")

            if occ != 0:
                spin_counted += occ

        if context.unrestricted() and iorb == ao_num - 1:
            # Change from up to down spin
            spin_counted = 0
            spin_max = context.elec_dn

    trexio.write_mo_class(trexfile, orb_class)

def handle_2e_integral(sparse_data, indices, val, g_mat=None, density=None, restricted=True, mo=True):
    sparse_data.add(indices, val)

    i = indices[0]
    j = indices[1]
    k = indices[2]
    l = indices[3]

    """
    What do we have in the polarized case?
    For aos:
        integrals are the same for both spins
        current implementation works
        indices only 0..n_basis
    
    For mos:
        indices from 0 to 2*n_basis = n_states
        g_mat should, in the end, be block diagonal because such is the Hamiltonian
        Input indices are in Dirac notation!
        integral (ij|kl): spin(i) == spin(k) and spin(j) == spin(l)
    """

    if not g_mat is None:
        if restricted:
            g_mat[i, k] += density[j, l] * val
            g_mat[i, l] -= 0.5 * density[k, j] * val
        else:
            # All matrices are block diagonal here
            # Both spins are handled at the same time
            o = density.shape[0] // 2
            if mo:
                # These index pairs are always of parallel spin
                g_mat[i, k] += density[j, l] * val
                if (i < o) == (l < o): # Check whether spins are the same
                    g_mat[i, l] -= density[k, j] * val
            else:
                # The core Hamiltonian only needs to be calculated for one spin
                g_mat[i, k] += (density[j, l] + density[j + o, l + o]) * val
                g_mat[i, l] -= density[k, j] * val
        # Mirroring of off-diagonal elements omitted due to lack of symmetry

def load_2e_integrals_ao(trexfile, dirpath, orb_cnt, orbital_indices,
                      orbital_signs, calculate_g=False, density=None,
                      restricted=True):
    # To reduce memory, it makes sense to calculate the G matrix on the fly

    # Handle both full and other type
    filename = dirpath + "/bielec_ao"
    symmetry = True
    if not os.path.isfile(filename):
        filename = dirpath + "/coulomb_integrals_ao.out"
        symmetry = False
        if not os.path.isfile(filename):
            print(f"No ao two-electron integral file could be found; "\
                   "if the integrals are needed, please check that one of the "\
                   "corresponding options is set in control.in.")
            return None

    g_mat = None # No need to do the allocation if unnecessary
    if density is None:
        calculate_g = False

    if calculate_g:
        orb_cnt = len(orbital_indices)
        g_mat = np.zeros((orb_cnt, orb_cnt), dtype=float)

    with open(filename) as file:
        sparse_data = SparseData(trexfile, trexio.write_ao_2e_int_eri, 1000)
        two_e_integral_threshold = 1e-10

        # Load integral, save to batch and apply to g matrix
        for line in file:
            data = line.split()
            i0 = int(data[0]) - 1
            j0 = int(data[1]) - 1
            k0 = int(data[2]) - 1
            l0 = int(data[3]) - 1
            i = orbital_indices[i0]
            j = orbital_indices[j0]
            k = orbital_indices[k0]
            l = orbital_indices[l0]
            indices = [i, j, k, l]
            val = float(data[4]) * orbital_signs[i0] \
                * orbital_signs[j0] * orbital_signs[k0] \
                * orbital_signs[l0]

            if np.fabs(val) < two_e_integral_threshold:
                continue

            handle_2e_integral(sparse_data, indices, val, g_mat, density, restricted, mo=False)

        # Flush the remaining integrals if there are any
        sparse_data.write_batch()

    return g_mat

# The routine is a bit simpler for mo since there is no shuffling,
# so it is probably worth to have this leaner routine
def load_2e_integrals_mo(trexfile, dirpath, orb_cnt, calculate_g=False, 
                         density=None, restricted=True):
    # To reduce memory, it makes sense to calculate the G matrix on the fly

    # Handle both full and other type
    filename = dirpath + "/bielec_mo"
    symmetry = True
    if not os.path.isfile(filename):
        filename = dirpath + "/coulomb_integrals_mo.out"
        symmetry = False
        if not os.path.isfile(filename):
            print(f"No mo two-electron integral file could be found; "\
                   "if the integrals are needed, please check that one of the "\
                   "corresponding options is set in control.in.")
            return None

    g_mat = None # No need to do the allocation if unnecessary
    if density is None:
        calculate_g = False

    if calculate_g:
        orb_cnt = density.shape[0]
        g_mat = np.zeros((orb_cnt, orb_cnt), dtype=float)

    with open(filename) as file:
        sparse_data = SparseData(trexfile, trexio.write_mo_2e_int_eri, 1000)
        two_e_integral_threshold = 1e-10

        # Load integral, save to batch and apply to g matrix
        for line in file:
            data = line.split()
            i = int(data[0]) - 1
            j = int(data[1]) - 1
            k = int(data[2]) - 1
            l = int(data[3]) - 1
            indices = [i, j, k, l]
            val = float(data[4])

            if np.fabs(val) < two_e_integral_threshold:
                continue

            handle_2e_integral(sparse_data, indices, val, g_mat, density, 
                               restricted, mo=True)

        # Flush the remaining integrals if there are any
        sparse_data.write_batch()

    return g_mat

def load_plaintext_matrix_ao(filename, context, info, shape):
    matrix_indices = context.matrix_indices
    matrix_signs = context.matrix_signs

    matrix = np.zeros(shape, dtype=float)

    try:
        with open(filename, "r") as file:
            lines = file.readlines()
            for line in lines:
                data = line.split()
                a = int(data[1]) - 1
                b = int(data[2]) - 1
                i = matrix_indices[a]
                j = matrix_indices[b]
                val = float(data[0]) * matrix_signs[a] * matrix_signs[b]
                matrix[i, j] = val
                matrix[j, i] = val
        return matrix
    except Exception as err:
        print(err)
        if len(info) > 1: # Can be used to mute warning
            print(f"Matrix file {filename} could not be found; if the {info[0]}" \
                + f" is needed, please check that \"{info[1]}\" is set in control.in.")
    return None

def load_plaintext_matrix_mo(filename, context, info, shape):
    matrix = np.zeros(shape, dtype=float)

    try:
        with open(filename, "r") as file:
            lines = file.readlines()
            for line in lines:
                data = line.split()
                i = int(data[1]) - 1
                j = int(data[2]) - 1
                val = float(data[0])
                matrix[i, j] = val
                matrix[j, i] = val
        return matrix
    except Exception as err:
        print(err)
        if len(info) > 1: # Can be used to mute warning
            print(f"Matrix file {filename} could not be found; if the {info[0]}" \
                + f" is needed, please check that \"{info[1]}\" is set in control.in.")
    return None


def load_elsi_matrix(filename, context, info=[], fix_cols=False):
    matrix_indices = context.matrix_indices
    matrix_signs = context.matrix_signs
    try:
        with open(filename, "rb") as file:
            data = file.read()

        # Fetch header
        start = 0
        end = 128
        header = struct.unpack("l"*16, data[start:end])
        n_basis = header[3]
        val_cnt = header[5] # Number of non-zero elements
        if header[2] != 0:
            raise Exception("The matrix containts complex values which are not supported by this script")

        start = end
        end = start + 8*n_basis
        pointer = struct.unpack("l"*n_basis, data[start:end])
        pointer += (val_cnt+1,)
        pointer = np.array(pointer) - 1

        start = end
        end = start + 4*val_cnt
        rows = struct.unpack("i"*val_cnt, data[start:end])
        rows = np.array(rows) - 1

        start = end

        end = start+val_cnt*8
        vals = struct.unpack("d" * val_cnt, data[start:end])

        matrix = np.zeros((n_basis, n_basis), dtype=float)
        val_at = 0
        for col in range(n_basis):
            for row in rows[pointer[col]:pointer[col+1]]:
                i = matrix_indices[row]
                sign = 1 # Cf matrix_signs in Context
                if fix_cols: # Needed so that coefficient matrix does not change MO order
                    j = col
                    sign = matrix_signs[row]
                else:
                    j = matrix_indices[col]
                    sign = matrix_signs[col] * matrix_signs[row]
                matrix[i, j] = vals[val_at] * sign
                val_at += 1

        return matrix
    except Exception as err:
        print(err)
        if len(info) > 1: # Not provided -> warning muted
            print(f"Matrix file {filename} could not be found; if the {info[0]}" \
                + f" is needed, please check that \"{info[1]}\" is set in control.in.")
    return None

def transform_to_mo(matrix, coeffs):
    cnt = matrix.shape[0]
    ret = np.zeros(matrix.shape, dtype=float)
    for i in range(cnt):
        for j in range(i, cnt):
            v = 0
            
            for a in range(cnt):
                coeff0 = coeffs[a, i]
                for b in range(cnt):
                    coeff1 = coeffs[b, j]
                    v += coeff0 * coeff1 * matrix[a, b]

            ret[i, j] = v
            ret[j, i] = v

    return ret

def load_integrals(trexfile, dirpath, context):
    matrix_indices = context.matrix_indices
    matrix_signs = context.matrix_signs
    restricted = context.spin_moment == 0
    # MO Coefficients
    # Columns need to be fixed so that the MO stay ordered by their energy
    # and the coefficient matrix in the MO basis diagonal
    coeffs = load_elsi_matrix(dirpath + "/C_spin_01_kpt_000001.csc", context, \
                               ["coefficient matrix", "elsi_output_matrix eigenvectors"], \
                              fix_cols=True)
    coeffs_up = coeffs
    coeffs_dn = None
    if not coeffs is None:
        ao_num = coeffs.shape[0]
        # Could have been set earlier, but only used by integral part
        context.ao_num = ao_num
        mo_num = ao_num

        # If basis_indices.out was not found, matrix_indices needs to be set
        if len(matrix_indices) == 0:
            matrix_indices = [i for i in range(coeffs.shape[0])]
            matrix_signs = [1 for i in range(coeffs.shape[0])]

        # If the system is spin polarized, join the two matrices
        if context.unrestricted() != 0:
            coeffs_dn = load_elsi_matrix(dirpath + "/C_spin_02_kpt_000001.csc", context, \
                                       ["coefficient matrix", "elsi_output_matrix eigenvectors"], \
                                      fix_cols=True)

            if not coeffs_dn is None:
                coeffs = np.hstack((coeffs, coeffs_dn))
                mo_num = 2*ao_num

        trexio.write_mo_num(trexfile, mo_num)
        trexio.write_mo_coefficient(trexfile, coeffs.T)

        context.mo_num = mo_num
    else:
        print("Coefficient matrix could not be found. Integrals cannot be loaded.")
        return

    get_occupation_and_class(trexfile, dirpath, context)
    occupation = context.mo_occupation

    # trexio does not store this matrix, but it is useful in constructing the Fock operator for testing
    density_up = load_elsi_matrix(dirpath + "/D_spin_01_kpt_000001.csc", context, [])
    if density_up is None:
        density_up = np.zeros(coeffs_up.shape, dtype=float)
        for mu in range(len(matrix_indices)):
            for nu in range(len(matrix_indices)):
                ret = 0

                # The counter should go up to the number of up-spin MOs,
                # which is here always the same as ao_num
                # This "overcomplicated" expression is necessary because\
                # of core holes
                for a in range(context.ao_num):
                    ret += coeffs_up[mu, a] * coeffs_up[nu, a] \
                        * occupation[a]
                density_up[mu, nu] = ret

    ao_density = density_up

    if context.unrestricted():
        density_dn = load_elsi_matrix(dirpath + "/D_spin_02_kpt_000001.csc", context, [])
        if density_dn is None and context.unrestricted():
            density_dn = np.zeros(coeffs_dn.shape, dtype=float)
            for mu in range(len(matrix_indices)):
                for nu in range(len(matrix_indices)):
                    ret = 0

                    # The counter should go up to the number of up-spin 
                    # MOs, which is here always the same as ao_num
                    for a in range(context.ao_num):
                        ret += coeffs_dn[mu, a] * coeffs_dn[nu, a] \
                            * occupation[a]
                    density_dn[mu, nu] = ret

        ao_density = block_diag(density_up, density_dn)

    mo_density = np.zeros((mo_num, mo_num), dtype=float)
    mo_density_up = np.zeros((ao_num, ao_num), dtype=float)
    mo_density_dn = np.zeros((ao_num, ao_num), dtype=float)

    if context.unrestricted():
        for i in range(context.elec_up):
            mo_density[i, i] = 1
            mo_density_up[i, i] = 1
        for i in range(context.elec_dn):
            mo_density[ao_num + i, ao_num + i] = 1
            mo_density_dn[i, i] = 1
        spin = [0 if i < ao_num else 1 for i in range(mo_num)]
        trexio.write_mo_spin(trexfile, spin)
    else:
        for i in range(context.elec_cnt // 2):
            mo_density[i, i] = 2

    # Overlap
    overlap = load_elsi_matrix(dirpath + "/S_spin_01_kpt_000001.csc", context, \
                               ["overlap matrix", "elsi_output_matrix overlap"])
    if not overlap is None:
        trexio.write_ao_1e_int_overlap(trexfile, overlap)
        mo_overlap = transform_to_mo(overlap, coeffs) #coeffs.transpose() * overlap * coeffs
        trexio.write_mo_1e_int_overlap(trexfile, mo_overlap)

    # Full hamiltonian; trexio supports only core Hamiltonian -> can be calculated with 2e integrals
    # After having modified aims to print the core Hamiltonian, this code
    # is obsolete but can still be used for verification
    #ao_ham_full = load_elsi_matrix(dirpath + "/H_spin_01_kpt_000001.csc", context, \
    #                           ["hamiltonian matrix", "elsi_output_matrix hamiltonian"])

    ao_core_ham = load_plaintext_matrix_ao(dirpath + "/core_hamiltonian_ao1.out", 
            context, ["ao core hamiltonian matrix", "output core_hamiltonian"], (ao_num, ao_num));

    if not ao_core_ham is None:
        trexio.write_ao_1e_int_core_hamiltonian(trexfile, ao_core_ham)

    # MO 2e integrals are directly loaded into the trexio file
    # G-matrix is only useful if the Hamiltonian has been found
    # Since the MOs are not affected by changes in the basis, there is no index transformation etc

    mo_core_ham_up = load_plaintext_matrix_mo(dirpath + "/core_hamiltonian_mo1.out", 
            context, ["mo core hamiltonian matrix", "output core_hamiltonian"], (coeffs_up.shape[1], coeffs_up.shape[1]));
    mo_core_ham = mo_core_ham_up

    if context.unrestricted():
        mo_core_ham_dn = load_plaintext_matrix_mo(dirpath + "/core_hamiltonian_mo2.out", 
                context, ["mo core hamiltonian matrix", "output core_hamiltonian"], (coeffs_dn.shape[1], coeffs_dn.shape[1]));

        if not mo_core_ham_dn is None and not mo_core_ham_up is None:
            mo_core_ham = block_diag(mo_core_ham_up, mo_core_ham_dn)

    if not mo_core_ham is None:
        trexio.write_mo_1e_int_core_hamiltonian(trexfile, mo_core_ham)

    # Obsolete, but interesting for testing
    #if not ao_ham is None:
    #    mo_core_ham_up2 = transform_to_mo(ao_core_ham_up, coeffs_up)
    #    mo_core_ham2 = mo_core_ham_up

    #    if context.unrestricted():
    #            mo_core_ham_dn2 = transform_to_mo(ao_ham, coeffs_dn)
    #            mo_core_ham2 = block_diag(mo_core_ham_up, mo_core_ham_dn)

    mo_g_matrix = load_2e_integrals_mo(trexfile, dirpath, mo_num,
                                    False and not mo_ham is None, mo_density,
                                    not context.unrestricted())

    # Testing only
    #if not mo_g_matrix is None and not mo_ham is None:
    #        mo_core_ham2 = mo_ham - mo_g_matrix

    #for i in range(mo_core_ham.shape[0]):
    #    for j in range(i, mo_core_ham.shape[1]):
    #        print(i, j, mo_core_ham[i, j], mo_core_ham_dir[i, j])

    # Note that the functionality for printing the ao 2e integrals is an
    # addition within this project and not necessarily available in
    # public versions of FHI-aims
    ao_g_matrix = load_2e_integrals_ao(trexfile, dirpath, ao_num,
                                    matrix_indices, matrix_signs, 
                                    False and not mo_ham is None,
                                    ao_density, not context.unrestricted())
    
    #if not ao_ham is None and not ao_g_matrix is None:
    #    ao_core_ham2 = ao_ham_up - ao_g_matrix

def convert_aims_trexio(trexfile, aimsoutpath):
    dirpath = os.path.dirname(os.path.abspath(aimsoutpath))
    if not os.path.isdir(dirpath):
        raise Exception("The provided path " + dirpath \
                        + " does not seem to be a directory.")

    trexio.write_metadata_code_num(trexfile, 1)
    trexio.write_metadata_code(trexfile, ["FHI-aims"])
    trexio.write_metadata_author_num(trexfile, 1)
    trexio.write_metadata_author(trexfile, [os.environ["USER"]])

    context = Context(dirpath, aimsoutpath)

    control_path = dirpath + "/control.in"
    read_control(control_path, context)
    if context.species == None:
        # Technically, integrals could still be exported, but it is simpler
        # to just require the file.
        msg = "The control file anticipated at " + control_path \
              + "could not be found. Export cannot be done without this file"
        raise Exception(msg)

    geom_path = dirpath + "/geometry.in"
    import_geometry(trexfile, geom_path, context)
    # Species list needs to be sorted by order of appearance in geometry.in
    # This sometimes fixes the not-finding of spline data
    #context.species.sort(key=lambda x: x.geom_index)
    if context.atoms == None:
        # geometry.in contains the number of electrons, so the export makes
        # no sense without it
        raise Exception("The geometry file anticipated at", geom_path, \
                        "could not be found. Export cannot be done without this file")

    # Populate the electron group
    context.elec_cnt = int(context.nuclear_charge - context.system_charge)
    elec_up = (context.elec_cnt + context.spin_moment) // 2
    if context.spin_moment == 0 and 2*elec_up != context.elec_cnt: # ROHF case
        elec_up = context.elec_cnt - elec_up # Ensure elec_up > elec_dn
        context.rohf = True
    context.elec_up = elec_up
    context.elec_dn = context.elec_cnt - context.elec_up

    trexio.write_electron_num(trexfile, context.elec_cnt)
    trexio.write_electron_up_num(trexfile, context.elec_up)
    trexio.write_electron_dn_num(trexfile, context.elec_dn)

    #print("Total / up electrons: ", elec_cnt, elec_cnt_up)

    # basis_indices.out
    indices_path = dirpath + "/basis-indices.out"
    basis_indices_found = os.path.isfile(indices_path)
    matrix_indices = [] # If basis is not known, matrices will be imported as-is
    matrix_signs = []
    if not basis_indices_found:
        print("The basis_indices.out file anticipated at", indices_path,
                        "could not be found.")
    else:
        load_basis_set(trexfile, dirpath, context)

        # From the aos, associate the indices of input matrices with those of output matrices
        matrix_indices = np.zeros(context.ao_count(), dtype=int)
        matrix_signs = np.ones(context.ao_count(), dtype=int)
        signs = [
            [1], [1, -1, 1], [1, -1, 1, 1, 1], [1, -1, 1, 1, 1, -1, 1], [1, -1, 1, 1, 1, -1, 1, 1, 1]
        ]
        for i, ao in enumerate(context.aos):
            matrix_indices[ao.id_num] = i
            # For everything higher, this isn't implemented. 
            # The signs can be found using check_basis.
            if ao.shell.l < len(signs):
                m = ao.m
                n = 2 * np.abs(m) - (np.sign(m) + 1) // 2
                matrix_signs[ao.id_num] = signs[ao.shell.l][n]
        #print(matrix_signs)


    context.matrix_indices = matrix_indices
    context.matrix_signs = matrix_signs

    load_integrals(trexfile, dirpath, context)

    trexfile.close()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise Exception("No path to the FHI-aims output data was provided.")

    path = sys.argv[1]
    outname = "tmp.h5"
    if len(sys.argv) > 2:
        outname = sys.argv[2]

    trexfile = trexio.File(outname, "w")

    convert_aims_trexio(trexfile, path)
    print("Execution finished")

