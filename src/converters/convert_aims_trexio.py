
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

import sys
import os
import numpy as np
from scipy.linalg.lapack import dgtsv # For interpolation parameters
from scipy.linalg import block_diag

# This should be a module import, but the module is a utility of FHI-aims.
# To avoid license issues, it is currently not provided with trexio_tools.
# Hence, the ImportError would always shut it down and the import must be
# located elsewhere.
read_elsi = None

hartree_to_ev = 27.2113845
bohr_to_angstrom = 0.52917721

class Species:
    def __init__(self, id_num, symbol, charge):
        self.id_num = id_num
        self.symbol = symbol
        self.charge = charge
        self.numgrid_ids = []

    def add_numgrid_id(self, id_num):
        self.numgrid_ids.append(id_num)

    def __repr__(self):
        return f"Species(id={self.id_num}, symbol={self.symbol})"

class Atom:
    def __init__(self, id_num, species, coords):
        self.id_num = id_num
        self.species = species
        self.coords = coords

class OrbitalShell:
    # numgrid_id is set later on
    def __init__(self, id_num, atom_id, fn_type, n, l):
        self.id_num = id_num
        self.atom_id = atom_id
        self.fn_type = fn_type[:2].lower() # Needed to find the u(r) file
        self.n = n
        self.l = l
        self.numgrid_id = -1 # Dummy value

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
    def __init__(self, dirpath):
        self.dirpath = dirpath

        # Everything is initialized to None so availability can be checked
        self.species = None
        self.system_charge = None
        self.spin_moment = None

        self.atoms = None
        self.nuclear_charge = None

        self.aos = None
        self.elec_cnt = None
        self.elec_up = None
        self.elec_dn = None

        # Permutation from the FHI-aims orbital order to trexio
        self.matrix_indices = None

    def ao_count(self):
        return len(self.aos)

    # Occupation number per orbital
    def occupation(self):
        if spin_moment == 0:
            return 2
        return 1
    
    def unrestricted(self):
        return self.spin_moment != 0

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

    # aims also allows to set the spin in a different way, so inform user
    # to add redundant information if necessary
    if spin_type == "none" and fixed_spin_moment != 0:
        raise Exception("fixed_spin_moment is set, but spin is set to none!")
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
    #for ao in aos:
    #    print(ao.shell.atom_id, end="\t")
    #print()
    aos.sort()
    #or ao in aos:
    #   print(ao.shell.atom_id, end="\t")
    #rint()
    # And must at some point be converted to cartesian orbs

    # List of aos is done, shells can be loaded
    numgrid_r = []
    numgrid_phi = []
    numgrid_kin = []
    numgrid_start = [] # len = number of radial functions

    #for ao in aos:
    #    print(ao)
    #for shell in shells:
    #    print(shell)

    species_atoms = {}
    radial_cnt = 0

    shell_in_atom = 0
    last_atom = -1

    for shell in shells:
        # Need to get species id of atom
        curr_atom = atoms[shell.atom_id]
        species_id = curr_atom.species.id_num

        if curr_atom != last_atom:
            shell_in_atom = -1
            last_atom = curr_atom

        shell_in_atom += 1

        # Only load this data once per species
        if species_id in species_atoms:
            if curr_atom != species_atoms[species_id]:
                # Still need to set numgrid_id!
                shell.numgrid_id = species_list[species_id].numgrid_ids[shell_in_atom]
                continue
        else:
            species_atoms[species_id] = curr_atom

        # TODO dynamically set the number of zeros in file names
        filename = f"{shell.fn_type}_{species_id+1}_{radial_cnt+1:0>2}_" \
            f"{shell.n:0>2}_{angular_letter(shell.l)}.dat"
        
        #print(filename)

        # Load the actual data file
        if not os.path.isfile(dirpath + "/" + filename):
            raise Exception("Expected to find NAO data file " + filename \
                            + " but did not.")

        with open(dirpath + "/" + filename) as dfile:
            # First line contains number of data points == len(lines)
            lines = dfile.readlines()[1:]
            start = len(numgrid_r)
            numgrid_id = len(numgrid_start)
            numgrid_start.append(start)
            species_list[species_id].add_numgrid_id(numgrid_id)
            shell.numgrid_id = numgrid_id
            for line in lines:
                data = line.split()
                r = float(data[0])
                phi = float(data[1])
                numgrid_r.append(r)
                numgrid_phi.append(phi)

        # Repeat for kinetic energy spline
        filename = f"kin_{shell.fn_type}_{species_id+1}_{radial_cnt+1:0>2}_" \
            f"{shell.n:0>2}_{angular_letter(shell.l)}.dat"
        radial_cnt += 1
        
        #print(filename)

        # Load the actual data file
        if not os.path.isfile(dirpath + "/" + filename):
            raise Exception("Expected to find NAO data file " + filename \
                            + " but did not.")

        with open(dirpath + "/" + filename) as dfile:
            # First line contains number of data points == len(lines)
            lines = dfile.readlines()[1:]
            for line in lines:
                data = line.split()
                kin = float(data[1])
                numgrid_kin.append(kin)

    interp = np.zeros((0, 4), dtype=float)
    kin_interp = np.zeros((0, 4), dtype=float)

    # Compute interpolation coefficients
    #print(len(numgrid_phi))
    buffer_zero = np.zeros((1), dtype=float)
    for radial_at in range(radial_cnt):
        if radial_at < radial_cnt - 1:
            i0 = numgrid_start[radial_at]
            i1 = numgrid_start[radial_at + 1]
            r_sub = numgrid_r[i0:i1]
            phi_sub = numgrid_phi[i0:i1]
            kin_sub = numgrid_kin[i0:i1]
        else:
            i0 = numgrid_start[radial_at]
            r_sub = numgrid_r[i0:]
            phi_sub = numgrid_phi[i0:]
            kin_sub = numgrid_kin[i0:]
        spline_params = create_cubic_spline(r_sub, phi_sub)
        kin_spline_params = create_cubic_spline(r_sub, kin_sub)
        interp = np.concatenate((interp, spline_params)) 
        kin_interp = np.concatenate((kin_interp, kin_spline_params)) 

    # Scaffold starts are known for radials -> list for shells
    nucleus_index = [] # len = number of shells
    shell_start = []   # len = number of shells
    shell_size = []    # len = number of shells
    shell_ang_mom = [] # len = number of shells
    normalization = []

    shell_r_power = np.full(len(shells), -1, dtype=int)

    for shell in shells:
        shell_start.append(numgrid_start[shell.numgrid_id])
        nucleus_index.append(shell.atom_id)
        shell_ang_mom.append(shell.l)
        #normalization.append((np.pi*atoms[shell.atom_id].species.charge)**-0.5 / 2)
        normalization.append(np.pi**-0.5 / 2)


    for i in range(len(shells)):
        #print(shell_start[i])
        if shell.numgrid_id == len(numgrid_start) - 1:
            shell_size.append(len(numgrid_r) - shell_start[-1])
        else:
            shell_size.append(shell_start[i+1] - shell_start[i])

    #print(shell_size)

    #print(shell_start)
    #print(numgrid_start)

    #for i in range(len(numgrid_r) // 20):
    #    print(numgrid_r[i], numgrid_phi[i])

    ao_shells = []
    ao_norms = [] #np.ones(len(aos), dtype=float)

    for i in range(len(aos)):
        ao = aos[i]
        ao_shells.append(ao.shell.id_num)
        l = ao.shell.l
        m = ao.m

        # Only implemented up to g-orbitals since these norms are neither necessary for QMC nor FCIQMC
        if l < 5:
            l_facs = [1, 3, 15, 105, 35]
            l_fac = np.math.factorial(2*l+2)/np.math.factorial(l+1)/2**(l+1)
            l_fac = l_fac**0.5
            # For spherical harmonics the normalization factor also depends on m
            # Trexio-order: 0, 1, -1, 2, -2 ...
            m_fac = [[1], # s
                     [1, 1, 1], # p
                     [0.5*3**-0.5, 1, 1, 0.5, 1], # d
                     [2*15**0.5, 2*10**0.5, 2*10**0.5, 0.5, 1, 2*6**0.5, 2*6**0.5], # f
                     [3*35**0.5/280, 3*14**0.5/28, 3*14**0.5/28, 3*7**0.5/28, 
                      3*7**0.5/14, 3*2**0.5/4, 3*2**0.5/4, 3/8, 1.5]] # g
            n = 2 * np.abs(m) - (np.sign(m) + 1) // 2

            ao_norms.append(l_fac * m_fac[l][n])
        else:
            ao_norms.append(1)

    ao_norms = np.array(ao_norms) # Factor in normalization of radial functions
    #print(ao_norms**2)

    # All data is accumulated, write to trexio file
    trexio.write_ao_cartesian(trexfile, 0)
    trexio.write_ao_num(trexfile, len(aos))
    trexio.write_ao_shell(trexfile, ao_shells)
    trexio.write_ao_normalization(trexfile, ao_norms)

    trexio.write_basis_type(trexfile, "Numerical")
    trexio.write_basis_shell_num(trexfile, len(shells))
    trexio.write_basis_numgrid_num(trexfile, len(numgrid_r))
    trexio.write_basis_interp_coeff_cnt(trexfile, 4)

    trexio.write_basis_nucleus_index(trexfile, nucleus_index)
    trexio.write_basis_shell_ang_mom(trexfile, shell_ang_mom)
    trexio.write_basis_r_power(trexfile, shell_r_power)
    trexio.write_basis_shell_factor(trexfile, normalization)

    trexio.write_basis_numgrid_radius(trexfile, numgrid_r)
    trexio.write_basis_numgrid_phi(trexfile, numgrid_phi)
    trexio.write_basis_numgrid_start(trexfile, shell_start)
    trexio.write_basis_numgrid_size(trexfile, shell_size)

    trexio.write_basis_interpolator(trexfile, interp)
    trexio.write_basis_interpolator_kin(trexfile, kin_interp)

    context.aos = aos

def handle_2e_integral(sparse_data, indices, val, g_mat = None, density=None):
    sparse_data.add(indices, val)

    i = indices[0]
    j = indices[1]
    k = indices[2]
    l = indices[3]

    if not g_mat is None:
        g_mat[i, k] += density[j, l] * val
        g_mat[i, l] -= 0.5 * density[k, j] * val
        # Mirroring of off-diagonal elements omitted due to lack of symmetry
        #print(density[j, l])

def load_2e_integrals(trexfile, dirpath, suffix, write, orbital_indices, calculate_g = False, density=None):
    # To reduce memory, it makes sense to calculate the G matrix on the fly

    # Handle both full and other type
    filename = dirpath + "/bielec_" + suffix + ".out"
    symmetry = True
    if not os.path.isfile(filename):
        filename = dirpath + "/coulomb_integrals_" + suffix + ".out"
        symmetry = False
        if not os.path.isfile(filename):
            print(f"Two-electron integral file {filename} could not be found; "\
                   "if the integrals are needed, please check that one of the "\
                   "corresponding options is set in control.in.")
            return None

    g_mat = None # No need to do the allocation if unnecessary
    if density is None:
        calculate_g = False

    if calculate_g:
        g_mat = np.zeros(density.shape, dtype=float)

    with open(filename) as file:
        sparse_data = SparseData(trexfile, write, 1000)
        two_e_integral_threshold = 1e-10

        # Load integral, save to batch and apply to g matrix
        for line in file:
            data = line.split()
            i = orbital_indices[int(data[0]) - 1]
            j = orbital_indices[int(data[1]) - 1]
            k = orbital_indices[int(data[2]) - 1]
            l = orbital_indices[int(data[3]) - 1]
            indices = [i, j, k, l]
            val = float(data[4])
            if np.fabs(val) < two_e_integral_threshold:
                continue

            if not symmetry:
                handle_2e_integral(sparse_data, indices, val, g_mat, density)
            else:
                # Permute indices and handle individually
                # All orbitals are real, so the symmetry is eightfold
                # But filter out duplicates
                permutations = [
                    [0, 1, 2, 3], # i j k l
                    [1, 0, 3, 2], # j i l k
                    [2, 3, 0, 1], # k l i j
                    [3, 2, 1, 0], # l k j i
                    [2, 1, 0, 3], # k j i l
                    [3, 0, 1, 2], # l i j k
                    [0, 3, 2, 1], # i l k j
                    [1, 2, 3, 0]  # j k l i
                ]
                found_permutations = []
                for perm in permutations:
                    p_indices = [indices[perm[0]], indices[perm[1]], indices[perm[2]], indices[perm[3]]]
                    if p_indices in found_permutations:
                        continue
                    found_permutations.append(p_indices)
                    handle_2e_integral(sparse_data, p_indices, val, g_mat, density)

        # Flush the remaining integrals if there are any
        sparse_data.write_batch()

    return g_mat

def load_elsi_matrix(filename, matrix_indices, info=[], fix_cols=False):
    try:
        matrix = read_elsi.read_elsi_to_csc(filename)
        matrix = matrix.tocoo() # COO format better suited for the permutations
        for i in range(len(matrix.row)):
            if not fix_cols:
                matrix.col[i] = matrix_indices[matrix.col[i]]
            matrix.row[i] = matrix_indices[matrix.row[i]]
        return matrix.toarray()
    except Exception as err:
        print(err)
        if len(info) > 1: # Can be used to mute warning
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
    # MO Coefficients
    # Columns need to be fixed so that the MO stay ordered by their energy
    # and the coefficient matrix in the MO basis diagonal
    coeffs = load_elsi_matrix(dirpath + "/C_spin_01_kpt_000001.csc", matrix_indices, \
                               ["coefficient matrix", "elsi_output_matrix eigenvectors"], \
                              fix_cols=True)
    coeffs_up = coeffs
    ao_num = len(matrix_indices)
    mo_num = ao_num
    coeffs_dn = None
    if not coeffs is None:
        # If basis_indices.out was not found, matrix_indices needs to be set
        if len(matrix_indices) == 0:
            matrix_indices = [i for i in range(coeffs.shape[0])]

        # If the system is spin polarized, join the two matrices
        if context.spin_moment != 0:
            coeffs_dn = load_elsi_matrix(dirpath + "/C_spin_02_kpt_000001.csc", matrix_indices, \
                                       ["coefficient matrix", "elsi_output_matrix eigenvectors"], \
                                      fix_cols=True)

            if not coeffs_dn is None:
                coeffs = np.hstack((coeffs, coeffs_dn))
                mo_num = 2*ao_num

        trexio.write_mo_num(trexfile, coeffs.shape[1])
        trexio.write_mo_coefficient(trexfile, coeffs)
    else:
        print("Coefficient matrix could not be found. Integrals cannot be loaded.")
        return

    # 1 body density matrix - only useful to construct ao core Hamiltonian
    # Since trexio does not store the density matrix for ao
    # It is needed to construct the ao core Hamiltonian, but since that one is spin
    # independent it is sufficient to calculate it for one spin only
    # No warning necessary since it can also be constructed from the coeffcícient matrix
    density_up = load_elsi_matrix(dirpath + "/D_spin_01_kpt_000001.csc", matrix_indices, [])
    if density_up is None:
        density_up = np.zeros(coeffs_up.shape, dtype=float)
        for mu in range(len(matrix_indices)):
            for nu in range(len(matrix_indices)):
                ret = 0

                for a in range(context.elec_up):
                    ret += coeffs_up[mu, a] * coeffs_up[nu, a]
                density_up[mu, nu] = context.occupation()*ret

    mo_density = np.zeros((mo_num, mo_num), dtype=float)

    if context.unrestricted():
        for i in range(context.elec_up):
            mo_density[i, i] = 1
        for i in range(context.elec_dn):
            mo_density[ao_num + i, ao_num + i] = 1
        spin = [0 if i < ao_num else 1 for i in range(mo_num)]
        trexio.write_mo_spin(trexfile, spin)
    else:
        for i in range(context.elec_cnt // 2):
            mo_density[i, i] = 2

    occupation = [mo_density[i, i] for i in range(mo_num)]
    trexio.write_mo_occupation(trexfile, occupation)

    # Overlap
    overlap = load_elsi_matrix(dirpath + "/S_spin_01_kpt_000001.csc", matrix_indices, \
                               ["overlap matrix", "elsi_output_matrix overlap"])
    if not overlap is None:
        trexio.write_ao_1e_int_overlap(trexfile, overlap)
        overlap_mo = transform_to_mo(overlap, coeffs)#coeffs.transpose() * overlap * coeffs
        trexio.write_mo_1e_int_overlap(trexfile, overlap_mo)

    # Full hamiltonian; trexio supports only core Hamiltonian -> can be calculated with 2e integrals
    hamiltonian = load_elsi_matrix(dirpath + "/H_spin_01_kpt_000001.csc", matrix_indices, \
                               ["hamiltonian matrix", "elsi_output_matrix hamiltonian"])

    # MO 2e integrals are directly loaded into the trexio file
    # G-matrix is only useful if the Hamiltonian has been found
    # Since the MOs are not affected by changes in the basis, there is no index transformation
    mo_indices = [i for i in range(len(matrix_indices))]
    mo_g_matrix = load_2e_integrals(trexfile, dirpath, "mo", \
                                    trexio.write_mo_2e_int_eri, mo_indices, \
                                    not hamiltonian is None, mo_density)
    
    if not hamiltonian is None:
        hamiltonian_mo = transform_to_mo(hamiltonian, coeffs_up)

        if context.unrestricted():
            ham_ao_dn = load_elsi_matrix(dirpath + "/H_spin_02_kpt_000001.csc", matrix_indices, \
                               ["down spin hamiltonian matrix", "elsi_output_matrix hamiltonian"])
            ham_mo_dn = transform_to_mo(ham_ao_dn, coeffs_dn)
            if not ham_mo_dn is None:
                hamiltonian_mo = block_diag(hamiltonian_mo, ham_mo_dn)

        if not mo_g_matrix is None:
            core_ham_mo = hamiltonian_mo - mo_g_matrix
            #print_matrix(hamiltonian_mo, "ham")
            #print_matrix(core_ham_mo, "Core ham")
            #print_matrix(mo_g_matrix, "g")
            trexio.write_mo_1e_int_core_hamiltonian(trexfile, core_ham_mo)
            #print_matrix(core_ham_mo, "Default mo core Hamiltonian")

    # Note that the functionality for printing the ao 2e integrals is an
    # addition within this project and not necessarily available in
    # public versions of FHI-aims
    # Since it is only an intermediary, one spin is sufficient
    ao_g_matrix = load_2e_integrals(trexfile, dirpath, "ao", \
        trexio.write_ao_2e_int_eri, matrix_indices, True, density_up)
    
    if not hamiltonian is None and not ao_g_matrix is None:
        core_ham_ao = hamiltonian - ao_g_matrix
        trexio.write_ao_1e_int_core_hamiltonian(trexfile, core_ham_ao)

        if context.unrestricted():
            # The mo Hamiltonian is most easily accessed from its ao equivalent
            core_ham_mo_up = transform_to_mo(core_ham_ao, coeffs_up)
            core_ham_mo_dn = transform_to_mo(core_ham_ao, coeffs_dn)
            #print_matrix(core_ham_mo_up, "MO Hamiltonian up");
            #print_matrix(core_ham_mo_dn, "MO Hamiltonian down");
            #print_matrix(core_ham_mo_dn + core_ham_mo_up, "MO Hamiltonian full");
            core_ham_mo = block_diag(core_ham_mo_up, core_ham_mo_dn)
            #trexio.write_mo_1e_int_core_hamiltonian(trexfile, core_ham_mo)
            #print_matrix(core_ham_mo, "Correct MO core Hamiltonian")


def convert_aims_trexio(trexfile, dirpath):
    # The exception is triggered when the module is loaded; if located in
    # the main body, it would block other converters
    try:
        global read_elsi
        from . import read_elsi as read_elsi
    except ImportError:
        raise Exception("This programm relies on the file \"read_elsi.py\" of FHIaims. " \
                        "Please copy it from \"FHIaims/utilities/elsi_matrix/read_elsi.py\" " \
                        "to \"trexio_tools/src/converters/\" and " \
                        "reinstall using \"pip install .\"")

    #print(dir(read_elsi))

    if not os.path.isdir(dirpath):
        raise Exception("The provided path " + dirpath \
                        + " does not seem to be a directory.")

    trexio.write_metadata_code_num(trexfile, 1)
    trexio.write_metadata_code(trexfile, ["FHI-aims"])
    trexio.write_metadata_author_num(trexfile, 1)
    trexio.write_metadata_author(trexfile, [os.environ["USER"]])

    context = Context(dirpath)

    control_path = dirpath + "/control.in"
    read_control(control_path, context)
    if context.species == None:
        # Technically, integrals could still be exported, but it is simpler
        # to just work with the file.
        msg = "The control file anticipated at " + control_path \
              + "could not be found. Export cannot be done without this file"
        raise Exception(msg)

    geom_path = dirpath + "/geometry.in"
    import_geometry(trexfile, geom_path, context)
    if context.atoms == None:
        # geometry.in contains the number of electrons, so the export makes
        # no sense without it
        raise Exception("The geometry file anticipated at", geom_path, \
                        "could not be found. Export cannot be done without this file")

    # Populate the electron group
    context.elec_cnt = int(context.nuclear_charge - context.system_charge)
    # aims always has N_up >= N_down
    context.elec_up = (context.elec_cnt + context.spin_moment) // 2
    context.elec_dn = context.elec_cnt - context.elec_up
    
    trexio.write_electron_num(trexfile, context.elec_cnt)
    trexio.write_electron_up_num(trexfile, context.elec_up)
    trexio.write_electron_dn_num(trexfile, context.elec_dn)

    #print("Total / up electrons: ", elec_cnt, elec_cnt_up)

    # basis_indices.out
    indices_path = dirpath + "/basis-indices.out"
    basis_indices_found = os.path.isfile(indices_path)
    matrix_indices = [] # If basis is not known, matrices will be imported as-is
    if not basis_indices_found:
        print("The basis_indices.out file anticipated at", indices_path,
                        "could not be found.")
    else:
        load_basis_set(trexfile, dirpath, context)

        # From the aos, associate the indices of input matrices with those of output matrices
        matrix_indices = np.zeros(context.ao_count(), dtype=int)
        for i, ao in enumerate(context.aos):
            matrix_indices[ao.id_num] = i

    context.matrix_indices = matrix_indices

    load_integrals(trexfile, dirpath, context)

    trexfile.close()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise Exception("No path to the FHI-aims output data was provided.")

    path = sys.argv[1]
    outname = ""
    if len(sys.argv) > 2:
        outname = sys.argv[2]

    convert_aims_trexio(path, outname)
    print("Execution finished")

