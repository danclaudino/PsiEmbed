from embedding_module import run_closed_shell, run_open_shell

def driver(keywords):
    """
    Checks for the control keywords and calls the 
    appropriate embedding routine.
    keywords is of dict type and takes the following keys:

    Parameters
    ----------
    package : str (Psi4)
        Name of the quantum chemical package.
        Psi4 is the only option at present.
    num_threads : int (1)
        Number of threads.
    memory : str ('1000 MB')
        Allocated memory.
    charge : int (0)
        Molecular charge.
    multiplicity : int (1)
        Spin multiplicity.
    low_level_reference : str ('rhf')
        Can be RHF, ROHF (HF only), or UHF.
    high_level_reference : str ('rhf')
        Can be RHF, ROHF (HF only), or UHF.
    partition_method : str ('spade')
        Partition method for the occupied space.
    e_convergence : float (1.0e-6)
        SCF energy convergence threshold. 
    d_convergence : float (1.0e-6)
        SCF density convergence threshold. 
    eri : str ('df')
        Algorithm for computing electron repulsion integrals.
    ints_tolerance : float (1.0e-10)
        Threshold below which ERI's are neglected.
    driver_output : str ('output.dat')
        Output file for 'package'. 
    embedding_output : str
        Output of the embedded calculation.
    operator : str ('F')
        One-particle operator for CL shells. 
        Can be F (Fock), K (exchange), V (electron-nuclei potential),
        H (core Hamiltonian), and K_orb (K-orbitals).
    level_shift : float (1.0e6)
        Level shift parameter to orthogonalize subsystems.
    low_level_damping_percentage : int (0) 
        Percentage of damping in the low level calculation.
    high_level_damping_percentage : int (0)
        Percentage of damping in the high level calculation.
    low_level_soscf : str ('false')
        Second order convergence for low level SCF calculation.
    high_level_soscf : str ('false')
        Second order convergence for high level SCF calculation.
    molden : bool (False)
        Create the following molden files:
        before_pseudocanonical - active, occupied SPADE orbitals.|
        after_pseudocanonical - pseudocanonical SPADE orbitals.|
        embedded - occupied embedded orbitals.|
        Numbered molden files correspond to CL shells labeled
        by the numbers.
    print_level : int (1)
        Amount of printing in output.
    cc_type : str ('df')
        Algorithm for ERI MO transformation for coupled cluster
        calculations.
    write_embedded_potential : bool (False)
        Writes embedded potential to embedding_potential.txt in numpy format. 
    write_embedded_h_core : bool (False)
        Writes embedded core Hamiltonian to embedded_h_core.txt in numpy format.
    write_embedded_orbitals : bool (False)
        Writes embedded orbitals to embedded_orbitals.txt in numpy format.
    """

    # Default keywords
    default_keywords = {}
    default_keywords['package'] = 'psi4'
    default_keywords['num_threads'] = 1
    default_keywords['memory'] = '1000 MB'
    default_keywords['charge'] = 0
    default_keywords['multiplicity'] = 1
    default_keywords['low_level_reference'] = 'rhf'
    default_keywords['high_level_reference'] = 'rhf'
    default_keywords['partition_method'] = 'spade'
    default_keywords['e_convergence'] = 1e-6
    default_keywords['d_convergence'] = 1e-6
    default_keywords['eri'] = 'df'
    default_keywords['ints_tolerance'] = 1e-10
    default_keywords['driver_output'] = 'output.dat'
    default_keywords['embedding_output'] = 'embedding.log'
    default_keywords['operator'] = 'F'
    default_keywords['level_shift'] = 1.0e6
    default_keywords['low_level_damping_percentage'] = 0
    default_keywords['high_level_damping_percentage'] = 0
    default_keywords['low_level_soscf'] = 'False'
    default_keywords['high_level_soscf'] = 'False'
    default_keywords['molden'] = False
    default_keywords['print_level'] = 1
    default_keywords['cc_type'] = 'df'
    default_keywords['write_embedding_potential'] = False
    default_keywords['write_embedded_h_core'] = False
    default_keywords['write_embedded_orbitals'] = False

    # Checking if the necessary keywords have been defined
    assert 'low_level' in keywords, ('\n Choose level of theory',
                                    'for the environment')
    assert 'high_level' in keywords, ('\n Choose level of theory',
                                    'for the active region')
    assert 'basis' in keywords, '\n Choose a basis set'
    assert 'n_active_atoms' in keywords, ('\n Provide the number of active', 
        'atoms, which the first atoms in your coordinates string')

    for key in default_keywords.keys():
        if key not in keywords:
            keywords[key] = default_keywords[key]

    if ('n_virtual_shell' in keywords and 
        'virtual_projection_basis' not in keywords):
        keywords['virtual_projection_basis'] = keywords['basis']

    if (keywords['low_level_reference'] == 'rhf' and 
            keywords['high_level_reference'] == 'rhf'):
        run_closed_shell(keywords)
    elif (keywords['low_level_reference'] == 'uhf' and 
            keywords['high_level_reference'] == 'uhf' or
            keywords['low_level_reference'] == 'uhf' and 
            keywords['high_level_reference'] == 'rohf'):
        run_open_shell(keywords)
    else:
        raise Exception(' The allowed combinations of' 
            + 'low/high_level_reference keywords are: RHF/RHF, UHF/UHF, '
            + 'and UHF/ROHF.')

