from embedding_module import run_closed_shell, run_open_shell

def driver(keywords):
    """
    This driver checks for the control keywords and calls the 
    appropriate embedding solver.

    List of keywords:
        package (str): name of the quantum chemical package.
            Defaults to Psi4, which is the only option at present.
        num_threads (int): number of threads. Defaults to 1.
        memory (str): allocated memory. Defaults to '1000 MB'.
        charge (int): charge. Defaults to 0.
        multiplicity (int): spin multiplicity. Defaults to 1.
        low_level_reference (str): can be RHF, ROHF (HF only), and 
            UHF. Defaults to RHF.
        high_level_reference (str): can be RHF, ROHF (HF only), and 
            UHF. Defaults to RHF.
        partition_method (str): Partition method for the occupied space.
            Defaults to SPADE.
        e_convergence (float): SCF energy convergence threshold. 
            Defaults to 1.0e-6.
        d_convergence (float): SCF density convergence threshold. 
            Defaults to 1.0e-6.
        eri (str): algorithm for computing electron repulsion
            integrals. Defaults to 'df' (density fitting).
        ints_tolerance (float): threshold below which ERI's are 
            neglected. Defaults to 1.0e-10.
        driver_output (str): output file for 'package'. 
            Defaults to 'output.dat'.
        embedding_output (str): output of the embedded calculation.
            Defaults to 'embedding.log'.
        operator (str): one-particle operator for CL shells. Can be
            F (Fock), K (exchange), V (electron-nuclei potential),
            H (core Hamiltonian), and K_orb (K-orbitals of Feller
            and Davidson). Defaults to F.
        level_shift (float): level shift parameter to enforce 
            orthogonalize between subsystems. Defaults to 1.0e6.
        low_level_damping_percentage (int): percentage of damping in
            the low level calculation. Defaults to 0.
        high_level_damping_percentage (int): percentage of damping in
            the high level calculation. Defaults to 0.
        low_level_soscf (str): second order convergence for low
            level SCF calculation. Defaults to False.
        high_level_soscf (str): second order convergence for high
            level SCF calculation. Defaults to False.
        molden (bool): create the following molden files:
            before_pseudocanonical - active, occupied SPADE orbitals 
            after_pseudocanonical - pseudocanonical SPADE orbitals
            embedded - occupied embedded orbitals.
            Numbered molden files correspond to CL shells labeled
            by the numbers.
            Defaults to False.
        print_level (int): amount of print in 'driver_output'.
            Defaults to 1.
        cc_type (str): algorithm for ERI MO transformation. 
            Defaults to 'df' (density-fitting).
        write_embedded_potential (bool): writes embedded potential
            to embedding_potential.txt in numpy format. 
            Defaults to False.
        write_embedded_h_core (bool): writes embedded core Hamiltonian
            to embedded_h_core.txt in numpy format.
            Defaults to False.
        write_embedded_orbitals (bool): writes embedded orbitals 
            to embedded_orbitals.txt in numpy format.
            Defaults to False.
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

