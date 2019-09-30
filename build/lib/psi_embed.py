from embedding_solver import run_closed_shell, run_open_shell

def driver(keywords):
    """
    This driver checks for the control keywords and calls the 
    appropriate embedding solver.

    List of keywords:
        package (str): name of the quantum chemical package.
            Defaults to Psi4, which is the only option at present.
        cc_type (str): algorithm for ERI MO transformation. 
            Defaults to 'df' (density-fitting).
        partition_method (str): Partition method for the occupied space.
            Defaults to SPADE.

    """

    # Default keywords
    default_keywords = {}
    default_keywords['package'] = 'psi4'
    default_keywords['cc_type'] = 'df'
    default_keywords['partition_method'] = 'spade'
    default_keywords['low_level_damping_percentage'] = 0
    default_keywords['high_level_damping_percentage'] = 0
    default_keywords['low_level_soscf'] = 'False'
    default_keywords['high_level_soscf'] = 'False'
    default_keywords['ints_tolerance'] = 1e-10
    default_keywords['e_convergence'] = 1e-6
    default_keywords['d_convergence'] = 1e-6
    default_keywords['eri'] = 'df'
    default_keywords['charge'] = 0
    default_keywords['multiplicity'] = 1
    default_keywords['reference'] = 'rhf'
    default_keywords['num_threads'] = 1
    default_keywords['memory'] = '1000 MB'
    default_keywords['driver_output'] = 'output.dat'
    default_keywords['embedding_output'] = 'embedding.log'
    default_keywords['operator'] = 'F'
    default_keywords['level_shift'] = 1.0e6
    default_keywords['molden'] = False
    default_keywords['print_level'] = 1

    # Checking if the necessary keywords have been defined
    assert 'low_level' in keywords, ('\nChoose level of theory',
                                    'for the environment')
    assert 'high_level' in keywords, ('Choose level of theory',
                                    '\nfor the active region')
    assert 'basis' in keywords, '\nChoose a basis set'
    assert 'n_active_atoms' in keywords, ('\nProvide the number of active', 
        'atoms, which the first atoms in your coordinates string')

    for key in default_keywords.keys():
        if key not in keywords:
            keywords[key] = default_keywords[key]

    if ('n_virtual_shell' in keywords and 'virtual_projection_basis' not in keywords):
        keywords['virtual_projection_basis'] = keywords['basis']

    if keywords['reference'] == 'rhf':
        run_closed_shell(keywords)
    else:
        run_open_shell(keywords)

