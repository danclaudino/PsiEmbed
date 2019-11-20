import os
import psi4
import numpy as np
import embedding_methods

def run_closed_shell(keywords):

    if keywords['package'].lower() == 'psi4':
        embed = embedding_methods.Psi4Embed(keywords)
    embed.banner()
    embed.run_psi4()
    low_level_orbitals = embed.occupied_orbitals

    # Whether or not to project occupied orbitals onto another (minimal) basis
    if 'occupied_projection_basis' in keywords:
        projection_orbitals = embed.basis_projection(low_level_orbitals,
            keywords['occupied_projection_basis'])
        ao_overlap = embed.projection_orbitals
        n_act_aos = embed.count_active_aos(
            keywords['occupied_projection_basis'])
    else:
        projection_orbitals = low_level_orbitals
        ao_overlap = embed.ao_overlap
        n_act_aos = embed.count_active_aos(keywords['basis'])

    # Orbital rotation and partition into subsystems A and B
    rotation_matrix, sigma = embed.orbital_rotation(projection_orbitals,
        n_act_aos, ao_overlap)
    n_act_mos, n_env_mos = embed.orbital_partition(sigma)

    # Defining active and environment orbitals and density
    act_orbitals = low_level_orbitals @ rotation_matrix.T[:, :n_act_mos]
    env_orbitals = low_level_orbitals @ rotation_matrix.T[:, n_act_mos:]
    act_density = act_orbitals @ act_orbitals.T
    env_density = env_orbitals @ env_orbitals.T

    # Retrieving the subsytem energy terms and potential matrices
    if keywords['package'].lower() == 'psi4':
        e_act, e_xc_act, j_act, k_act, v_xc_act = (
            embed.closed_shell_subsystem(act_orbitals))
        e_env, e_xc_env, j_env, k_env, v_xc_env = (
            embed.closed_shell_subsystem(env_orbitals))

    # Computing cross subsystem terms
    j_cross = 2.0*(embed.dot(act_density, j_env)
            + embed.dot(env_density, j_act))
    k_cross = -embed.alpha*(embed.dot(act_density, k_env)
            + embed.dot(env_density, k_act))
    xc_cross = embed.e_xc_total - e_xc_act - e_xc_env
    two_e_cross = j_cross + k_cross + xc_cross

    # Generating molden files before and after pseudocanonicalization
    if keywords['molden']:
        wfn_low_level.Ca().copy(psi4.core.Matrix.from_array(orbitals_act))
        psi4.driver.molden(wfn_low_level, 'before_pseudocanonical.molden')
        v, w = np.linalg.eigh(orbitals_act.T @ wfn_low_level.Fa().np
            @ orbitals_act)
        pseudo_orbitals_act = orbitals_act @ w
        wfn_low_level.Ca().copy(psi4.core.Matrix.from_array(
            pseudo_orbitals_act))
        psi4.driver.molden(wfn_low_level, 'after_pseudocanonical.molden')

    # Defining the embedded core Hamiltonian
    ao_overlap = embed.ao_overlap
    projector = keywords['level_shift']*(ao_overlap @ env_density @ ao_overlap)
    h_core = embed.h_core
    h_core_emb = (h_core + 2.0*j_env - embed.alpha*k_env
                + projector + embed.v_xc_total - v_xc_act)

    # Saving embedded core Hamiltonian to 'newH.dat',
    # which is read by my version of Psi4
    f = open('newH.dat', 'w')
    for i in range(h_core_emb.shape[0]):
        for j in range(h_core_emb.shape[1]):
            f.write("%s\n" % h_core_emb[i,j])
    f.close()

    # Running embedded calculation
    embed.run_psi4('hf')

    if keywords['molden']:
        psi4.driver.molden(wfn_hf, 'embedded.molden')

    # Overlap between the C_A determinant and the embedded determinant
    embed.determinant_overlap(act_orbitals)

    # Computing the embedded SCF energy
    density_emb = embed.occupied_orbitals @ embed.occupied_orbitals.T
    j_emb = embed.j
    k_emb = embed.k
    e_act_emb = embed.dot(density_emb, 2*h_core + 2*j_emb - k_emb)
    correction = 2.0*(embed.dot(h_core_emb - h_core,
        density_emb - act_density))
    e_mf_emb = e_act_emb + e_env + two_e_cross + embed.nre + correction
    embed.print_scf(e_act, e_env, two_e_cross, e_act_emb, correction)

    # --- Post embedded HF calculation ---
    if 'n_virtual_shell' not in keywords:
        e_correlation = embed.correlation_energy()
        e_total = e_mf_emb + e_correlation
        embed.outfile.write(' Embedded {:>5}-in-{:<5} E[A] \t = {:>16.10f}\n'.
            format(keywords['high_level'].upper(),
            keywords['low_level'].upper(), e_total))
    else:
        assert isinstance(keywords['n_virtual_shell'], int)
        embed.outfile.write('\n Singular values of '
            + str(keywords['n_virtual_shell'] + 1) + ' virtual shells\n')
        embed.outfile.write(' Shells constructed with the %s operator\n'.
            format(keywords['operator_name']))

        # First virtual shell
        n_act_aos = embed.count_active_aos(keywords['virtual_projection_basis'])
        effective_virtuals = embed.effective_virtuals()

        # Projecting the effective virtuals onto (maybe) another basis
        projected_orbitals = embed.basis_projection(effective_virtuals,
            keywords['virtual_projection_basis'])

        # Defining the first shell
        shell_overlap = (projected_orbitals.T
            @ embed.overlap_two_basis[:n_act_aos, :] @ effective_virtuals)
        rotation_matrix, sigma = embed.orbital_rotation(shell_overlap,
            shell_overlap.shape[1])
        shell_size = (sigma[:n_act_aos] >= 1.0e-15).sum()
        embed.shell_size = shell_size
        span_orbitals = effective_virtuals @ rotation_matrix.T[:,:shell_size]
        kernel_orbitals = effective_virtuals @ rotation_matrix.T[:,shell_size:]
        sigma = sigma[:shell_size]

        # Computing energy in pseudocanonical basis
        e_orbital_span, pseudo_span = embed.pseudocanonical(span_orbitals)
        e_orbital_kernel, pseudo_kernel = (
            embed.pseudocanonical(kernel_orbitals))
        e_correlation = embed.correlation_energy(pseudo_span,
            pseudo_kernel, e_orbital_span, e_orbital_kernel)

        # Printing shell results
        embed.print_sigma(sigma, 0)
        embed.outfile.write(' {}-in-{} energy of shell # {} with {} orbitals = {:^12.10f}\n'.format(keywords['high_level'].upper(),
            keywords['low_level'].upper(), 0, shell_size,
            e_mf_emb + e_correlation))
        if keywords['molden']:
            embed.molden(virtual_span, virtual_ker, 0, n_env_mos)

        # Checking the maximum number of shells
        max_shell, n_virtual_shell = embed.count_shells()

        # Choosing the operator to be used to grow the shells
        embed.ao_operator()
        operator = embed.operator
        ishell = 0

        # Iterative spanning of the virtual space via CL orbitals
        for ishell in range(1, n_virtual_shell + 1):
            mo_operator = span_orbitals.T @ operator @ kernel_orbitals
            rotation_matrix, sigma = embed.orbital_rotation(mo_operator,
                shell_size)
            sigma = sigma[:shell_size]

            new_shell = kernel_orbitals @ rotation_matrix.T[:, :shell_size]
            span_orbitals = np.hstack((span_orbitals, new_shell))
            kernel_orbitals = (kernel_orbitals
                @ rotation_matrix.T[:, shell_size:])

            if keywords['molden']:
                embed.molden(virtual_span, virtual_kernel, ishell, n_env_mos)
                embed.heatmap(virtual_span, virtual_kernel, ishell, operator)

            e_orbital_span, pseudo_span = embed.pseudocanonical(span_orbitals)
            e_orbital_kernel, pseudo_kernel = (
                embed.pseudocanonical(kernel_orbitals))
            e_correlation = embed.correlation_energy(pseudo_span,
                pseudo_kernel, e_orbital_span, e_orbital_kernel)

            embed.print_sigma(sigma, ishell)
            embed.outfile.write(
                ' {}-in-{} energy of shell # {} with {} orbitals = {:^12.10f}\n'
                .format(keywords['high_level'].upper(),
                keywords['low_level'].upper(), ishell, shell_size*(ishell + 1),
                e_mf_emb + e_correlation))

        if ishell == max_shell and keywords['n_virtual_shell'] > max_shell:
            virtual_span = np.hstack((span_orbitals, kernel_orbitals))
            e_orbital_span, pseudo_span = embed.pseudocanonical(virtual_span)
            e_correlation = embed.correlation_energy()
            embed.outfile.write(' Energy of all ({}) orbitals = {:^12.10f}\n'.
                format(virtual_span.shape[1], e_mf_emb + e_correlation))

        embed.print_summary(e_mf_emb)
        projected_env_correction = embed.dot(projector,
            act_density - density_emb)
        embed.outfile.write(' Correction from the projected B\t = {:>16.2e}\n'.
            format(projected_env_correction))
        embed.outfile.close()
    os.system('rm newH.dat')
    
def run_open_shell(keywords):

    if keywords['package'].lower() == 'psi4':
        embed = embedding_methods.Psi4Embed(keywords)
    embed.banner()
    embed.run_psi4()
    alpha_occupied_orbitals = embed.alpha_occupied_orbitals
    beta_occupied_orbitals = embed.beta_occupied_orbitals

    # Whether or not to project occupied orbitals onto another (minimal) basis
    if 'occupied_projection_basis' in keywords:
        alpha_projection_orbitals = embed.basis_projection(
            alpha,occupied_orbitals, keywords['occupied_projection_basis'])
        beta_projection_orbitals = embed.basis_projection(
            beta,occupied_orbitals, keywords['occupied_projection_basis'])
        ao_overlap = embed.ao_overlap
        n_act_aos = embed.count_active_aos(
            keywords['occupied_projection_basis'])
    else:
        alpha_projection_orbitals = alpha_occupied_orbitals
        beta_projection_orbitals = beta_occupied_orbitals
        ao_overlap = embed.ao_overlap
        n_act_aos = embed.count_active_aos(keywords['basis'])

    # Orbital rotation and partition into subsystems A and B
    alpha_rotation_matrix, alpha_sigma = embed.orbital_rotation(
        alpha_projection_orbitals, n_act_aos, ao_overlap)
    beta_rotation_matrix, beta_sigma = embed.orbital_rotation(
        beta_projection_orbitals, n_act_aos, ao_overlap)
    alpha_n_act_mos, alpha_n_env_mos, beta_n_act_mos, beta_n_env_mos = (
        embed.orbital_partition(alpha_sigma, beta_sigma))

    alpha_act_orbitals = (alpha_occupied_orbitals
        @ alpha_rotation_matrix.T[:, :alpha_n_act_mos])
    alpha_env_orbitals = (alpha_occupied_orbitals
        @ alpha_rotation_matrix.T[:, alpha_n_act_mos:])
    beta_act_orbitals = (beta_occupied_orbitals
        @ beta_rotation_matrix.T[:, :beta_n_act_mos])
    beta_env_orbitals = (beta_occupied_orbitals
        @ beta_rotation_matrix.T[:, beta_n_act_mos:])

    alpha_act_density = alpha_act_orbitals @ alpha_act_orbitals.T
    beta_act_density = beta_act_orbitals @ beta_act_orbitals.T
    alpha_env_density = alpha_env_orbitals @ alpha_env_orbitals.T
    beta_env_density = beta_env_orbitals @ beta_env_orbitals.T

    # Retrieving the subsytem energy terms and potential matrices
    if keywords['package'].lower() == 'psi4':
        (e_act, e_xc_act, alpha_j_act, beta_j_act, alpha_k_act,
            beta_k_act, alpha_v_xc_act, beta_v_xc_act) = (
            embed.open_shell_subsystem(alpha_act_orbitals, beta_act_orbitals))
        (e_env, e_xc_env, alpha_j_env, beta_j_env, alpha_k_env,
            beta_k_env, alpha_v_xc_env, beta_v_xc_env) = (
            embed.open_shell_subsystem(alpha_env_orbitals, beta_env_orbitals))

    # Computing cross subsystem terms
    j_cross = 0.5*(embed.dot(alpha_j_act, alpha_env_density)
            + embed.dot(alpha_j_act, beta_env_density)
            + embed.dot(beta_j_act, alpha_env_density)
            + embed.dot(beta_j_act, beta_env_density)
            + embed.dot(alpha_j_env, alpha_act_density)
            + embed.dot(alpha_j_env, beta_act_density)
            + embed.dot(beta_j_env, alpha_act_density)
            + embed.dot(beta_j_env, beta_act_density))
    k_cross = -0.5*embed.alpha*(embed.dot(alpha_k_act, alpha_env_density)
            + embed.dot(beta_k_act, beta_env_density)
            + embed.dot(alpha_k_env, alpha_act_density)
            + embed.dot(beta_k_env, beta_act_density))
    xc_cross = embed.e_xc_total - e_xc_act - e_xc_env
    two_e_cross = j_cross + k_cross + xc_cross

    # Generating molden files before and after pseudocanonicalization
    if keywords['molden']:
        wfn_low_level.Ca().copy(psi4.core.Matrix.from_array(orbitals_act))
        psi4.driver.molden(wfn_low_level, 'before_pseudocanonical.molden')
        v, w = np.linalg.eigh(orbitals_act.T @ wfn_low_level.Fa().np
            @ orbitals_act)
        pseudo_orbitals_act = orbitals_act @ w
        wfn_low_level.Ca().copy(psi4.core.Matrix.from_array(
            pseudo_orbitals_act))
        psi4.driver.molden(wfn_low_level, 'after_pseudocanonical.molden')

    # Defining the embedded core Hamiltonian
    ao_overlap = embed.ao_overlap
    alpha_projector = keywords['level_shift']*(ao_overlap @ alpha_env_density
        @ ao_overlap)
    beta_projector = keywords['level_shift']*(ao_overlap @ beta_env_density
        @ ao_overlap)
    alpha_v_emb = (alpha_j_env + beta_j_env - embed.alpha*alpha_k_env
                + alpha_projector + embed.alpha_v_xc_total - alpha_v_xc_act)
    beta_v_emb = (alpha_j_env + beta_j_env - embed.alpha*beta_k_env
                + beta_projector + embed.beta_v_xc_total - beta_v_xc_act)

    # Saving embedded core Hamiltonian to 'newH.dat',
    # which is read by my version of Psi4
    fa = open('Va_emb.dat', 'w')
    fb = open('Vb_emb.dat', 'w')
    for i in range(alpha_v_emb.shape[0]):
        for j in range(alpha_v_emb.shape[1]):
            fa.write("%s\n" % alpha_v_emb[i,j])
            fb.write("%s\n" % beta_v_emb[i,j])
    fa.close()
    fb.close()

    # Running embedded calculation
    embed.run_psi4('hf')
    if keywords['molden']:
        psi4.driver.molden(wfn_hf, 'embedded.molden')

    # Overlap between the C_A determinant and the embedded determinant
    embed.determinant_overlap(alpha_act_orbitals, beta_act_orbitals)

    # Computing the embedded SCF energy
    alpha_density_emb = (embed.alpha_occupied_orbitals
        @ embed.alpha_occupied_orbitals.T)
    beta_density_emb = (embed.beta_occupied_orbitals
        @ embed.beta_occupied_orbitals.T)
    alpha_j_emb = embed.alpha_j
    beta_j_emb = embed.beta_j
    alpha_k_emb = embed.alpha_k
    beta_k_emb = embed.beta_k
    h_core = embed.h_core
    e_act_emb = (embed.dot(alpha_density_emb + beta_density_emb, h_core)
                + 0.5*embed.dot(alpha_density_emb + beta_density_emb,
                alpha_j_emb + beta_j_emb)
                - 0.5*(embed.dot(alpha_density_emb, alpha_k_emb)
                + embed.dot(beta_density_emb, beta_k_emb)))
    correction = (embed.dot(alpha_v_emb, alpha_density_emb - alpha_act_density)
                + embed.dot(beta_v_emb, beta_density_emb - beta_act_density))
    e_mf_emb = e_act_emb + e_env + two_e_cross + embed.nre + correction
    embed.print_scf(e_act, e_env, two_e_cross, e_act_emb, correction)

    # --- Post embedded HF calculation ---
    if 'n_virtual_shell' not in keywords:
        e_correlation = embed.correlation_energy()
        e_total = e_mf_emb + e_correlation
        embed.outfile.write(' Embedded {:>5}-in-{:<5} E[A] \t = {:>16.10f}\n'.
            format(keywords['high_level'].upper(),
            keywords['low_level'].upper(), e_total))
    else:
        raise NotImplementedError('CL orbitals for open-shells coming soon!')
        assert isinstance(keywords['n_virtual_shell'], int)
        embed.outfile.write('\n Singular values of '
            + str(keywords['n_virtual_shell'] + 1) + ' virtual shells\n')
        embed.outfile.write(' Shells constructed with the %s operator\n'.format(
            keywords['operator_name']))

        # First virtual shell
        n_act_aos = embed.count_active_aos(keywords['virtual_projection_basis'])
        effective_virtuals = embed.effective_virtuals()

        # Projecting the effective virtuals onto (maybe) another basis
        projected_orbitals = embed.basis_projection(effective_virtuals,
            keywords['virtual_projection_basis'])

        # Defining the first shell
        shell_overlap = (projected_orbitals.T
            @ embed.overlap_two_basis[:n_act_aos, :] @ effective_virtuals)
        rotation_matrix, sigma = embed.orbital_rotation(shell_overlap,
            n_act_aos)
        shell_size = (sigma[:n_act_aos] >= 1.0e-15).sum()
        embed.shell_size = shell_size
        span_orbitals = effective_virtuals @ rotation_matrix.T[:,:shell_size]
        kernel_orbitals = effective_virtuals @ rotation_matrix.T[:,shell_size:]
        sigma = sigma[:shell_size]

        # Pseudocanonicalizing the 1st shell
        e_orbital_span, pseudo_span = embed.pseudocanonical(span_orbitals)
        e_orbital_kernel, pseudo_kernel = embed.pseudocanonical(kernel_orbitals)
        e_correlation = embed.correlation_energy(span_orbitals,
            kernel_orbitals, e_orbital_span, e_orbital_kernel)

        embed.print_sigma(sigma, 0)
        embed.outfile.write('{}-in-{} energy of shell # {} with {} orbitals '
            + '= {:^12.10f}\n'.format(keywords['high_level'].upper(), \
            keywords['low_level'].upper(), 0, shell_size,
            e_mf_emb + e_correlation))
        if keywords['molden']:
            embed.molden(vritual_span, virtual_ker, 0, n_env_mos)

        # Checking the maximum number of shells
        max_shell, n_virtual_shell = embed.count_shells()

        # Choosing the operator to be used to grow the shells
        embed.ao_operator()
        operator = embed.operator
           
        # Iterative spanning of the virtual space via CL orbitals
        for ishell in range(1, n_virtual_shell + 1):
            mo_operator = span_orbitals.T @ operator @ kernel_orbitals
            rotation_matrix, sigma = embed.orbital_rotation(mo_operator,
                shell_size)
            sigma = sigma[:shell_size]

            new_shell = kernel_orbitals @ rotation_matrix.T[:, :shell_size]
            span_orbitals = np.hstack((span_orbitals, new_shell))
            kernel_orbitals = kernel_orbitals @ rotation_matrix.T[:, shell_size:]

            if keywords['molden']:
                embed.molden(new_shell, ishell)
                embed.heatmap(virtual_span, virtual_kernel, ishell, operator)

            e_orbital_span, pseudo_span = embed.pseudocanonical(span_orbitals)
            e_orbital_kernel, pseudo_kernel = (
                embed.pseudocanonical(kernel_orbitals))
            e_correlation = embed.correlation_energy(span_orbitals,
                kernel_orbitals, e_orbital_span, e_orbital_kernel)

            embed.print_sigma(sigma, ishell)
            embed.outfile.write(' {}-in-{} energy of shell # {} with {} '
                + 'orbitals = {:^12.10f}\n'.format(keywords['high_level'].upper(),
                keywords['low_level'].upper(), ishell, shell_size*(ishell + 1),
                e_mf_emb + e_correlation))

        if ishell == max_shell and keywords['n_virtual_shell'] > max_shell:
            virtual_span = np.hstack((span_orbitals, kernel_orbitals))
            e_orbital_span, pseudo_span = embed.pseudocanonical(virtual_span)
            e_correlation = embed.correlation_energy()
            embed.outfile.write(' Energy of all ({}) orbitals = {:^12.10f}\n'.
                format(virtual_span.shape[1], e_mf_emb + e_correlation))

        embed.print_summary(e_mf_emb)
        projected_env_correction = embed.dot(projector,
            act_density - density_emb)
        embed.outfile.write(' Correction from the projected B\t = {:>16.2e}\n'.
            format(projected_env_correction))
        embed.outfile.close()
    os.system('rm Va_emb.dat Vb_emb.dat')
    
