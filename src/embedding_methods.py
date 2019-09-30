import numpy as np
import psi4
import scipy.linalg
#import os

class Embed:
    """ Class with package-independent embedding methods."""

    def __init__(self, keywords):
        """Initialize the Embed class.

        Args:
            keywords (dict): dictionary with embedding options.
            mol (Psi4 molecule object): the molecule object.
        """
        self.keywords = keywords
        self.correlation_energy_shell = []
        self.shell_size = 0
        self.outfile = open(self.keywords['embedding_output'], 'w')
        return None

    @staticmethod
    def dot(A, B):
        """ Computes the trace (dot product) of matrices A and B

        Args:
            A, B (numpy.array): matrices to compute tr(A*B)

        Returns:
            The trace (dot product) of A * B
        """
        return np.einsum('ij, ij', A, B)

    def orbital_rotation(self, orbitals, n_active_aos, ao_overlap = None):
        """SVD orbitals projected onto active AOs to rotate orbitals.

        If ao_overlap is not provided, C is assumed to be in an
            orthogonal basis.
        
        Args:
            orbitals (numpy.array): MO coefficient matrix.
            n_active_aos (int): number of atomic orbitals in the
                active atoms.
            ao_overlap (numpy.array): AO overlap matrix.

        Returns:
            rotation_matrix (numpy.array): matrix to rotate orbitals
                right singular vectors of projected orbitals.
            singular_values (numpy.array): singular vectors of
                projected orbitals.
        """
        if ao_overlap is None:
            orthogonal_orbitals = orbitals[:n_active_aos, :]
        else:
            s_half = scipy.linalg.fractional_matrix_power(ao_overlap, 0.5)
            orthogonal_orbitals = (s_half @ orbitals)[:n_active_aos, :]

        u, s, v = np.linalg.svd(orthogonal_orbitals, full_matrices=True)
        rotation_matrix = v
        singular_values = s
        return rotation_matrix, singular_values

    def orbital_partition(self, sigma, beta_sigma = None):
        """
        Partition the orbital space by SPADE or all AOs in the
        projection basis. Beta variables are only used for open shells.

        Args:
            sigma (numpy.array): (alpha) singular values.
            beta_sigma (numpy.array): beta singular values.

        Returns:
            self.n_act_mos (int) = (alpha) number of active MOs.
            self.n_env_mos (int) = (alpha) number of environment MOs.
            self.beta_n_act_mos (int) = beta number of active MOs.
            self.beta_n_env_mos (int) = beta number of environment MOs.
        """
        if self.keywords['partition_method'] == 'spade':
            delta_s = [-(sigma[i+1] - sigma[i]) for i in range(len(sigma) - 1)]
            self.n_act_mos = np.argpartition(delta_s, -1)[-1] + 1
            self.n_env_mos = len(sigma) - self.n_act_mos
        else:
            assert isinstance(self.keywords['occupied_projection_basis'], str),\
                '\n Define a projection basis'
            self.n_act_mos = self.n_active_aos
            self.n_env_mos = len(sigma) - self.n_act_mos

        if self.keywords['reference'] == 'rhf':
            return self.n_act_mos, self.n_env_mos
        else:
            assert beta_sigma is not None, 'Provide beta singular values'
            if self.keywords['partition_method'] == 'spade':
                beta_delta_s = [-(beta_sigma[i+1] - beta_sigma[i]) \
                    for i in range(len(beta_sigma) - 1)]
                self.beta_n_act_mos = np.argpartition(beta_delta_s, -1)[-1] + 1
                self.beta_n_env_mos = len(beta_sigma) - self.beta_n_act_mos
            else:
                assert isinstance(self.keywords['occupied_projection_basis'], str),\
                    '\n Define a projection basis'
                self.beta_n_act_mos = self.beta_n_active_aos
                self.beta_n_env_mos = len(beta_sigma) - self.beta_n_act_mos
            return (self.n_act_mos, self.n_env_mos, self.beta_n_act_mos,
                    self.beta_n_env_mos)

    def banner(self):
        """Prints the banner in the output file."""
        self.outfile.write('\n')
        self.outfile.write(' ' + 65*'-' + '\n')
        self.outfile.write('                          PsiEmbed\n\n')
        self.outfile.write('                   Python Stack for Improved\n')
        self.outfile.write('  and Efficient Methods and Benchmarking in' 
                            + ' Embedding Development\n\n')
        self.outfile.write('                       Daniel Claudino\n')
        self.outfile.write('                       September  2019\n')
        self.outfile.write(' ' + 65*'-' + '\n')
        self.outfile.write('\n')
        self.outfile.write(' Main references: \n\n')
        self.outfile.write('     Projection-based embedding:\n')
        self.outfile.write('     F.R. Manby, M. Stella, J.D. Goodpaster,'
                            + ' T.F. Miller. III,\n')
        self.outfile.write('     J. Chem. Theory Comput. 2012, 8, 2564.\n\n')

        if self.keywords['partition_method'] == 'spade':
            self.outfile.write('     SPADE partition:\n')
            self.outfile.write('     D. Claudino, N.J. Mayhall,\n')
            self.outfile.write('     J. Chem. Theory Comput. 2019, 15, 1053.\n')
        return None

    def print_scf(self, e_act, e_env, two_e_cross, e_act_emb, correction):
        """Prints mean-field info from before and after embedding.

        Args:
            e_act (float): energy of the active subsystem.
            e_env (float): energy of the environment subsystem.
            two_e_cross (float): intersystem interaction energy.
            e_act_emb (float): energy of the embedded active subsystem.
            correction (float): correction from the embedded density.
        """
        self.outfile.write('\n\n Energy values in atomic units\n')
        self.outfile.write(' Embedded calculation: '
            + self.keywords['high_level'].upper()
            + '-in-' + self.keywords['low_level'].upper() + '\n\n')
        if self.keywords['partition_method'] == 'spade':
            if 'occupied_projection_basis' not in self.keywords:
                self.outfile.write(' Orbital partition method: SPADE\n')
            else:
                self.outfile.write((' Orbital partition method: SPADE with ',
                    'occupied space projected onto '
                    + self.keywords['occupied_projection_basis'].upper() + '\n'))
        else:
            self.outfile.write(' Orbital partition method: All AOs in '
                + self.keywords['occupied_projection_basis'].upper()
                + ' from atoms in A\n')

        self.outfile.write('\n')
        if hasattr(self, 'beta_n_act_mos') == False:
            self.outfile.write(' Number of orbitals in active subsystem: %s\n'
                                % self.n_act_mos)
            self.outfile.write(' Number of orbitals in environment: %s\n'
                                % self.n_env_mos)
        else:
            self.outfile.write(' Number of alpha orbitals in active subsystem:'
                                + ' %s\n' % self.n_act_mos)
            self.outfile.write(' Number of beta orbitals in active subsystem:'
                                + ' %s\n' % self.beta_n_act_mos)
            self.outfile.write(' Number of alpha orbitals in environment:'
                                + ' %s\n' % self.n_env_mos)
            self.outfile.write(' Number of beta orbitals in environment:'
                                + ' %s\n' % self.beta_n_env_mos)
        self.outfile.write('\n')
        self.outfile.write(' --- Before embedding --- \n')
        self.outfile.write(' {:<7} {:<6} \t\t = {:>16.10f}\n'.format('('
            + self.keywords['low_level'].upper() +')', 'E[A]', e_act))
        self.outfile.write(' {:<7} {:<6} \t\t = {:>16.10f}\n'.format('('
            + self.keywords['low_level'].upper() +')', 'E[B]', e_env))
        self.outfile.write(' Intersystem interaction G \t = {:>16.10f}\n'.
            format(two_e_cross))
        self.outfile.write(' Nuclear repulsion energy \t = {:>16.10f}\n'.
            format(self.nre))
        self.outfile.write(' {:<7} {:<6} \t\t = {:>16.10f}\n'.format('('
            + self.keywords['low_level'].upper() + ')', 'E[A+B]',
            e_act + e_env + two_e_cross + self.nre))
        self.outfile.write('\n')
        self.outfile.write(' --- After embedding --- \n')
        self.outfile.write(' Embedded SCF E[A] \t\t = {:>16.10f}\n'.
            format(e_act_emb))
        self.outfile.write(' Embedded density correction \t = {:>16.10f}\n'.
            format(correction))
        self.outfile.write(' Embedded HF-in-{:<5} E[A] \t = {:>16.10f}\n'.
            format(self.keywords['low_level'].upper(),
            e_act_emb + e_env + two_e_cross + self.nre + correction))
        self.outfile.write(' <SD_before|SD_after> \t\t = {:>16.10f}\n'.format(
            abs(self.determinant_overlap)))
        self.outfile.write('\n')
        return None

    def print_summary(self, e_mf_emb):
        """Prints summary of CL shells.

        Args:
            e_mf_emb (float): mean-field embedded energy.
        """
        self.outfile.write('\n Summary of virtual shell energy convergence\n\n')
        self.outfile.write('{:^8} \t {:^8} \t {:^12} \t {:^16}\n'.format(
            'Shell #', '# active', ' Correlation', 'Total'))
        self.outfile.write('{:^8} \t {:^8} \t {:^12} \t {:^16}\n'.format(
            8*'', 'virtuals', 'energy', 'energy'))
        self.outfile.write('{:^8} \t {:^8} \t {:^12} \t {:^16}\n'.format(
            7*'-', 8*'-', 13*'-', 16*'-'))

        for ishell in range(self.n_virtual_shell+1):
            self.outfile.write('{:^8d} \t {:^8} \t {:^12.10f} \t {:^12.10f}\n'\
                .format(ishell, self.shell_size*(ishell+1),
                self.correlation_energy_shell[ishell],
                e_mf_emb + self.correlation_energy_shell[ishell]))

        if (ishell == self.max_shell and
            self.keywords['n_virtual_shell'] > self.max_shell):
            n_virtuals = self._n_basis_functions - self.n_act_mos
            n_effective_virtuals = (self._n_basis_functions - self.n_act_mos
                                 - self.n_env_mos)
            self.outfile.write('{:^8} \t {:^8} \t {:^12.10f} \t {:^12.10f}\n'.
                format('Eff.', n_effective_virtuals,
                self.correlation_energy_shell[-1],
                e_mf_emb + self.correlation_energy_shell[-1]))
            self.outfile.write('{:^8} \t {:^8} \t {:^12.10f} \t {:^12.10f}\n'.
                format('Full', n_virtuals, self.correlation_energy_shell[-1],
                e_mf_emb + self.correlation_energy_shell[-1]))
        self.outfile.write('\n')
        return None
    
    def print_sigma(self, sigma, ishell):
        """ Formats the printing of singular values for the PRIME shells.

        Args:
            sigma (numpy.array or list): singular values.
            ishell (int): CL shell index.
        """
        self.outfile.write('\n{:>10} {:>2d}\n'.format('Shell #', ishell))
        self.outfile.write('  ------------\n')
        self.outfile.write('{:^5} \t {:^14}\n'.format('#','Singular value'))
        for i in range(len(sigma)):
            self.outfile.write('{:^5d} \t {:>12.10f}\n'.format(i, sigma[i]))
        self.outfile.write('\n')
        return None


class Psi4Embed(Embed):
    """Class with embedding methods using Psi4."""

    def run_psi4(self, level = None):
        """Runs Psi4 (PySCF is coming soon).

        Args:
            level (str): level of theory to run calculation.
        """
        if hasattr(self, '_mol'):
            psi4.set_options({'docc': [self.n_act_mos]
                'reference': self.keywords['high_level_reference']})
            if (self.keywords['high_level'][:2] == 'cc' and
                self.keywords['cc_type'] == 'df'):
                psi4.set_options({'cc_type': self.keywords['cc_type'],
                                'df_ints_io': 'save' })
        else:
            # Preparing molecule string with C1 symmetry
            add_c1 = self.keywords['geometry'].splitlines()
            add_c1.append('symmetry c1')
            self.keywords['geometry'] = '\n'.join(add_c1)

            # Running psi4 for the env (low level)
            psi4.set_memory(self.keywords['memory'])
            psi4.core.set_num_threads(self.keywords['num_threads'])
            self._mol = psi4.geometry(self.keywords['geometry'])
            self._mol.set_molecular_charge(self.keywords['charge'])
            self._mol.set_multiplicity(self.keywords['multiplicity'])

        psi4.core.be_quiet()
        psi4.core.set_output_file(self.keywords['driver_output'], True)
        psi4.set_options({'save_jk': 'true',
                        'basis': self.keywords['basis'],
                        'reference': self.keywords['low_level_reference'],
                        'ints_tolerance': self.keywords['ints_tolerance'],
                        'e_convergence': self.keywords['e_convergence'],
                        'd_convergence': self.keywords['d_convergence'],
                        'scf_type': self.keywords['eri'],
                        'print': self.keywords['print_level'],
                        'damping_percentage':
                            self.keywords['low_level_damping_percentage'],
                        'soscf': self.keywords['low_level_soscf']
                        })

        if level == None:
            self.e, self._wfn = psi4.energy(self.keywords['low_level'],
                molecule = self._mol, return_wfn=True)
            self._n_basis_functions = self._wfn.basisset().nbf()
            if self.keywords['low_level'] != 'HF' :
                self.e_xc_total = psi4.core.VBase.quadrature_values\
                            (self._wfn.V_potential())["FUNCTIONAL"]
                if self.keywords['reference'] == 'rhf':
                    self.v_xc_total = self._wfn.Va().clone().np
                else:
                    self.alpha_v_xc_total = self._wfn.Va().clone().np
                    self.beta_v_xc_total = self._wfn.Vb().clone().np
            else:
                if self.keywords['reference'] == 'rhf':
                    self.v_xc_total = np.zeros([self._n_basis_functions,
                        self._n_basis_functions])
                else:
                    self.alpha_v_xc_total = np.zeros([self._n_basis_functions,
                        self._n_basis_functions])
                    self.beta_v_xc_total = np.zeros([self._n_basis_functions,
                        self._n_basis_functions])
                self.e_xc_total = 0.0
        else:
            self.e, self._wfn = psi4.energy('hf',
                molecule = self._mol, return_wfn=True)

        if self.keywords['reference'] == 'rhf':
            self.occupied_orbitals = self._wfn.Ca_subset('AO', 'OCC').np
            self.j = self._wfn.jk().J()[0].np
            self.k = self._wfn.jk().K()[0].np
        else:
            self.alpha_occupied_orbitals = self._wfn.Ca_subset('AO', 'OCC').np
            self.beta_occupied_orbitals = self._wfn.Ca_subset('AO', 'OCC').np
            self.alpha_j = self._wfn.jk().J()[0].np
            self.beta_j = self._wfn.jk().J()[1].np
            self.alpha_k = self._wfn.jk().K()[0].np
            self.beta_k = self._wfn.jk().K()[1].np

        self.nre = self._mol.nuclear_repulsion_energy()
        self.ao_overlap = self._wfn.S().np
        self.h_core = self._wfn.H().np
        self.alpha = self._wfn.functional().x_alpha()
        return None

    def count_active_aos(self, basis):
        """Computes the number of AOs from active atoms.

        Args:
            basis (str): name of basis set from which to count
                active AOs.
        
        Returns:
            self.n_active_aos (int): number of AOs in the active atoms.
        """
        if basis == self.keywords['basis']:
            basis = self._wfn.basisset()
            n_basis_functions = basis.nbf()
        else:
            projected_wfn = psi4.core.Wavefunction.build(self._mol, basis)
            basis = projected_wfn.basisset()
            n_basis_functions = basis.nbf()
            
        self.n_active_aos = 0
        active_atoms = list(range(self.keywords['n_active_atoms']))
        for ao in range(n_basis_functions):
            for atom in active_atoms:
                if basis.function_to_center(ao) == atom:
                   self.n_active_aos += 1
        return self.n_active_aos
        
    def basis_projection(self, orbitals, projection_basis):
        """Defines a projection of orbitals in one basis onto another.
        
        Args:
            orbitals (numpy.array): MO coefficients to be projected.
            projection_basis (str): name of basis set onto which
                orbitals are to be projected.

        Returns:
            projected_orbitals (numpy.array): MO coefficients of
                orbitals projected onto projection_basis.
        """
        projected_wfn = psi4.core.Wavefunction.build(self._mol,
            projection_basis)
        mints = psi4.core.MintsHelper(projected_wfn.basisset())
        self.projected_overlap = (
            mints.ao_overlap().np[:self.n_active_aos, :self.n_active_aos])
        self.overlap_two_basis = (mints.ao_overlap(projected_wfn.basisset(),
                            self._wfn.basisset()).np[:self.n_active_aos, :])
        projected_orbitals = (np.linalg.inv(self.projected_overlap)
                            @ self.overlap_two_basis @ orbitals)
        return projected_orbitals

    def closed_shell_subsystem(self, orbitals):
        """
        Computes the potential matrices J, K, and V and subsystem energies.

        Args:
            orbitals (numpy.array): MO coefficients of subsystem.

        Returns:
            e (float): total energy of subsystem.
            e_xc (float): (DFT) Exchange-correlation energy of subsystem.
            j (numpy.array): Coulomb matrix of subsystem.
            k (numpy.array): Exchange matrix of subsystem.
            v_xc (numpy.array): Kohn-Sham potential matrix of subsystem.
        """

        density = orbitals @ orbitals.T
        psi4_orbitals = psi4.core.Matrix.from_array(orbitals)

        if hasattr(self._wfn, 'get_basisset'):
            jk = psi4.core.JK.build(self._wfn.basisset(),
                self._wfn.get_basisset('DF_BASIS_SCF'), 'DF')
        else:
            jk = psi4.core.JK.build(self._wfn.basisset())
        jk.set_memory(int(1.25e9))
        jk.initialize()
        jk.C_left_add(psi4_orbitals)
        jk.compute()
        jk.C_clear()
        jk.finalize()

        j = jk.J()[0].np
        k = jk.K()[0].np

        if(self._wfn.functional().name() != 'HF'):
            self._wfn.Da().copy(psi4.core.Matrix.from_array(density))
            self._wfn.form_V()
            v_xc = self._wfn.Va().clone().np
            e_xc = psi4.core.VBase.quadrature_values(
                self._wfn.V_potential())["FUNCTIONAL"]

        else:
            basis = self._wfn.basisset()
            n_basis_functions = basis.nbf()
            v_xc = np.zeros([n_basis_functions, n_basis_functions])
            e_xc = 0.0

        # Energy
        e = self.dot(density, 2.0*(self.h_core + j) - self.alpha*k) + e_xc
        return e, e_xc, j, k, v_xc

    def pseudocanonical(self, orbitals):
        """Returns pseudocanonical orbitals and the corresponding
            orbital energies.
        
        Args:
            orbitals (numpy.array): MO coefficients of orbitals to be
                pseudocanonicalized.

        Returns:
            e_orbital_pseudo (numpy.array): diagonal elements of the
                Fock matrix in the pseudocanonical basis.
            pseudo_orbitals (numpy.array): pseudocanonical orbitals.
        """
        mo_fock = orbitals.T @ self._wfn.Fa().np @ orbitals
        e_orbital_pseudo, pseudo_transformation = np.linalg.eigh(mo_fock)
        pseudo_orbitals = orbitals @ pseudo_transformation
        return e_orbital_pseudo, pseudo_orbitals

    def ao_operator(self):
        """Returns the matrix representation of the operator chosen to
            construct the shells.
        
        Returns:

            K (numpy.array): exchange.
            V (numpy.array): electron-nuclei potential.
            T (numpy.array): kinetic energy.
            H (numpy.array): core Hamiltonian.
            S (numpy.array): overlap matrix.
            F (numpy.array): Fock matrix.
            K_orb (numpy.array): K orbitals
                (see Feller and Davidson, JCP, 74, 3977 (1981)).
        """
        if (self.keywords['operator'] == 'K' or
            self.keywords['operator'] == 'K_orb'):
            jk = psi4.core.JK.build(self._wfn.basisset(),
                self._wfn.get_basisset('DF_BASIS_SCF'),'DF')
            jk.set_memory(int(1.25e9))
            jk.initialize()
            jk.print_header()
            jk.C_left_add(self._wfn.Ca())
            jk.compute()
            jk.C_clear()
            jk.finalize()
            self.operator = jk.K()[0].np
            if self.keywords['operator'] == 'K_orb':
                self.operator = 0.06*self._wfn.Fa().np - self.K
        elif self.keywords['operator'] == 'V':
            mints = psi4.core.MintsHelper(self._wfn.basisset())
            self.operator = mints.ao_potential().np
        elif self.keywords['operator'] == 'T':
            mints = psi4.core.MintsHelper(self._wfn.basisset())
            self.operator = mints.ao_kinetic().np
        elif self.keywords['operator'] == 'H':
            self.operator = self._wfn.H().np
        elif self.keywords['operator'] == 'S':
            self.operator = self._wfn.S().np
        elif self.keywords['operator'] == 'F':
            self.operator = self._wfn.Fa().np

    def open_shell_subsystem(self, alpha_orbitals, beta_orbitals):
        """
        Computes the potential matrices J, K, and V and subsystem
        energies for open shell cases.

        Args:
            alpha_orbitals (numpy.array): alpha MO coefficients.
            beta_orbitals (numpy.array): beta MO coefficients.

        Returns:
            e (float): total energy of subsystem.
            e_xc (float): Exchange-correlation energy of subsystem.
            alpha_j (numpy.array): alpha Coulomb matrix of subsystem.
            beta_j (numpy.array): beta Coulomb matrix of subsystem.
            alpha_k (numpy.array): alpha Exchange matrix of subsystem.
            beta_k (numpy.array): beta Exchange matrix of subsystem.
            alpha_v_xc (numpy.array): alpha Kohn-Sham potential matrix
                of subsystem.
            beta_v_xc (numpy.array): beta Kohn-Sham potential matrix
                of subsystem.
        """
        alpha_density = alpha_orbitals @ alpha_orbitals.T
        beta_density = beta_orbitals @ beta_orbitals.T

        # J and K
        jk = psi4.core.JK.build(self._wfn.basisset(),
            self._wfn.get_basisset('DF_BASIS_SCF'), 'DF')
        jk.set_memory(int(1.25e9))
        jk.initialize()
        jk.C_left_add(psi4.core.Matrix.from_array(alpha_orbitals))
        jk.C_left_add(psi4.core.Matrix.from_array(beta_orbitals))
        jk.compute()
        jk.C_clear()
        jk.finalize()
        alpha_j = jk.J()[0].np
        beta_j = jk.J()[1].np
        alpha_k = jk.K()[0].np
        beta_k = jk.K()[1].np
        
        if(self._wfn.functional().name() != 'HF'):
            self._wfn.Da().copy(psi4.core.Matrix.from_array(alpha_density))
            self._wfn.Db().copy(psi4.core.Matrix.from_array(beta_density))
            self._wfn.form_V()
            alpha_v_xc = self._wfn.Va().clone().np
            beta_v_xc = self._wfn.Vb().clone().np
            e_xc = psi4.core.VBase.quadrature_values(
                self._wfn.V_potential())['FUNCTIONAL']
        else:
            alpha_v_xc = np.zeros([self._n_basis_functions,
                self._n_basis_functions])
            beta_v_xc = np.zeros([self._n_basis_functions,
                self._n_basis_functions])
            e_xc = 0.0

        e = (self.dot(self.h_core, alpha_density + beta_density)
            + 0.5*(self.dot(alpha_j + beta_j, alpha_density + beta_density)
            - self.alpha*self.dot(alpha_k, alpha_density)
            - self.alpha*self.dot(beta_k, beta_density)) + e_xc)

        return e, e_xc, alpha_j, beta_j, alpha_k, beta_k, alpha_v_xc, beta_v_xc

    def orthonormalize(self, S, C, n_non_zero):
        """(Deprecated) Orthonormalizes a set of orbitals (vectors).

        Args:
            S (numpy.array): overlap matrix in AO basis.
            C (numpy.array): MO coefficient matrix (vectors to be orthonormalized).
            n_non_zero (int): number of orbitals that have non-zero norm.

        Returns:
            C_orthonormal (numpy.array): set of n_non_zero orthonormal orbitals.
        """

        overlap = C.T @ S @ C
        v, w = np.linalg.eigh(overlap)
        idx = v.argsort()[::-1]
        v = v[idx]
        w = w[:,idx]
        C_orthonormal = C @ w
        for i in range(n_non_zero):
            C_orthonormal[:,i] = C_orthonormal[:,i]/np.sqrt(v[i])
        return C_orthonormal[:,:n_non_zero]

    def molden(self, shell_orbitals, shell):
        """Creates molden file from orbitals at the shell.

        Args:
            span_orbitals (numpy.array): span orbitals.
            shell (int): shell index.
        """
        self._wfn.Ca().copy(psi4.core.Matrix.from_array(shell_orbitals))
        psi4.driver.molden(self._wfn, str(shell) + '.molden')
        return None

    def heatmap(self, span_orbitals, kernel_orbitals, shell):
        """Creates heatmap file from orbitals at the i-th shell.

        Args:
            span_orbitals (numpy.array): span orbitals.
            kernel_orbitals (numpy.array): kernel orbitalss.
            shell (int): shell index.
        """
        orbitals = np.hstack((span_orbitals, kernel_orbitals))
        mo_operator = orbitals.T @ self.operator @ orbitals
        np.savetxt('heatmap_'+str(shell)+'.dat', mo_operator)
        return None

    def determinant_overlap(self, orbitals, beta_orbitals = None):
        """
        Compute the overlap between determinants formed from the
        provided orbitals and the embedded orbitals

        Args:
            orbitals (numpy.array): orbitals to compute the overlap
                with embedded orbitals.
            beta_orbitals (numpy.array): beta orbitals, if running
                with references other then rhf.
        """
        if self.keywords['reference'] == 'rhf' and beta_orbitals == None:
            overlap = self.occupied_orbitals.T @ self.ao_overlap @ orbitals
            u, s, vh = np.linalg.svd(overlap)
            self.determinant_overlap = (
                np.linalg.det(u)*np.linalg.det(vh)*np.prod(s))
        else:
            assert beta_orbitals is not None, '\nProvide beta orbitals.'
            alpha_overlap = (self.alpha_occupied_orbitals.T @ self.ao_overlap
                @ beta_orbitals)
            u, s, vh = np.linalg.svd(alpha_overlap)
            self.determinant_overlap = 0.5*(
                np.linalg.det(u)*np.linalg.det(vh)*np.prod(s))
            beta_overlap = (self.beta_occupied_orbitals.T @ self.ao_overlap
                @ beta_orbitals)
            u, s, vh = np.linalg.svd(beta_overlap)
            self.determinant_overlap += 0.5*(
                np.linalg.det(u)*np.linalg.det(vh)*np.prod(s))
        return None

    def correlation_energy(self, span_orbitals = None, kernel_orbitals = None,
        span_orbital_energies = None, kernel_orbital_energies = None):
        """
        Computes the correlation energy for the current set of active
        virtual orbitals.
        
        Args:
            span_orbitals (numpy.array): orbitals transformed by the
                span of the previous shell.
            kernel_orbitals (numpy.array): orbitals transformed by the
                kernel of the previous shell.
            span_orbital_energies (nmpy.array): orbitals energies
                of the span orbitals.
            kernel_orbital_energies (nmpy.array): orbitals energies
                of the kernel orbitals.

        Returns:
            correlation_energy (float): correlation energy of the
                span_orbitals.
        """
        shift = self._n_basis_functions - self.n_env_mos
        if span_orbitals is None:
            nfrz = self.n_env_mos
        else:
            effective_orbitals = np.hstack((span_orbitals,
                kernel_orbitals))
            orbital_energies = np.concatenate((span_orbital_energies,
                kernel_orbital_energies))
            nfrz = (self._n_basis_functions - self.n_act_mos
                 - span_orbitals.shape[1])
            orbitals = np.hstack((self.occupied_orbitals,
                effective_orbitals, self._wfn.Ca().np[:, shift:]))
            orbital_energies = (
                np.concatenate((self._wfn.epsilon_a().np[:self.n_act_mos],
                orbital_energies, self._wfn.epsilon_a().np[shift:])))
            self._wfn.Ca().copy(psi4.core.Matrix.from_array(orbitals))
            self._wfn.epsilon_a().np[:] = orbital_energies[:]

        # Update the number of frozen orbitals and compute energy
        frzvpi = psi4.core.Dimension.from_list([nfrz])
        self._wfn.new_frzvpi(frzvpi)
        #wf_eng, wf_wfn = psi4.energy(self.keywords['high_level'],
            #ref_wfn = self._wfn, return_wfn = True)
        psi4.energy(self.keywords['high_level'], ref_wfn = self._wfn)
        correlation_energy = psi4.core.get_variable(
            self.keywords['high_level'].upper() + " CORRELATION ENERGY")
        self.correlation_energy_shell.append(correlation_energy)
        return correlation_energy

    def effective_virtuals(self):
        """Slices the effective virtuals from the entire virtual space.

        Returns:
            effective_orbitals (numpy.array): virtual orbitals without
                the level-shifted orbitals from the environment.
        """
        shift = self._n_basis_functions - self.n_env_mos
        effective_orbitals = self._wfn.Ca().np[:, self.n_act_mos:shift]
        return effective_orbitals

    def count_shells(self):
        """Guarantees the correct number of shells are computed.

        Returns:
            max_shell (int): maximum number of virtual shells.
            self.n_virtual_shell (int): number of virtual shells.
        """
        effective_dimension = (self._n_basis_functions - self.n_act_mos
                            - self.n_env_mos)
        self.max_shell = int(effective_dimension/self.shell_size)-1
        if (self.keywords['n_virtual_shell']
            > int(effective_dimension/self.shell_size)):
            self.n_virtual_shell = self.max_shell
        elif effective_dimension % self.shell_size == 0:
            self.n_virtual_shell = self.max_shell - 1
        else:
            self.n_virtual_shell = self.keywords['n_virtual_shell']
        return self.max_shell, self.n_virtual_shell
