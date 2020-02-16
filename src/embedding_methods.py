import numpy as np
import psi4
import scipy.linalg
import sys
from pyscf import gto, dft, scf, lib, mp, cc
#import os

class Embed:
    """ Class with package-independent embedding methods."""

    def __init__(self, keywords):
        """
        Initialize the Embed class.

        Parameters
        ----------
        keywords (dict): dictionary with embedding options.
        """
        self.keywords = keywords
        self.correlation_energy_shell = []
        self.shell_size = 0
        self.outfile = open(keywords['embedding_output'], 'w')
        return None

    @staticmethod
    def dot(A, B):
        """
        (Deprecated) Computes the trace (dot or Hadamard product) 
        of matrices A and B.
        This has now been replaced by a lambda function in 
        embedding_module.py.

        Parameters
        ----------
        A : numpy.array
        B : numpy.array

        Returns
        -------
        The trace (dot product) of A * B

        """
        return np.einsum('ij, ij', A, B)

    def orbital_rotation(self, orbitals, n_active_aos, ao_overlap = None):
        """
        SVD orbitals projected onto active AOs to rotate orbitals.

        If ao_overlap is not provided, C is assumed to be in an
        orthogonal basis.
        
        Parameters
        ----------
        orbitals : numpy.array
            MO coefficient matrix.
        n_active_aos : int
            Number of atomic orbitals in the active atoms.
        ao_overlap : numpy.array (None)
            AO overlap matrix.

        Returns
        -------
        rotation_matrix : numpy.array
            Matrix to rotate orbitals.
        singular_values : numpy.array
            Singular values.
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

        Parameters
        ----------
        sigma : numpy.array
            Singular values.
        beta_sigma : numpy.array (None)
            Beta singular values.

        Returns
        -------
        self.n_act_mos : int
            (alpha) number of active MOs.
        self.n_env_mos : int
            (alpha) number of environment MOs.
        self.beta_n_act_mos : int
            Beta number of active MOs.
        self.beta_n_env_mos : int
            Beta number of environment MOs.
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

        if self.keywords['low_level_reference'] == 'rhf':
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

    def header(self):
        """Prints the header in the output file."""
        self.outfile.write('\n')
        self.outfile.write(' ' + 75*'-' + '\n')
        self.outfile.write('                               PsiEmbed\n\n')
        self.outfile.write('                        Python Stack for Improved\n')
        self.outfile.write('       and Efficient Methods and Benchmarking in'
                            + ' Embedding Development\n\n')
        self.outfile.write('                            Daniel Claudino\n')
        self.outfile.write('                            September  2019\n')
        self.outfile.write(' ' + 75*'-' + '\n')
        self.outfile.write('\n')
        self.outfile.write(' Main references: \n\n')
        self.outfile.write('     Projection-based embedding:\n')
        self.outfile.write('     F.R. Manby, M. Stella, J.D. Goodpaster,'
                            + ' T.F. Miller. III,\n')
        self.outfile.write('     J. Chem. Theory Comput. 2012, 8, 2564.\n\n')

        if self.keywords['partition_method'] == 'spade':
            self.outfile.write('     SPADE partition:\n')
            self.outfile.write('     D. Claudino, N.J. Mayhall,\n')
            self.outfile.write('     J. Chem. Theory Comput. 2019, 15, 1053.\n\n')

        if 'n_cl_shell' in self.keywords.keys():
            self.outfile.write('     Concentric localization (CL):\n')
            self.outfile.write('     D. Claudino, N.J. Mayhall,\n')
            self.outfile.write('     J. Chem. Theory Comput. 2019, 15, 6085.\n\n')

        if self.keywords['package'].lower() == 'psi4':
            self.outfile.write('     Psi4:\n')
            self.outfile.write('     R. M. Parrish, L. A. Burns, D. G. A. Smith'
                + ', A. C. Simmonett, \n')
            self.outfile.write('     A. E. DePrince III, E. G. Hohenstein'
                + ', U. Bozkaya, A. Yu. Sokolov,\n')
            self.outfile.write('     R. Di Remigio, R. M. Richard, J. F. Gonthier'
                + ', A. M. James,\n') 
            self.outfile.write('     H. R. McAlexander, A. Kumar, M. Saitow'
                + ', X. Wang, B. P. Pritchard,\n')
            self.outfile.write('     P. Verma, H. F. Schaefer III'
                + ', K. Patkowski, R. A. King, E. F. Valeev,\n')
            self.outfile.write('     F. A. Evangelista, J. M. Turney,'
                + 'T. D. Crawford, and C. D. Sherrill,\n')
            self.outfile.write('     J. Chem. Theory Comput. 2017, 13, 3185.')

        if self.keywords['package'].lower() == 'pyscf':
            self.outfile.write('     PySCF:\n')
            self.outfile.write('     Q. Sun, T. C. Berkelbach, N. S. Blunt'
                + ', G. H. Booth, S. Guo, Z. Li,\n')
            self.outfile.write('     J. Liu, J. D. McClain, E. R. Sayfutyarova'
                + ', S. Sharma, S. Wouters,\n')
            self.outfile.write('     and G. K.‐L. Chan,\n')
            self.outfile.write('     WIREs Comput. Mol. Sci. 2018, 8, e1340.')
        self.outfile.write('\n\n')
        self.outfile.write(' ' + 75*'-' + '\n')
        return None

    def print_scf(self, e_act, e_env, two_e_cross, e_act_emb, correction):
        """
        Prints mean-field info from before and after embedding.

        Parameters
        ----------
        e_act : float
            Energy of the active subsystem.
        e_env : float
            Energy of the environment subsystem.
        two_e_cross : float
            Intersystem interaction energy.
        e_act_emb : float
            Energy of the embedded active subsystem.
        correction : float
            Correction from the embedded density.
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
        """
        Prints summary of CL shells.

        Parameters
        ----------
        e_mf_emb : float
            Mean-field embedded energy.
        """
        self.outfile.write('\n Summary of virtual shell energy convergence\n\n')
        self.outfile.write('{:^8} \t {:^8} \t {:^12} \t {:^16}\n'.format(
            'Shell #', '# active', ' Correlation', 'Total'))
        self.outfile.write('{:^8} \t {:^8} \t {:^12} \t {:^16}\n'.format(
            8*'', 'virtuals', 'energy', 'energy'))
        self.outfile.write('{:^8} \t {:^8} \t {:^12} \t {:^16}\n'.format(
            7*'-', 8*'-', 13*'-', 16*'-'))

        for ishell in range(self.n_cl_shell+1):
            self.outfile.write('{:^8d} \t {:^8} \t {:^12.10f} \t {:^12.10f}\n'\
                .format(ishell, self.shell_size*(ishell+1),
                self.correlation_energy_shell[ishell],
                e_mf_emb + self.correlation_energy_shell[ishell]))

        if (ishell == self.max_shell and
            self.keywords['n_cl_shell'] > self.max_shell):
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
        """
        Formats the printing of singular values from the CL shells.

        Parameters
        ----------
        sigma : numpy.array or list
            Singular values.
        ishell :int
            CL shell index.
        """
        self.outfile.write('\n{:>10} {:>2d}\n'.format('Shell #', ishell))
        self.outfile.write('  ------------\n')
        self.outfile.write('{:^5} \t {:^14}\n'.format('#','Singular value'))
        for i in range(len(sigma)):
            self.outfile.write('{:^5d} \t {:>12.10f}\n'.format(i, sigma[i]))
        self.outfile.write('\n')
        return None

    def determinant_overlap(self, orbitals, beta_orbitals = None):
        """
        Compute the overlap between determinants formed from the
        provided orbitals and the embedded orbitals

        Parameters
        ----------
        orbitals : numpy.array
            Orbitals to compute the overlap with embedded orbitals.
        beta_orbitals : numpy.array (None)
            Beta orbitals, if running with references other than RHF.
        """
        if self.keywords['high_level_reference'] == 'rhf' and beta_orbitals == None:
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

    def count_shells(self):
        """
        Guarantees the correct number of shells are computed.

        Returns
        -------
        max_shell : int
            Maximum number of virtual shells.
        self.n_cl_shell : int
            Number of virtual shells.
        """
        effective_dimension = (self._n_basis_functions - self.n_act_mos
                            - self.n_env_mos)
        self.max_shell = int(effective_dimension/self.shell_size)-1
        if (self.keywords['n_cl_shell']
            > int(effective_dimension/self.shell_size)):
            self.n_cl_shell = self.max_shell
        elif effective_dimension % self.shell_size == 0:
            self.n_cl_shell = self.max_shell - 1
        else:
            self.n_cl_shell = self.keywords['n_cl_shell']
        return self.max_shell, self.n_cl_shell


class PySCFEmbed(Embed):
    """Class with embedding methods using PySCF."""
    
    def run_mean_field(self, v_emb = None):
        """
        Runs mean-field calculation with PySCF.
        If 'level' is not provided, it runs the a calculation at the level
        given by the 'low_level' key in self.keywords. HF otherwise.

        Parameters
        ----------
        v_emb : numpy.array or list of numpy.array (None)
            Embedding potential.
        """
        self._mol = gto.mole.Mole()
        self._mol.verbose = self.keywords['print_level']
        #self._mol.output = self.keywords['driver_output']
        self._mol.atom = self.keywords['geometry']
        self._mol.max_memory = self.keywords['memory']
        self._mol.basis = self.keywords['basis']
        if v_emb is None: # low-level/environment calculation
            self._mol.output = self.keywords['driver_output']
            if self.keywords['low_level'] == 'hf':
                if self.keywords['low_level_reference'].lower() == 'rhf':
                    self._mean_field = scf.RHF(self._mol)
                if self.keywords['low_level_reference'].lower() == 'uhf':
                    self._mean_field = scf.UHF(self._mol)
                if self.keywords['low_level_reference'].lower() == 'rohf':
                    self._mean_field = scf.ROHF(self._mol)
                self.e_xc = 0.0
            else:
                if self.keywords['low_level_reference'].lower() == 'rhf':
                    self._mean_field = dft.RKS(self._mol)
                if self.keywords['low_level_reference'].lower() == 'uhf':
                    self._mean_field = dft.UKS(self._mol)
                if self.keywords['low_level_reference'].lower() == 'rohf':
                    self._mean_field = dft.ROKS(self._mol)
            self._mean_field.conv_tol = self.keywords['e_convergence']
            self._mean_field.xc = self.keywords['low_level']
            self._mean_field.kernel()
            self.v_xc_total = self._mean_field.get_veff()
            self.e_xc_total = self._mean_field.get_veff().exc
        else:
            if self.keywords['high_level_reference'].lower() == 'rhf':
                self._mean_field = scf.RHF(self._mol)
            if self.keywords['high_level_reference'].lower() == 'uhf':
                self._mean_field = scf.UHF(self._mol)
            if self.keywords['high_level_reference'].lower() == 'rohf':
                self._mean_field = scf.ROHF(self._mol)
            if self.keywords['low_level_reference'].lower() == 'rhf':
                self._mol.nelectron = 2*self.n_act_mos
                self._mean_field.get_hcore = lambda *args: v_emb + self.h_core
            if (self.keywords['low_level_reference'].lower() == 'rohf'
                or self.keywords['low_level_reference'].lower() == 'uhf'):
                self._mol.nelectron = self.n_act_mos + self.beta_n_act_mos
                self._mean_field.get_vemb = lambda *args: v_emb
            self._mean_field.conv_tol = self.keywords['e_convergence']
            self._mean_field.kernel()

        if self.keywords['low_level_reference'] == 'rhf':
            docc = (self._mean_field.mo_occ == 2).sum()
            self.occupied_orbitals = self._mean_field.mo_coeff[:, :docc]
            self.j, self.k = self._mean_field.get_jk() 
            self.v_xc_total = self._mean_field.get_veff() - self.j
        else:
            if (self.keywords['low_level_reference'] == 'uhf' and v_emb is None
                or self.keywords['high_level_reference'] == 'uhf'
                and v_emb is not None):
                n_alpha = (self._mean_field.mo_occ[0] == 1).sum()
                n_beta = (self._mean_field.mo_occ[1] == 1).sum()
                self.alpha_occupied_orbitals = self._mean_field.mo_coeff[
                    0, :, :n_alpha]
                self.beta_occupied_orbitals = self._mean_field.mo_coeff[
                    1, :, :n_beta]
            if (self.keywords['low_level_reference'] == 'rohf' and v_emb is None
                or self.keywords['high_level_reference'] == 'rohf'
                and v_emb is not None):
                n_beta = (self._mean_field.mo_occ == 2).sum()
                n_alpha = n_beta + (self._mean_field.mo_occ == 1).sum()
                self.alpha_occupied_orbitals = self._mean_field.mo_coeff[:, :n_alpha]
                self.beta_occupied_orbitals = self._mean_field.mo_coeff[:, :n_beta]
            j, k = self._mean_field.get_jk() 
            self.alpha_j = j[0] 
            self.beta_j = j[1]
            self.alpha_k = k[0]
            self.beta_k = k[1]
            self.alpha_v_xc_total = self._mean_field.get_veff()[0] - j[0] - j[1]
            self.beta_v_xc_total = self._mean_field.get_veff()[1] - j[0] - j[1]

        self.alpha = 0.0
        self._n_basis_functions = self._mol.nao
        self.nre = self._mol.energy_nuc()
        self.ao_overlap = self._mean_field.get_ovlp(self._mol)
        self.h_core = self._mean_field.get_hcore(self._mol)
        return None

    def count_active_aos(self, basis = None):
        """
        Computes the number of AOs from active atoms.

        Parameters
        ----------
        basis : str
            Name of basis set from which to count active AOs.
        
        Returns
        -------
            self.n_active_aos : int
                Number of AOs in the active atoms.
        """
        if basis is None:
            self.n_active_aos = self._mol.aoslice_nr_by_atom()[
                self.keywords['n_active_atoms']-1][3]
        else:
            self._projected_mol = gto.mole.Mole()
            self._projected_mol.atom = self.keywords['geometry']
            self._projected_mol.basis = basis 
            self._projected_mf = scf.RHF(self._projected_mol)
            self.n_active_aos = self._projected_mol.aoslice_nr_by_atom()[
                self.keywords['n_active_atoms']-1][3]
        return self.n_active_aos
        
    def basis_projection(self, orbitals, projection_basis):
        """
        Defines a projection of orbitals in one basis onto another.
        
        Parameters
        ----------
        orbitals : numpy.array
            MO coefficients to be projected.
        projection_basis : str
            Name of basis set onto which orbitals are to be projected.

        Returns
        -------
        projected_orbitals : numpy.array
            MO coefficients of orbitals projected onto projection_basis.
        """
        self.projected_overlap = (self._projected_mf.get_ovlp(self._mol)
            [:self.n_active_aos, :self.n_active_aos])
        self.overlap_two_basis = gto.intor_cross('int1e_ovlp_sph', 
            self._mol, self._projected_mol)[:self.n_active_aos, :]
        projected_orbitals = (np.linalg.inv(self.projected_overlap)
                            @ self.overlap_two_basis @ orbitals)
        return projected_orbitals

    def closed_shell_subsystem(self, orbitals):
        """
        Computes the potential matrices J, K, and V and subsystem energies.

        Parameters
        ----------
        orbitals : numpy.array
            MO coefficients of subsystem.

        Returns
        -------
        e : float
            Total energy of subsystem.
        e_xc : float
            DFT Exchange-correlation energy of subsystem.
        j : numpy.array
            Coulomb matrix of subsystem.
        k : numpy.array
            Exchange matrix of subsystem.
        v_xc : numpy.array
            Kohn-Sham potential matrix of subsystem.
        """
        density = 2.0*orbitals @ orbitals.T
        #It seems that PySCF lumps J and K in the J array 
        j = self._mean_field.get_j(dm = density)
        k = np.zeros([self._n_basis_functions, self._n_basis_functions])
        two_e_term =  self._mean_field.get_veff(self._mol, density)
        e_xc = two_e_term.exc
        v_xc = two_e_term - j 

        # Energy
        e = self.dot(density, self.h_core + j/2) + e_xc
        return e, e_xc, j, k, v_xc

    def open_shell_subsystem(self, alpha_orbitals, beta_orbitals):
        """
        Computes the potential matrices J, K, and V and subsystem
        energies for open shell cases.

        Parameters
        ----------
        alpha_orbitals : numpy.array
            Alpha MO coefficients.
        beta_orbitals : numpy.array
            Beta MO coefficients.

        Returns
        -------
        e : float
            Total energy of subsystem.
        e_xc : float
            Exchange-correlation energy of subsystem.
        alpha_j : numpy.array
            Alpha Coulomb matrix of subsystem.
        beta_j : numpy.array
            Beta Coulomb matrix of subsystem.
        alpha_k : numpy.array
            Alpha Exchange matrix of subsystem.
        beta_k : numpy.array
            Beta Exchange matrix of subsystem.
        alpha_v_xc : numpy.array
            Alpha Kohn-Sham potential matrix of subsystem.
        beta_v_xc : numpy.array
            Beta Kohn-Sham potential matrix of subsystem.
        """
        alpha_density = alpha_orbitals @ alpha_orbitals.T
        beta_density = beta_orbitals @ beta_orbitals.T

        # J and K
        j = self._mean_field.get_j(dm = [alpha_density, beta_density])
        alpha_j = j[0]
        beta_j = j[1]
        alpha_k = np.zeros([self._n_basis_functions, self._n_basis_functions])
        beta_k = np.zeros([self._n_basis_functions, self._n_basis_functions])
        two_e_term =  self._mean_field.get_veff(self._mol, [alpha_density,
            beta_density])
        e_xc = two_e_term.exc
        alpha_v_xc = two_e_term[0] - (j[0] + j[1])
        beta_v_xc = two_e_term[1] - (j[0]+j[1])

        # Energy
        e = (self.dot(self.h_core, alpha_density + beta_density)
            + 0.5*(self.dot(alpha_j + beta_j, alpha_density + beta_density))
            + e_xc)

        return e, e_xc, alpha_j, beta_j, alpha_k, beta_k, alpha_v_xc, beta_v_xc
        
    def correlation_energy(self, span_orbitals = None, kernel_orbitals = None,
        span_orbital_energies = None, kernel_orbital_energies = None):
        """
        Computes the correlation energy for the current set of active
        virtual orbitals.
        
        Parameters
        ----------
        span_orbitals : numpy.array
            Orbitals transformed by the span of the previous shell.
        kernel_orbitals : numpy.array
            Orbitals transformed by the kernel of the previous shell.
        span_orbital_energies : numpy.array
            Orbitals energies of the span orbitals.
        kernel_orbital_energies : numpy.array
            Orbitals energies of the kernel orbitals.

        Returns
        -------
        correlation_energy : float
            Correlation energy of the span_orbitals.
        """

        shift = self._n_basis_functions - self.n_env_mos
        if span_orbitals is None:
            # If not using CL orbitals, just freeze the level-shifted MOs
            frozen_orbitals = [i for i in range(shift, self._n_basis_functions)]
        else:
            # Preparing orbitals and energies for CL shell
            effective_orbitals = np.hstack((span_orbitals, kernel_orbitals))
            orbital_energies = np.concatenate((span_orbital_energies,
                kernel_orbital_energies))
            frozen_orbitals = [i for i in range(self.n_act_mos
                + span_orbitals.shape[1], self._n_basis_functions)]
            orbitals = np.hstack((self.occupied_orbitals,
                effective_orbitals, self._mean_field.mo_coeff[:, shift:]))
            orbital_energies = (
                np.concatenate((self._mean_field.mo_energy[:self.n_act_mos],
                orbital_energies, self._mean_field.mo_energy[shift:])))
            # Replace orbitals in the mean_field obj by the CL orbitals
            # and compute correlation energy
            self._mean_field.mo_energy = orbital_energies
            self._mean_field.mo_coeff = orbitals

        if self.keywords['high_level'].lower() == 'mp2':
            #embedded_wf = mp.MP2(self._mean_field).run()
            embedded_wf = mp.MP2(self._mean_field).set(frozen = frozen_orbitals).run()
            correlation_energy = embedded_wf.e_corr
        if (self.keywords['high_level'].lower() == 'ccsd' or 
            self.keywords['high_level'].lower() == 'ccsd(t)'):
            embedded_wf = cc.CCSD(self._mean_field).set(frozen = frozen_orbitals).run()
            correlation_energy = embedded_wf.e_corr
            if self.keywords['high_level'].lower() == 'ccsd(t)':
                t_correction = embedded_wf.ccsd_t().T
                correlation_energy += t_correction
        # if span_orbitals provided, store correlation energy of shells
        if span_orbitals is not None:
            self.correlation_energy_shell.append(correlation_energy)
        return correlation_energy

    def effective_virtuals(self):
        """
        Slices the effective virtuals from the entire virtual space.

        Returns
        -------
        effective_orbitals : numpy.array
            Virtual orbitals without the level-shifted orbitals
            from the environment.
        """
        shift = self._n_basis_functions - self.n_env_mos
        effective_orbitals = self._mean_field.mo_coeff[:, self.n_act_mos:shift]
        return effective_orbitals

    def pseudocanonical(self, orbitals):
        """
        Returns pseudocanonical orbitals and the corresponding
        orbital energies.
        
        Parameters
        ----------
        orbitals : numpy.array
            MO coefficients of orbitals to be pseudocanonicalized.

        Returns
        -------
        e_orbital_pseudo : numpy.array
            diagonal elements of the Fock matrix in the
            pseudocanonical basis.
        pseudo_orbitals : numpy.array
            pseudocanonical orbitals.
        """
        fock_matrix = self._mean_field.get_fock()
        mo_fock = orbitals.T @ fock_matrix @ orbitals
        e_orbital_pseudo, pseudo_transformation = np.linalg.eigh(mo_fock)
        pseudo_orbitals = orbitals @ pseudo_transformation
        return e_orbital_pseudo, pseudo_orbitals

    def ao_operator(self):
        """
        Returns the matrix representation of the operator chosen to
        construct the shells.

        Returns
        -------

        K : numpy.array
            Exchange.
        V : numpy.array
            Electron-nuclei potential.
        T : numpy.array
            Kinetic energy.
        H : numpy.array
            Core (one-particle) Hamiltonian.
        S : numpy.array
            Overlap matrix.
        F : numpy.array
            Fock matrix.
        K_orb : numpy.array
            K orbitals (see Feller and Davidson, JCP, 74, 3977 (1981)).
        """
        if (self.keywords['operator'] == 'K' or
            self.keywords['operator'] == 'K_orb'):
            self.operator = self._mean_field.get_k()
            if self.keywords['operator'] == 'K_orb':
                self.operator = 0.06*self._mean_field.get_fock() - self.operator
        elif self.keywords['operator'] == 'V':
            self.operator = self._mol.intor_symmetric('int1e_nuc')
        elif self.keywords['operator'] == 'T':
            self.operator = self._mol.intor_symmetric('int1e_kin')
        elif self.keywords['operator'] == 'H':
            self.operator = self._mean_field.get_hcore()
        elif self.keywords['operator'] == 'S':
            self.operator = self._mean_field.get_ovlp()
        elif self.keywords['operator'] == 'F':
            self.operator = self._mean_field.get_fock()
        return None

class Psi4Embed(Embed):
    """Class with embedding methods using Psi4."""


    def run_mean_field(self, v_emb = None):
        """
        Runs Psi4 (PySCF is coming soon).
        If 'level' is not provided, it runs the a calculation at the level
        given by the 'low_level' key in self.keywords.

        Parameters
        ----------
        v_emb : numpy.array or list of numpy.array (None)
            Embedding potential.
        """
        if v_emb is None:
            self.outfile = open(self.keywords['embedding_output'], 'w')
            # Preparing molecule string with C1 symmetry
            add_c1 = self.keywords['geometry'].splitlines()
            add_c1.append('symmetry c1')
            self.keywords['geometry'] = '\n'.join(add_c1)

            # Running psi4 for the env (low level)
            psi4.set_memory(str(self.keywords['memory']) + ' MB')
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

            self.e, self._wfn = psi4.energy(self.keywords['low_level'],
                molecule = self._mol, return_wfn=True)
            self._n_basis_functions = self._wfn.basisset().nbf()
            if self.keywords['low_level'] != 'HF' :
                self.e_xc_total = psi4.core.VBase.quadrature_values\
                            (self._wfn.V_potential())["FUNCTIONAL"]
                if self.keywords['low_level_reference'] == 'rhf':
                    self.v_xc_total = self._wfn.Va().clone().np
                else:
                    self.alpha_v_xc_total = self._wfn.Va().clone().np
                    self.beta_v_xc_total = self._wfn.Vb().clone().np
            else:
                if self.keywords['low_level_reference'] == 'rhf':
                    #self.v_xc_total = np.zeros([self._n_basis_functions,
                        #self._n_basis_functions])
                    self.v_xc_total = 0.0
                else:
                    #self.alpha_v_xc_total = np.zeros([self._n_basis_functions,
                        #self._n_basis_functions])
                    #self.beta_v_xc_total = np.zeros([self._n_basis_functions,
                        #self._n_basis_functions])
                    self.alpha_v_xc_total = 0.0 
                    self.beta_v_xc_total = 0.0 
                self.e_xc_total = 0.0
        else:
            psi4.set_options({'docc': [self.n_act_mos],
                'reference': self.keywords['high_level_reference']})
            if self.keywords['high_level_reference'] == 'rhf':
                f = open('newH.dat', 'w')
                for i in range(self.h_core.shape[0]):
                    for j in range(self.h_core.shape[1]):
                        f.write("%s\n" % (self.h_core + v_emb)[i, j])
                f.close()
            else:
                psi4.set_options({'socc': [self.n_act_mos - self.beta_n_act_mos]})
                fa = open('Va_emb.dat', 'w')
                fb = open('Vb_emb.dat', 'w')
                for i in range(self.h_core.shape[0]):
                    for j in range(self.h_core.shape[1]):
                        fa.write("%s\n" % v_emb[0][i, j])
                        fb.write("%s\n" % v_emb[1][i, j])
                fa.close()
                fb.close()

            if (self.keywords['high_level'][:2] == 'cc' and
                self.keywords['cc_type'] == 'df'):
                psi4.set_options({'cc_type': self.keywords['cc_type'],
                                'df_ints_io': 'save' })
            self.e, self._wfn = psi4.energy('hf',
                molecule = self._mol, return_wfn=True)

        if self.keywords['low_level_reference'] == 'rhf':
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

    def count_active_aos(self, basis = None):
        """
        Computes the number of AOs from active atoms.

        Parameters
        ----------
        basis : str
            Name of basis set from which to count active AOs.
        
        Returns
        -------
            self.n_active_aos : int
                Number of AOs in the active atoms.
        """
        if basis is None:
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
        """
        Defines a projection of orbitals in one basis onto another.
        
        Parameters
        ----------
        orbitals : numpy.array
            MO coefficients to be projected.
        projection_basis : str
            Name of basis set onto which orbitals are to be projected.

        Returns
        -------
        projected_orbitals : numpy.array
            MO coefficients of orbitals projected onto projection_basis.
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

        Parameters
        ----------
        orbitals : numpy.array
            MO coefficients of subsystem.

        Returns
        -------
        e : float
            Total energy of subsystem.
        e_xc : float
            DFT Exchange-correlation energy of subsystem.
        j : numpy.array
            Coulomb matrix of subsystem.
        k : numpy.array
            Exchange matrix of subsystem.
        v_xc : numpy.array
            Kohn-Sham potential matrix of subsystem.
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
            v_xc = 0.0
            e_xc = 0.0

        # Energy
        e = self.dot(density, 2.0*(self.h_core + j) - self.alpha*k) + e_xc
        return e, e_xc, 2.0 * j, k, v_xc

    def pseudocanonical(self, orbitals):
        """
        Returns pseudocanonical orbitals and the corresponding
        orbital energies.
        
        Parameters
        ----------
        orbitals : numpy.array
            MO coefficients of orbitals to be pseudocanonicalized.

        Returns
        -------
        e_orbital_pseudo : numpy.array
            diagonal elements of the Fock matrix in the
            pseudocanonical basis.
        pseudo_orbitals : numpy.array
            pseudocanonical orbitals.
        """
        mo_fock = orbitals.T @ self._wfn.Fa().np @ orbitals
        e_orbital_pseudo, pseudo_transformation = np.linalg.eigh(mo_fock)
        pseudo_orbitals = orbitals @ pseudo_transformation
        return e_orbital_pseudo, pseudo_orbitals

    def ao_operator(self):
        """
        Returns the matrix representation of the operator chosen to
        construct the shells.

        Returns
        -------

        K : numpy.array
            Exchange.
        V : numpy.array
            Electron-nuclei potential.
        T : numpy.array
            Kinetic energy.
        H : numpy.array
            Core (one-particle) Hamiltonian.
        S : numpy.array
            Overlap matrix.
        F : numpy.array
            Fock matrix.
        K_orb : numpy.array
            K orbitals (see Feller and Davidson, JCP, 74, 3977 (1981)).
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

        Parameters
        ----------
        alpha_orbitals : numpy.array
            Alpha MO coefficients.
        beta_orbitals : numpy.array
            Beta MO coefficients.

        Returns
        -------
        e : float
            Total energy of subsystem.
        e_xc : float
            Exchange-correlation energy of subsystem.
        alpha_j : numpy.array
            Alpha Coulomb matrix of subsystem.
        beta_j : numpy.array
            Beta Coulomb matrix of subsystem.
        alpha_k : numpy.array
            Alpha Exchange matrix of subsystem.
        beta_k : numpy.array
            Beta Exchange matrix of subsystem.
        alpha_v_xc : numpy.array
            Alpha Kohn-Sham potential matrix of subsystem.
        beta_v_xc : numpy.array
            Beta Kohn-Sham potential matrix of subsystem.
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
            #alpha_v_xc = np.zeros([self._n_basis_functions,
                #self._n_basis_functions])
            #beta_v_xc = np.zeros([self._n_basis_functions,
                #self._n_basis_functions])
            alpha_v_xc = 0.0
            beta_v_xc = 0.0
            e_xc = 0.0

        e = (self.dot(self.h_core, alpha_density + beta_density)
            + 0.5*(self.dot(alpha_j + beta_j, alpha_density + beta_density)
            - self.alpha*self.dot(alpha_k, alpha_density)
            - self.alpha*self.dot(beta_k, beta_density)) + e_xc)

        return e, e_xc, alpha_j, beta_j, alpha_k, beta_k, alpha_v_xc, beta_v_xc

    def orthonormalize(self, S, C, n_non_zero):
        """
        (Deprecated) Orthonormalizes a set of orbitals (vectors).

        Parameters
        ----------
        S : numpy.array
            Overlap matrix in AO basis.
        C : numpy.array
            MO coefficient matrix, vectors to be orthonormalized.
        n_non_zero : int
            Number of orbitals that have non-zero norm.

        Returns
        -------
        C_orthonormal : numpy.array
            Set of n_non_zero orthonormal orbitals.
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
        """
        Creates molden file from orbitals at the shell.

        Parameters
        ----------
        span_orbitals : numpy.array
            Span orbitals.
        shell : int
            Shell index.
        """
        self._wfn.Ca().copy(psi4.core.Matrix.from_array(shell_orbitals))
        psi4.driver.molden(self._wfn, str(shell) + '.molden')
        return None

    def heatmap(self, span_orbitals, kernel_orbitals, shell):
        """
        Creates heatmap file from orbitals at the i-th shell.

        Parameters
        ----------
        span_orbitals : numpy.array
            Span orbitals.
        kernel_orbitals : numpy.array
            Kernel orbitals.
        shell : int
            Shell index.
        """
        orbitals = np.hstack((span_orbitals, kernel_orbitals))
        mo_operator = orbitals.T @ self.operator @ orbitals
        np.savetxt('heatmap_'+str(shell)+'.dat', mo_operator)
        return None

    def correlation_energy(self, span_orbitals = None, kernel_orbitals = None,
        span_orbital_energies = None, kernel_orbital_energies = None):
        """
        Computes the correlation energy for the current set of active
        virtual orbitals.
        
        Parameters
        ----------
        span_orbitals : numpy.array
            Orbitals transformed by the span of the previous shell.
        kernel_orbitals : numpy.array
            Orbitals transformed by the kernel of the previous shell.
        span_orbital_energies : numpy.array
            Orbitals energies of the span orbitals.
        kernel_orbital_energies : numpy.array
            Orbitals energies of the kernel orbitals.

        Returns
        -------
        correlation_energy : float
            Correlation energy of the span_orbitals.
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
        """
        Slices the effective virtuals from the entire virtual space.

        Returns
        -------
        effective_orbitals : numpy.array
            Virtual orbitals without the level-shifted orbitals
            from the environment.
        """
        shift = self._n_basis_functions - self.n_env_mos
        effective_orbitals = self._wfn.Ca().np[:, self.n_act_mos:shift]
        return effective_orbitals

