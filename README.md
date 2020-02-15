# PsiEmbed

**PsiEmbed** (Python Stack for Improved and Efficient Methods and Benchmarking in Embedding Development) is a python package designed to perform wave function-in-density functional theory (WF-in-DFT) projection-based embedding calculations. In principle, it can embedded any combination of wave function/mean-field method in another mean-field method, as long as the said methods are implemented in the quantum chemical package chosen by the user.

## Latest

PsiEmbed can now work with PySCF, and the corresponding documentation should come out shortly. It is now possible to save the embedded potentials, embedded orbitals, and embedded core Hamiltonian (the latter only for closed-shells). In the meantime, feel free to reach with questions/comments/issues.

## Getting started

PsiEmbed works in conjunction with well-established quantum chemical packages to take advantage of molecular integrals and solvers. It can be paired with Psi4 and PySCF.

To install the PsiEmbed package:

```
git clone https://github.com/danclaudino/PsiEmbed.git
cd PsiEmbed
python setup.py install
```

### Psi4
Despite the many alternative ways to install Psi4, in order to take advantage of Psi4Embed, one needs to install it from source. More precisely, my version of Psi4. This is necessary because projection-based embedding introduces an embedding potential via a modified core Hamiltonian/Fock operator, which are overwritten and built everytime a new wavefunction object is created by virtue of running a mean-field calculation.

Start by forking my Psi4 repository and compiling it according to the instructions provided in the Psi4 manual ([Compiling and Installing from Source](http://psicode.org/psi4manual/1.1/build_planning.html)). This package assumes Psi4 is used as a Python module, which can be accomplished by following [Using Psi4 as a Python Module](http://psicode.org/psi4manual/1.1/build_planning.html). This package has been developed with Python3 (3.6) in mind, so there may be some complications in using it with Python2 due potential syntactic changes and incompatible dependencies.

### PySCF
First, install the latest version (tested for version 1.70 in the pip repository). More on how to install PySCF can be found [here](https://sunqm.github.io/pyscf/). 

If you don't expect to run embedding calculations with open-shells systems, nothing is need apart from the installation above.

If you will compute open-shells, then the easiest way to get PsiEmbed to work with PySCF (for now) is to make some slight changes in the installed PySCF python package. Add the following two lines to the SCF base class (I include them in lines 1457-1458) in `/path/to/pyscf/scf/hf.py`:

```
def get_vemb(self):
    return numpy.zeros([2, self.mol.nao, self.mol.nao])
```

This allows for inclusion of a constant spin-dependent term in the Fock matrix that is just a bunch of zeros in the absence of an embedding potential. In turn, we change the UHF mean-field by having lines 213 and 214 in `/path/to/pyscf/uhf.py` look like this:

```
v_emb = mf.get_vemb()
f = h1e + vhf + v_emb
```

For ROHF, lines 62-64 in `/path/to/pyscf/rohf.py` should look like this:

```
v_emb = mf.get_vemb()
focka = h1e + vhf[0] + v_emb[0]
fockb = h1e + vhf[1] + v_emb[1]
```

And that is it!
