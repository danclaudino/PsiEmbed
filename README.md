# PsiEmbed

**PsiEmbed** (Python Stack for Improved and Efficient Methods and Benchmarking in Embedding Development) is a python package designed to perform wave function-in-density functional theory (WF-in-DFT) projection-based embedding calculations. In principle, it can embedded any combination of wave function/mean-field method in another mean-field method, as long as the said methods are implemented in the quantum chemical package chosen by the user.

## Getting started
At present, this package only works in conjuntion with Psi4 and requires the latter to be installed. Despite the many alternative ways to install Psi4, in order to take advantage of Psi4Embed, one needs to install it from source. More precisely, my version of Psi4. This is necessary because projection-based embedding introduces an embedding potential via a modified core Hamiltonian/Fock operator, which are overwritten and built everytime a new wavefunction object is created by virtue of running a mean-field calculation.

Start by forking my Psi4 repository and compiling it according to the instructions provided in the Psi4 manual ([Compiling and Installing from Source](http://psicode.org/psi4manual/1.1/build_planning.html)). This package assumes Psi4 is used as a Python module, which can be accomplished by following [Using Psi4 as a Python Module](http://psicode.org/psi4manual/1.1/build_planning.html). This package has been developed with Python3 in mind, but no major complications are expected in using it with Python2 apart from potential syntactic changes.

