import psi4
import sys
from psi_embed import driver 

# Simple (minimal?) example of projection-based embedding using
# SPADE and CL orbitals
# Enter embedding options in the dictionary below:
options = {}
options['geometry'] = """
O       -1.1867 -0.2472  0.0000
H       -1.9237  0.3850  0.0000
H       -0.0227  1.1812  0.8852
C        0.0000  0.5526  0.0000
H       -0.0227  1.1812 -0.8852
C        1.1879 -0.3829  0.0000
H        2.0985  0.2306  0.0000
H        1.1184 -1.0093  0.8869
H        1.1184 -1.0093 -0.8869
"""
options['basis'] = '6-31g*' # basis set 
options['low_level'] = 'b3lyp' # level of theory of the environment 
options['high_level'] = 'mp2' # level of theory of the embedded system
options['n_active_atoms'] = 2 # number of active atoms (first n atoms in the geometry string)

# Extra options
options['num_threads']= 8
options['n_virtual_shell'] = 1
options['virtual_projection_basis'] = '6-31g*'

# Run embedding calculation
driver(options)
