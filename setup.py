from setuptools import setup, find_packages
from distutils.core import setup

# Read in requirements.txt
requirements = open('requirements.txt').readlines()
requirements = [r.strip() for r in requirements]


setup(name='psi_embed',
        version=0.1,
        description='Projection-based embedding calculations in Psi4 using SVD-based partitioning',
        url='https://github.com/danclaudino/PsiEmbed.git',
        author='Daniel Claudino',
        author_email='dclaudino@vt.edu',
        license='MIT',
        #packages=find_packages(where='src'),
        package_dir={'': 'src'},
        packages=[''],

        #packages=setuptools.find_packages(),
        install_requires=requirements,
        include_package_data=True,
        )

