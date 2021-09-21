from setuptools import setup, find_packages

VERSION = '0.0.1' 
DESCRIPTION =  "A package for training and constructing neural networks for the approximation of nucleation collective variables"
LONG_DESCRIPTION = 'README'

# Setting up
setup(
       # the name must match the folder name 'verysimplemodule'
        name="NNucleate", 
        version=VERSION,
        author="Florian Dietrich",
        author_email="ucecfmd@ucl.ac.uk",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=["numpy", "MDAnalysis", "mdtraj", "scipy", "cython"], # add any additional packages that 
        # needs to be installed along with your package. Eg: 'caer'
        
        keywords=['python', 'first package'],
        classifiers= [
            "Development Status :: pre - Alpha",
            "Intended Audience :: Academic",
            "Programming Language :: Python :: 3",
        ]
)