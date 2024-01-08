# NNucleate

This is the release version of NNucleate. NNucleate is a Python package developed for the training of GNN-based models approximating computationally expensive collective variables (CV). The primary intended application is enhanced sampling simulations of nucleation events. These types of simulations are typically limited by the computational cost of their CVs, since they require a computationally expensive, differentiable degree of order in the system to be calculated on the fly. 

When using this code please cite: 

*Dietrich, Florian, Xavier Rosas Advincula, Gianpaolo Gobbo, Michael Bellucci, and Matteo Salvalaglio. "Machine Learning Nucleation Collective Variables with Graph Neural Networks." The Journal of Chemical Theory and Computation, (2023),* [Link](https://pubs.acs.org/doi/10.1021/acs.jctc.3c00722?ref=pdf)

## Installation

`pip install git+https://github.com/mme-ucl/NNucleate.git`

## Documentation
For code documentation and a tutorial visit:
[https://flofega.github.io/NNucleate/](https://flofega.github.io/NNucleate/)

Examples can be found in the examples folder.
The development of this package occurs at [https://github.com/Flofega/NNucleate](https://github.com/Flofega/NNucleate).
