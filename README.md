# NNucleate

This is a preliminary code collection of the NNCV approximation project.
To train a model:
- (Optionally) augment  the trajectory of interest with functions from data_augmentation.py (evenly_augment).
-  Create an instance of the CVTrajectory object using the trajectory file  that you want to use for training.
-  Create  a dataloader wrapping the CVTrajectory object.
-  Create a model using the NNCV class
-  Write a training loop using the desired train class and test

## Installation

`pip install git+https://github.com/Flofega/NNucleate.git`

## Documentation

www.flofega.github.io/nnucleate/
