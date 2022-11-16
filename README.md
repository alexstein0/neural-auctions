# neural-auctions

This is an ICLR sumbission-ready version of our code.  
A complete de-anonymized version will be released and maintained on GitHub after the review process.

In this repository, models can by trained, tested, and inverted. These three activities correspond to three separate files.  
We use the Hydra package for config management and output organization. This package also allows for command line arguments.

## Getting started
We developed and tested this code with Python 3.10.4. The required packages are listed in [requirements.txt](requirements.txt). 
The `almost-unique-id` package contained in this repository must also be installed by running `$ pip install -e .` from within the package's top level directory.

## Training and Inverting
To reproduce the training runs used in the paper, run any/all the commands in [training.sh](launch/training.sh). In the paper, we report results for 10 trials of each. Additionally, to reproduce the inversion results presented in the paper, run any/all the commands in [inverting.sh](launch/inverting.sh).