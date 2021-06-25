# MOMoGP
PyTorch implementation of MOMoGP, proposed in 

Z. Yu, M. Zhu, M. Trapp, A. Skryagin, K. Kersting, 
**Leveraging Probabilistic Circuits for Nonparametric Multi-Output Regression**,
*UAI 2021*.

# Setup
This will clone the repo, install a python virtual env (requires pythn 3.6), the required packages, and will download some datasets.

    git clone https://github.com/minimrbanana/MOMoGP
    ./setup.sh

# Demos
To illustrate the usage of the code:

    source ./venv_momogp/bin/activate
    python run_MOMoGP.py --data=pakinsons

"pakinsons" can be replaced with "scm20d" or "wind" or "energy" or "usflight".
If not specified, the corresponding hyperparameters are set by default values.


