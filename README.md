# 3d_volume
DPGMM implementation for 3D, 2D and 1D density estimate from MCMC samples. 
Requirements:

numpy
scipy
healpy
lal
cython

Based on the DPGMM mixture model implementation of https://github.com/thaines/helit/tree/master/dpgmm 

Installation
python setup.py build_ext --inplace

HOT TO SETUP A VIRTUALENV ENVIRONMENT ON CLUSTERS AND INSTALLING THE REQUIRED PACKAGES THERE

virtualenv <destination folder>
pip install numpy
pip install scipy
pip install healpy
pip install cython

After logging out, one will need to activate the virtual environment by typing:

source <destination folder>/activate