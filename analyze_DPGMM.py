#!/usr/bin/env python

"""
Functions to extract information from DPGMM pickle files

(C) Walter Del Pozzo, Archisman Ghosh; 2017 Nov 23
"""

import sys, numpy as np
from scipy.misc import logsumexp
import cPickle as pickle
sys.path.append('../src')

# ---------------
# transformations
# ---------------

# Coordinate conversion
def SphericalToCartesian(d,ra,dec):
  x = d*np.cos(ra)*np.cos(dec)
  y = d*np.sin(ra)*np.cos(dec)
  z = d*np.sin(dec)
  return (x,y,z)

# ---------------
# DPGMM functions
# ---------------

# Logarithm of 3D density from DPGMM reconstruction
def get_log_prob((weights, probs), (d, (ra, dec))):
  return logsumexp([np.log(w) + p.logProb(SphericalToCartesian(d, ra, dec)) for (w, p) in zip(weights, probs)])

# -----------------
# HEALPix utilities
# -----------------

# RA and dec from HEALPix index
def ra_dec_from_ipix(nside, ipix):
  (theta, phi) = hp.pix2ang(nside, ipix)
  return (phi, np.pi/2.-theta)

# ---------------------
# Example with GW170817
# ---------------------

if __name__ == '__main__':

  import healpy as hp  
  import matplotlib.pyplot as plt
  
  # GW170817 result on LHO
  intMixture_file = '/home/archis/Work/cosmo_H0/PE_samples/GW170817/TidalP-18/volume_archis/DPGMM_intMixture.p'
  
  # Load the density reconstruction
  (weights, probs) = pickle.load(open(intMixture_file, 'rb'))
  
  # NGC 4993 coordinates
  ra_NGC4993 = 3.446
  dec_NGC4993 = -0.408
  dist_NGC4993 = 40.
  
  # Plot section of the density through sky location of NGC 4993
  d_arr = np.linspace(10., 100., 1000)
  logp_of_d_arr = np.array([get_log_prob((weights, probs), (dd, (ra_NGC4993, dec_NGC4993))) for dd in d_arr])
  
  plt.figure()
  plt.plot(d_arr, np.exp(logp_of_d_arr))
  plt.xlabel(r'$d_L$ (Mpc)')
  plt.title(r'Section of posterior density through sky location NGC 4993')
  
  # Plot a section of density through approximate distance of NGC 4993
  # NOTE: For illustration only: this step is time-consuming and unnecessarily dumb
  #       since the support of the posterior density is zero over most of the sky
  
  nside = 64 #1024
  logp_on_sky = np.array([get_log_prob((weights, probs), (dist_NGC4993, ra_dec_from_ipix(nside, ipix))) for ipix in xrange(hp.nside2npix(nside))])
  
  hp.mollview(np.exp(logp_on_sky))
  
  plt.show()
