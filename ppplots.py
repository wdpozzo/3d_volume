#!/usr/bin/env python

"""
Script to make p-p plots

(C) Archisman Ghosh, 2015-12-21
"""

import sys, os, glob, re, numpy as np
from scipy import interpolate
from scipy.stats import gaussian_kde
import h5py

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

# ### Input

outfolder = '.'
runfolder = '/home/spxkt1/ROQ/pp_plots_dataseed1915'

try:
  param = sys.argv[1]
except:
  param = 'dchi3'

posfile_list = glob.glob(os.path.join(runfolder, param+"_ROQ", 'GR_injection*dataseed1915', '*.hdf5'))

# ### rc params
plt.rc("lines", markeredgewidth=1)
plt.rc("lines", linewidth=2)
plt.rc('axes', labelsize=22) #24
plt.rc("axes", linewidth=1.0) #2)
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
plt.rc('legend', fontsize=16)
plt.rc('xtick.major', pad=6) #8)
plt.rc('ytick.major', pad=6) #8)
plt.rc('xtick.minor', size=5) #8)
plt.rc('ytick.minor', size=5) #8)
titlefontsize=22

# ### Module for credible level calculations ###

# ### NOTE: This module is reviewed and available on LALSuite in:
# ### lalinference/python/imrtgr_imr_consistency_test.py

class confidence(object):
  def __init__(self, counts):
    # Sort in descending order in frequency
    self.counts_sorted = np.sort(counts.flatten())[::-1]
    # Get a normalized cumulative distribution from the mode
    self.norm_cumsum_counts_sorted = np.cumsum(self.counts_sorted) / np.sum(counts)
    # Set interpolations between heights, bins and levels
    self._set_interp()
  def _set_interp(self):
    self._length = len(self.counts_sorted)
    # height from index
    self._height_from_idx = interpolate.interp1d(np.arange(self._length), self.counts_sorted, bounds_error=False, fill_value=(self.counts_sorted[0],0.))
    # index from height
    self._idx_from_height = interpolate.interp1d(self.counts_sorted[::-1], np.arange(self._length)[::-1], bounds_error=False, fill_value=(self._length,0.))
    # level from index
    self._level_from_idx = interpolate.interp1d(np.arange(self._length), self.norm_cumsum_counts_sorted, bounds_error=False, fill_value=(0.,1.))
    # index from level
    self._idx_from_level = interpolate.interp1d(self.norm_cumsum_counts_sorted, np.arange(self._length), bounds_error=False, fill_value=(0.,self._length))
  def level_from_height(self, height):
    return self._level_from_idx(self._idx_from_height(height))
  def height_from_level(self, level):
    return self._height_from_idx(self._idx_from_level(level))


if __name__ == '__main__':
  
  # ### Read the data and calculate the confidence levels for GR (param == 0.)
  
  if not os.path.isfile("GR_conf_%s.txt"%(param)):
    ev_list = np.array([])
    param_level_GR_list = np.array([])
    
    for posfile in posfile_list:
      
      #ev = int(re.findall('-[0-9]*\.dat', posfile.split('/')[-1])[0][1:-4])
      ev = int(re.findall('-[0-9]*\.hdf5', posfile.split('/')[-1])[0][1:-5])
      
      try:
      
        #possamp = np.genfromtxt(posfile, names=True, dtype=None)
        fileobject = h5py.File(posfile, 'r')
        possamp = fileobject.get('lalinference/lalinference_nest/posterior_samples').value
      
        param_kde = gaussian_kde(possamp[param])
        param_height_GR, = param_kde(0.)
        
        counts = param_kde(np.linspace(0.95*min(possamp[param]), 1.05*max(possamp[param]), 1000))
        param_conf_obj = confidence(counts)
        param_level_GR = param_conf_obj.level_from_height(param_height_GR)
        
        ev_list = np.append(ev_list, ev)
        param_level_GR_list = np.append(param_level_GR_list, param_level_GR)
        
        print "Event ", ev, ":\t", "GR confidence value = ", param_height_GR, "\t", "GR confidence level = ", param_level_GR
      
      except:
        
        print "FAILED: ", posfile.split('/')[-1]
    
    sort_idx = np.argsort(ev_list)
    
    print "Saving: ", 'GR_conf_%s.txt'%(param)
    np.savetxt(os.path.join(outfolder,'GR_conf_%s.txt'%(param)), np.array([ev_list[sort_idx], param_level_GR_list[sort_idx]]).T, fmt='%d %f', header='event GR_level_%s'%(param))
  else:
    param_level_GR_list=np.loadtxt("GR_conf_%s.txt"%(param)).T[1]
  
  # ### Calculate the p-p quantities

  # x-axis the GR confidence level running between 0 and 1
  pp_x = np.linspace(0., 1., 500)
  
  # y-axis is the fraction of events where the GR confidence is below the value on the x-axis
  pp_y = np.array([len(np.where(param_level_GR_list<p_x)[0]) for p_x in pp_x])/float(len(param_level_GR_list))

  # Plot also the theoretical 1-sigma and 2-sigmas curves
  OneSigma = lambda x: np.sqrt(x*(1.-x)/len(param_level_GR_list))
  pp_1sigma = np.vectorize(OneSigma)(pp_x)
  pp_2sigma = 2.*np.vectorize(OneSigma)(pp_x)

  print np.max(abs(pp_y-pp_x))
  # ### Plotting

  plt.figure(figsize=(8,8))
  plt.plot(pp_x, pp_y, 'g-')
  plt.plot(pp_x, pp_x, 'r--')
  plt.fill_between(pp_x, pp_x+pp_1sigma, pp_x-pp_1sigma, alpha=.2, color='grey')
  plt.fill_between(pp_x, pp_x+pp_2sigma, pp_x-pp_2sigma, alpha=.1, color='grey')
  plt.xlabel('GR confidence level')
  plt.ylabel('Fraction of events below level')
  plt.title(r'p-p plot for $%s$'%(param[0]+"\\"+param[1:-1]+"_"+param[-1]), size=titlefontsize)

  plt.savefig(os.path.join(outfolder,'pp_%s.png'%(param)))
  plt.savefig(os.path.join(outfolder,'pp_%s.pdf'%(param)))
