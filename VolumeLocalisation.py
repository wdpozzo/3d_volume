#! /usr/bin/env python

from __future__ import division
import os, sys, numpy as np
import cPickle as pickle
from dpgmm import *
import copy
import healpy as hp
from scipy.misc import logsumexp
import optparse as op
import lal
from pylal.xlal.datatypes.ligotimegps import LIGOTimeGPS
from pylal import SimInspiralUtils
import multiprocessing as mp
import copy_reg
import types
import cumulative
import matplotlib
import time
matplotlib.use("Agg")

def _pickle_method(m):
    if m.im_self is None:
        return getattr, (m.im_class, m.im_func.func_name)
    else:
        return getattr, (m.im_self, m.im_func.func_name)

copy_reg.pickle(types.MethodType, _pickle_method)

# ---------------------
# DPGMM posterior class
# ---------------------

class DPGMMSkyPosterior(object):
    """
        Dirichlet Process Gaussian Mixture model class
        input parameters:
        
        posterior_samples: posterior samples for which the density estimate needs to be calculated
        
        dimension: the dimensionality of the problem. default = 3
        
        max_stick: maximum number of mixture components. default = 16
        
        nside: nside for the healpix map on the sphere. default = 8
        
        dist_bins: number of bins in the radial (distance) dimension. default = 10
        
        dist_max: maximum radial distance to consider. default = 218 Mpc
        
        nthreads: number of multiprocessing pool workers to use. default = multiprocessing.cpu_count()
        
        injection: the injection file.
        
        catalog: the galaxy catalog for the ranked list of galaxies
        """
    def __init__(self,posterior_samples,dimension=3,max_sticks=16,nside=8,dist_bins=10,dist_max=218,nthreads=None,injection=None,catalog=None):
        self.posterior_samples = np.array(posterior_samples)
        self.dims = 3
        self.max_sticks = max_sticks
        if nthreads == None:
            self.nthreads = mp.cpu_count()
        else:
            self.nthreads = nthreads
        self.nside= np.int(nside)
        self.dist_bins = dist_bins
        self.dist_max = dist_max
        self.pool = mp.Pool(self.nthreads)
        self.injection = injection
        self.distance_max=dist_max
        self.catalog = None
        if catalog is not None:
            self.catalog = readGC(catalog)#read_galaxy_catalog(catalog)
        self._initialise_healpix()
        self._initialise_distance_grid()

    def _initialise_dpgmm(self):
        self.model = DPGMM(self.dims)
        for point in self.posterior_samples:
            self.model.add(point)
        dd = np.diff(self.d_grid)[0]
        dA = hp.nside2pixarea(self.nside)
        self.model.setPrior(scale = dd*dA)
        self.model.setThreshold(1e-4)
        self.model.setConcGamma(1,1)

    def _initialise_healpix(self):
        self.npix = np.int(hp.nside2npix(nside))
    
    def _initialise_distance_grid(self):
        a = np.maximum(0.9*samples[:,0].min(),1.0)
        b = np.minimum(1.1*samples[:,0].max(),self.distance_max)
        self.d_grid = np.linspace(a,b,n_dist)

    def compute_dpgmm(self):
        self._initialise_dpgmm()
        solve_args = [(nc, self.model) for nc in xrange(1, self.max_sticks+1)]
        solve_results = self.pool.map(solve_dpgmm, solve_args)
        self.scores = np.array([r[1] for r in solve_results])
        self.model = (solve_results[self.scores.argmax()][-1])
        print "best model has ",self.scores.argmax()+1,"components"
        self.density = self.model.intMixture()

    def rank_galaxies(self):
        sys.stderr.write("Ranking the galaxies: computing log posterior for %d galaxies\n"%(self.catalog.shape[0]))
        jobs = ((self.density,self.nside,di,rai,deci) for di,rai,deci in zip(self.catalog[:,2],self.catalog[:,0],self.catalog[:,1]))
        results = self.pool.imap(logPosterior ,jobs,  chunksize = np.int(self.catalog.shape[0]/ (self.nthreads * 16)))
        logProbs = np.array([r for r in results])

        idx = ~np.isnan(logProbs)
        self.ranked_probability = logProbs[idx]
        self.ranked_ra = self.catalog[idx,0]
        self.ranked_dec = self.catalog[idx,1]
        self.ranked_dl = self.catalog[idx,2]

        order = self.ranked_probability.argsort()[::-1]
        
        self.ranked_probability = self.ranked_probability[order]
        self.ranked_ra = self.ranked_ra[order]
        self.ranked_dec = self.ranked_dec[order]
        self.ranked_dl = self.ranked_dl[order]

    def logPosterior(self,d,ra,dec):
        theta,phi = eq2ang(ra,dec)
        ipix = hp.pixelfunc.ang2pix(self.nside,theta,phi, nest=True)
        logPs = [np.log(self.density[0][ind])+prob.logProb(d*np.array(hp.pix2vec(self.nside, ipix, nest=True))) for ind,prob in enumerate(self.density[1])]
        return logsumexp(logPs)+2.0*np.log(d)+np.log(np.abs(np.sin(theta)))

    def Posterior(self,d,ra,dec):
        theta,phi = eq2ang(ra,dec)
        ipix = hp.pixelfunc.ang2pix(self.nside,theta,phi, nest=True)
        Ps = [self.density[0][ind]*prob.prob(d*np.array(hp.pix2vec(self.nside, ipix, nest=True))) for ind,prob in enumerate(self.density[1])]
        return reduce(np.sum,Ps)*np.sin(theta)*d**2
    
    def evaluate_volume_map(self):
        sys.stderr.write("computing log posterior for %d grid poinds\n"%(self.npix*self.dist_bins))
        sample_args = ((self.density,d,ipix,self.nside) for d in self.d_grid for ipix in np.arange(self.npix))
        results = self.pool.imap(sample_volume, sample_args, chunksize = np.int(self.npix*self.dist_bins/ (self.nthreads * 16)))
        self.log_volume_map = np.array([r for r in results]).reshape(len(self.d_grid),self.npix)
        self.volume_map = np.exp(self.log_volume_map)
        dA = hp.nside2pixarea(self.nside)
        dd = np.diff(self.d_grid)[0]
        self.volume_map/=(self.volume_map*dd*dA).sum()

    def evaluate_sky_map(self):
        dd = np.diff(self.d_grid)[0]
        dA = hp.nside2pixarea(self.nside)
        # implementing a trapezoidal rule
        N = self.dist_bins
        left_log_sum = logsumexp(self.log_volume_map[1:N-1,:], axis=0)
        right_log_sum = logsumexp(self.log_volume_map[0:N-2,:], axis=0)
        self.log_skymap = np.logaddexp(left_log_sum,right_log_sum)+np.log(dd/2.)
#        # trapezoidal rule done line above
        self.log_skymap-= logsumexp(self.log_skymap)+np.log(dA)
        self.skymap = np.exp(self.log_skymap)
        self.skymap/=(self.skymap*dA).sum()

    def evaluate_distance_map(self):
        dA = hp.nside2pixarea(self.nside)
        self.log_distance_map = logsumexp(self.log_volume_map, axis=1) + np.log(dA)
        self.distance_map = np.exp(self.log_distance_map)
        self.distance_map/=(self.distance_map*np.diff(self.d_grid)[0]).sum()

    def ConfidenceVolume(self, adLevels):
        # create a normalized cumulative distribution
        self.log_volume_map_sorted = np.sort(self.log_volume_map.flatten())[::-1]
        self.log_volume_map_cum = cumulative.fast_log_cumulative(self.log_volume_map_sorted)
        
        # find the indeces  corresponding to the given CLs
        adLevels = np.ravel([adLevels])
        args = [(self.log_volume_map_sorted,self.log_volume_map_cum,level) for level in adLevels]
        adHeights = self.pool.map(FindHeights,args)
        self.heights = {str(lev):hei for lev,hei in zip(adLevels,adHeights)}
        dd = np.diff(self.d_grid)[0]
        dA = hp.nside2pixarea(self.nside)
        volumes = []
        for height in adHeights:
            (index_d, index_hp) = np.where(self.log_volume_map>height)
            volumes.append(np.sum([self.d_grid[i_d]**2. * dd * dA for i_d in index_d]))
        self.volume_confidence = np.array(volumes)

        if self.injection!=None:
            ra,dec = self.injection.get_ra_dec()
            distance = self.injection.distance
            logPval = self.logPosterior(distance,ra,dec)
            confidence_level = np.exp(self.log_volume_map_cum[np.abs(self.log_volume_map_sorted-logPval).argmin()])
            height = FindHeights((self.log_volume_map_sorted,self.log_volume_map_cum,confidence_level))
            (index_d, index_hp) = np.where(self.log_volume_map >= height)
            searched_volume = np.sum([self.d_grid[i_d]**2. * dd * dA for i_d in index_d])
            self.injection_volume_confidence = confidence_level
            self.injection_volume_height = height
            return self.volume_confidence,(confidence_level,searched_volume)

        del self.log_volume_map_sorted
        del self.log_volume_map_cum
        return self.volume_confidence,None

    def ConfidenceArea(self, adLevels):
        
        # create a normalized cumulative distribution
        self.log_skymap_sorted = np.sort(self.log_skymap.flatten())[::-1]
        self.log_skymap_cum = cumulative.fast_log_cumulative(self.log_skymap_sorted)
        # find the indeces  corresponding to the given CLs
        adLevels = np.ravel([adLevels])
        args = [(self.log_skymap_sorted,self.log_skymap_cum,level) for level in adLevels]
        adHeights = self.pool.map(FindHeights,args)

        dA = hp.nside2pixarea(self.nside, degrees=True)
        areas = []
        for height in adHeights:
            (index_hp,) = np.where(self.log_skymap>height)
            areas.append(len(index_hp)*dA)
        self.area_confidence = np.array(areas)
        
        if self.injection!=None:
            ra,dec = self.injection.get_ra_dec()
            theta,phi = eq2ang(ra,dec)
            ipix = hp.pixelfunc.ang2pix(self.nside,theta,phi, nest=True)
            logPval = self.log_skymap[ipix]
            confidence_level = np.exp(self.log_skymap_cum[np.abs(self.log_skymap_sorted-logPval).argmin()])
            height = FindHeights((self.log_skymap_sorted,self.log_skymap_cum,confidence_level))
            (index_hp,) = np.where(self.log_skymap >= height)
            searched_area = len(index_hp)*dA
            return self.area_confidence,(confidence_level,searched_area)

        del self.log_skymap_sorted
        del self.log_skymap_cum
        return self.area_confidence,None

    def ConfidenceDistance(self, adLevels):
        dd = np.diff(self.d_grid)[0]
        cumulative_distribution = np.cumsum(self.distance_map*dd)
        distances = []
        for cl in adLevels:
            idx = np.abs(cumulative_distribution-cl).argmin()
            distances.append(self.d_grid[idx])
        self.distance_confidence = np.array(distances)

        if self.injection!=None:
            idx = np.abs(self.injection.distance-self.d_grid).argmin()
            confidence_level = cumulative_distribution[idx]
            searched_distance = self.d_grid[idx]
            return self.distance_confidence,(confidence_level,searched_distance)

        return self.distance_confidence,None

# ---------------
# DPGMM functions
# ---------------

def log_cdf(logpdf):
    """
    compute the log cdf from the  log pdf
    
    cdf_i = \sum_i pdf
    log cdf_i = log(\sum_i \exp pdf)
    
    """
    logcdf = np.zeros(len(logpdf))
    logcdf[0] = logpdf[0]
    for j in xrange(1,len(logpdf)):
        logcdf[j]=np.logaddexp(logcdf[j-1],logpdf[j])
    return logcdf-logcdf[-1]

def Posterior(args):
    density,nside,d,ra,dec = args
    theta,phi = eq2ang(ra,dec)
    ipix = hp.pixelfunc.ang2pix(nside,theta,phi, nest=True)
    Ps = [density[0][ind]*prob.prob(d*np.array(hp.pix2vec(nside, ipix, nest=True))) for ind,prob in enumerate(density[1])]
    return reduce(np.add,Ps)*np.sin(theta)*d**2

def logPosterior(args):
    density,nside,d,ra,dec = args
    theta,phi = eq2ang(ra,dec)
    ipix = hp.pixelfunc.ang2pix(nside,theta,phi, nest=True)
    logPs = [np.log(density[0][ind])+prob.logProb(d*np.array(hp.pix2vec(nside, ipix, nest=True))) for ind,prob in enumerate(density[1])]
    return logsumexp(logPs)+2.0*np.log(d)+np.log(np.abs(np.sin(theta)))

def sample_volume(args):
    (dpgmm,d,ipix,nside) = args
    theta,phi = hp.pix2ang(nside,ipix, nest=True)
    logPs = [np.log(dpgmm[0][ind])+prob.logProb(d*np.array(hp.pix2vec(nside, ipix, nest=True))) for ind,prob in enumerate(dpgmm[1])]
    return logsumexp(logPs)+2.0*np.log(d)+np.log(np.abs(np.sin(theta)))


def solve_dpgmm(args):
    (nc, model) = args
    for _ in xrange(nc-1): model.incStickCap()
    try:
        it = model.solve(iterCap=1024)
        return (model.stickCap, model.nllData(), model)
    except:
        return (model.stickCap, -np.inf, model)

def sample_dpgmm(args):
    print 'sampling component ..'
    # parse the arguments
    ((w, p), (d_grid, npix)) = args
    # get the number of pixels for the healpy nside
    # calculate the log probablity at the cartesian coordinates
    logp = np.array([[p.logProb(d*np.array(hp.pix2vec(np.int(nside), pix, nest=True))) for pix in xrange(npix)] for d in d_grid])
    # multiply by weight of component and return
    return np.log(w) + logp

# --------
# jacobian
# --------

def log_jacobian(dgrid, nside):
  # get the number of pixels for the healpy nside
  npix = np.int(hp.nside2npix(nside))
  # calculate the jacobian on the d_grid, copy over for the required number of pixels, appropriately reshape the array and return
  return np.array([2.*np.log(d) for d in dgrid]*npix).reshape(npix,-1).T

# -----------------------
# confidence calculations
# -----------------------

def FindHeights(args):
    (sortarr,cumarr,level) = args
    return sortarr[np.abs(cumarr-np.log(level)).argmin()]

def FindHeightForLevel(inLogArr, adLevels):
    # flatten and create reversed sorted list
    adSorted = np.sort(inLogArr.flatten())[::-1]
    # create a normalized cumulative distribution
    adCum = np.array([logsumexp(adSorted[:i+1]) for i in xrange(len(adSorted))])
    adCum -= adCum[-1]
    # find values closest to levels
    adHeights = []
    adLevels = np.ravel([adLevels])
    for level in adLevels:
        idx = (np.abs(adCum-np.log(level))).argmin()
        adHeights.append(adSorted[idx])
    adHeights = np.array(adHeights)
    return adHeights

def FindLevelForHeight(inLogArr, logvalue):
    # flatten and create reversed sorted list
    adSorted = np.sort(inLogArr.flatten())[::-1]
    # create a normalized cumulative distribution
    adCum = np.array([logsumexp(adSorted[:i+1]) for i in xrange(len(adSorted))])
    adCum -= adCum[-1]
    # find index closest to value
    idx = (np.abs(adSorted-logvalue)).argmin()
    return np.exp(adCum[idx])

#---------
# utilities
#---------

def eq2ang(ra, dec):
    """
    convert equatorial ra,dec in radians to angular theta, phi in radians
    parameters
    ----------
    ra: scalar or array
        Right ascension in radians
    dec: scalar or array
        Declination in radians
    returns
    -------
    theta,phi: tuple
        theta = pi/2-dec*D2R # in [0,pi]
        phi   = ra*D2R       # in [0,2*pi]
    """
    phi = ra
    theta = np.pi/2. - dec
    return theta, phi

def ang2eq(theta, phi):
    """
    convert angular theta, phi in radians to equatorial ra,dec in radians
    ra = phi*R2D            # [0,360]
    dec = (pi/2-theta)*R2D  # [-90,90]
    parameters
    ----------
    theta: scalar or array
        angular theta in radians
    phi: scalar or array
        angular phi in radians
    returns
    -------
    ra,dec: tuple
        ra  = phi*R2D          # in [0,360]
        dec = (pi/2-theta)*R2D # in [-90,90]
    """
    
    ra = phi
    dec = np.pi/2. - theta
    return ra, dec

def cartesian_to_spherical(vector):
    """Convert the Cartesian vector [x, y, z] to spherical coordinates [r, theta, phi].

    The parameter r is the radial distance, theta is the polar angle, and phi is the azimuth.


    @param vector:  The Cartesian vector [x, y, z].
    @type vector:   numpy rank-1, 3D array
    @return:        The spherical coordinate vector [r, theta, phi].
    @rtype:         numpy rank-1, 3D array
    """

    # The radial distance.
    r = np.linalg.norm(vector)

    # Unit vector.
    unit = vector / r

    # The polar angle.
    theta = np.arccos(unit[2])

    # The azimuth.
    phi = np.arctan2(unit[1], unit[0])

    # Return the spherical coordinate vector.
    return r, theta, phi


def spherical_to_cartesian(spherical_vect):
    """Convert the spherical coordinate vector [r, theta, phi] to the Cartesian vector [x, y, z].

    The parameter r is the radial distance, theta is the polar angle, and phi is the azimuth.


    @param spherical_vect:  The spherical coordinate vector [r, theta, phi].
    @type spherical_vect:   3D array or list
    @param cart_vect:       The Cartesian vector [x, y, z].
    @type cart_vect:        3D array or list
    """
    cart_vect = np.zeros(3)
    # Trig alias.
    sin_theta = np.sin(spherical_vect[1])

    # The vector.
    cart_vect[0] = spherical_vect[0] * np.cos(spherical_vect[2]) * sin_theta
    cart_vect[1] = spherical_vect[0] * np.sin(spherical_vect[2]) * sin_theta
    cart_vect[2] = spherical_vect[0] * np.cos(spherical_vect[1])
    return cart_vect

def readGC(file,standard_cosmology=True):

    ra,dec,zs,zp =[],[],[],[]
    dl = []
    with open(file,'r') as f:
        if standard_cosmology: omega = lal.CreateCosmologicalParameters(0.7,0.3,0.7,-1.0,0.0,0.0)
        for line in f:
            fields = line.split(None)
            if 0.0 < np.float(fields[40]) > 0.0 or np.float(fields[41]) > 0.0:
                if not(standard_cosmology):
                    h = np.random.uniform(0.5,1.0)
                    om = np.random.uniform(0.0,1.0)
                    ol = 1.0-om
                    omega = lal.CreateCosmologicalParameters(h,om,ol,-1.0,0.0,0.0)
                
                ra.append(np.float(fields[0]))
                dec.append(np.float(fields[1]))
                zs.append(np.float(fields[40]))
                zp.append(np.float(fields[41]))
                if not(np.isnan(zs[-1])):
                    dl.append(lal.LuminosityDistance(omega,zs[-1]))
                elif not(np.isnan(zp[-1])):
                    dl.append(lal.LuminosityDistance(omega,zp[-1]))
                else:
                    dl.append(-1)
        f.close()
    return np.column_stack((np.radians(np.array(ra)),np.radians(np.array(dec)),np.array(dl)))

def read_galaxy_catalog(file):
    ra,dec,dl =[],[],[]
    with open(file,'r') as f:
        for j,line in enumerate(f):
            fields = line.split(None)
            ra.append(np.float(fields[0]))
            dec.append(np.float(fields[1]))
            dl.append(np.float(fields[2]))
        f.close()
    return np.column_stack((ra,dec,dl))
#---------
# plotting
#---------

fig_width_pt = 3*246.0  # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0/72.27               # Convert pt to inch
golden_mean = (np.sqrt(5)-1.0)/2.0         # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt  # width in inches
fig_height = fig_width*golden_mean      # height in inches
fig_size =  [fig_width,fig_height]
params = {'backend': 'TkAgg',
    'axes.labelsize': 32,
    'text.fontsize': 28,
    'legend.fontsize': 28,
    'xtick.labelsize': 22,
    'ytick.labelsize': 22,
    'text.usetex': False,
    'figure.figsize': fig_size}


#-------------------
# start the program
#-------------------

if __name__=='__main__':
    parser = op.OptionParser()
    parser.add_option("-i", "--input", type="string", dest="input", help="Input file")
    parser.add_option("--inj",type="string",dest="injfile",help="injection file",default=None)
    parser.add_option("-o", "--output", type="string", dest="output", help="Output file")
    parser.add_option("--nside", type="int", dest="nside", help="nside of healpix")
    parser.add_option("--dr", type="int", dest="n_dist", help="number of bins in distance")
    parser.add_option("--dmax", type="float", dest="dmax", help="maximum distance (Mpc)")
    parser.add_option("--max-stick", type="int", dest="max_stick", help="maximum number of gaussian components")
    parser.add_option("-e", type="int", dest="event_id", help="event ID")
    parser.add_option("--threads", type="int", dest="nthreads", help="number of threads to spawn", default=None)
    parser.add_option("--catalog", type="string", dest="catalog", help="galaxy catalog to use", default=None)
    parser.add_option("--plots", type="string", dest="plots", help="produce plots", default=False)
    parser.add_option("-N", type="int", dest="ranks", help="number of ranked galaxies to list in output", default=1000)
    parser.add_option("--nsamps", type="int", dest="nsamps", help="number of posterior samples to utilise", default=None)
    (options, args) = parser.parse_args()

    CLs = [0.1,0.2,0.25,0.3,0.4,0.5,0.6,0.68,0.7,0.75,0.8,0.9]
    input_file = options.input
    injFile = options.injfile
    eventID = options.event_id
    out_dir = options.output
    nside = options.nside
    n_dist = options.n_dist
  
    os.system('mkdir -p %s'%(out_dir))
  
    if injFile!=None:
        injections = SimInspiralUtils.ReadSimInspiralFromFiles([injFile])
        injection = injections[0]
        (ra_inj, dec_inj) = injection.get_ra_dec()
        tc = injection.get_time_geocent()
        GPSTime=lal.LIGOTimeGPS(str(tc))
        gmst_rad_inj = lal.GreenwichMeanSiderealTime(GPSTime)
        dist_inj = injection.distance
        print 'injection values -->',dist_inj,ra_inj,dec_inj,tc
    else:
        injection = None

    samples = np.genfromtxt(input_file,names=True)
    print  samples.dtype.names
    if "dist" in samples.dtype.names:
        samples = np.column_stack((samples["dist"],samples["ra"],samples["dec"],samples["time"]))
    else:
        samples = np.column_stack((samples["distance"],samples["ra"],samples["dec"],samples["time"]))

    npix = np.int(hp.nside2npix(nside))
    print 'The number of grid points in the sky is :',hp.nside2npix(nside),'resolution = ',hp.nside2pixarea(nside, degrees = True), ' deg^2'
    print 'The number of grid points in distance is :',n_dist,'resolution = ',(options.dmax-1.0)/n_dist,' Mpc'
    print 'Total grid size is :',n_dist*hp.nside2npix(nside)
    print 'Volume resolution is :',hp.nside2pixarea(nside)*(options.dmax-1.0)/n_dist,' Mpc^3'

    samps = []
    gmst_rad = []

    if options.nsamps is not None:
        idx = np.random.choice(range(0,len(samples[:,0])),size=options.nsamps)
    else: 
        idx = range(0,len(samples[:,0]))

    for k in xrange(len(samples[idx,0])):
        GPSTime=lal.LIGOTimeGPS(samples[k,3])
        gmst_rad.append(lal.GreenwichMeanSiderealTime(GPSTime))
#        theta,phi = eq2ang(samples[k,1]-gmst_rad[-1],samples[k,2])
        theta,phi = eq2ang(samples[k,1],samples[k,2])
        ipix = hp.pixelfunc.ang2pix(np.int(nside),theta,phi, nest=True)
        samps.append(samples[k,0]*np.array(hp.pix2vec(np.int(nside), ipix, nest=True)))

    dpgmm = DPGMMSkyPosterior(samps,dimension=3,
                              max_sticks=options.max_stick,
                              nside=options.nside,
                              dist_bins=options.n_dist,
                              dist_max=options.dmax,
                              nthreads=options.nthreads,
                              injection=injection,
                              catalog=options.catalog)
    dpgmm.compute_dpgmm()

    if dpgmm.catalog is not None:
        dpgmm.rank_galaxies()

        np.savetxt(os.path.join(options.output,'galaxy_ranks.txt'),
                   np.array([np.degrees(dpgmm.ranked_ra[:options.ranks]),np.degrees(dpgmm.ranked_dec[:options.ranks]),dpgmm.ranked_dl[:options.ranks],dpgmm.ranked_probability[:options.ranks]]).T,
                   fmt='%.9f\t%.9f\t%.9f\t%.9f\t',
                   header='ra[deg]\tdec[deg]\tDL[Mpc]\tlogposterior')

    dpgmm.evaluate_volume_map()
    dpgmm.evaluate_sky_map()
    dpgmm.evaluate_distance_map()
    volumes,searched_volume = dpgmm.ConfidenceVolume(CLs)
    areas,searched_area = dpgmm.ConfidenceArea(CLs)
    distances,searched_distance = dpgmm.ConfidenceDistance(CLs)

    if dpgmm.catalog is not None:
        number_of_galaxies = np.zeros(len(CLs),dtype=np.int)

        for i,CL in enumerate(CLs):
            threshold = dpgmm.heights[str(CL)]
            (k,) = np.where(dpgmm.ranked_probability>threshold)
            number_of_galaxies[i] = len(k)

        np.savetxt(os.path.join(options.output,'galaxy_in_confidence_regions.txt'), np.array([CLs,number_of_galaxies]).T, fmt='%.2f\t%d')

        if dpgmm.injection!=None:
            threshold = dpgmm.injection_volume_height
            (k,) = np.where(dpgmm.ranked_probability>threshold)
            number_of_galaxies = len(k)
            with open(os.path.join(options.output,'searched_galaxies.txt'),'w') as f:
                f.write('%.5f\t%d\n'%(dpgmm.injection_volume_confidence,number_of_galaxies))
                f.close()

    if options.plots:
        import matplotlib.pyplot as plt
        hp.visufunc.mollview(dpgmm.log_skymap)
        if injFile!=None:
            hp.visufunc.projscatter(eq2ang(ra_inj,dec_inj), c='y', s=256, marker='*')
        plt.savefig(os.path.join(options.output,'sky_map.pdf'),bbox_inches='tight')
        plt.figure()
        plt.plot(dpgmm.d_grid,dpgmm.distance_map,color="k",linewidth=2.0)
        plt.hist(samples[:,0],bins=dpgmm.d_grid,normed=True,facecolor="0.9")
        if injFile!=None: plt.axvline(dist_inj,color="k",linestyle="dashed")
        plt.xlabel(r"$\mathrm{Distance/Mpc}$")
        plt.ylabel(r"$\mathrm{probability}$ $\mathrm{density}$")
        plt.savefig(os.path.join(options.output,'distance_posterior.pdf'),bbox_inches='tight')
    np.savetxt(os.path.join(options.output,'confidence_levels.txt'), np.array([CLs, volumes, areas, distances]).T, fmt='%.2f\t%f\t%f\t%f')
    if dpgmm.injection!=None: np.savetxt(os.path.join(options.output,'searched_quantities.txt'), np.array([searched_volume,searched_area,searched_distance]), fmt='%s\t%s')

    # dist_inj,ra_inj,dec_inj,tc
    if injFile is not None:
        gmst_deg = np.mod(np.degrees(gmst_rad_inj), 360)
        lon_cen = lon_inj = np.degrees(ra_inj) - gmst_deg
        lat_cen = lat_inj = np.degrees(dec_inj)
    else:
        gmst_deg = np.mod(np.degrees(np.array(gmst_rad)), 360)
        lon_cen = np.degrees(np.mean(samples[:,1])) - np.mean(gmst_deg)
        lat_cen = np.degrees(np.mean(samples[:,2]))

    lon_samp = np.degrees(samples[:,1]) - gmst_deg
    lat_samp = np.degrees(samples[:,2])

    (theta_map, phi_map) = hp.pix2ang(dpgmm.nside, range(hp.nside2npix(nside)), nest=True)
    ra_map,dec_map = ang2eq(theta_map, phi_map)
    lon_map = np.degrees(ra_map) - np.mean(gmst_deg)
    lat_map = np.degrees(dec_map)
    if options.plots:
        sys.stderr.write("producing sky maps \n")
        plt.figure()
        plt.plot(np.arange(1,dpgmm.max_sticks+1),dpgmm.scores,'.')
        plt.xlabel(r"$\mathrm{number}$ $\mathrm{of}$ $\mathrm{components}$")
        plt.ylabel(r"$\mathrm{marginal}$ $\mathrm{likelihood}$")
        plt.savefig(os.path.join(out_dir, 'scores.pdf'))
        
        from mpl_toolkits.basemap import Basemap

        plt.figure()
        m = Basemap(projection='moll', lon_0=round(lon_cen, 2), lat_0=0, resolution='c')
        m.drawcoastlines(linewidth=0.5, color='0.5')
        m.drawparallels(np.arange(-90,90,30), labels=[1,0,0,0], labelstyle='+/-', linewidth=0.1, dashes=[1,1], alpha=0.5)
        m.drawmeridians(np.arange(0,360,60), linewidth=0.1, dashes=[1,1], alpha=0.5)
        m.drawmapboundary(linewidth=0.5, fill_color='white')
        plt.scatter(*m(lon_samp, lat_samp), color='g', s=0.1, lw=0)
        S = plt.scatter(*m(lon_map, lat_map), c=dpgmm.log_skymap, cmap='RdPu', s=2, lw=0)
        if injFile is not None: plt.scatter(*m(lon_inj, lat_inj), color='r', s=500, marker='+')
        cbar = m.colorbar(S,location='bottom',pad="5%")
        cbar.set_label(r"$\log(\mathrm{Probability})$")
        plt.savefig(os.path.join(out_dir, 'marg_log_sky_%d.pdf'%(eventID)))

        plt.figure()
        m = Basemap(projection='aeqd', lon_0=round(lon_cen, 2), lat_0=0, resolution='c')
        m.drawcoastlines(linewidth=0.5, color='0.5')
        m.drawparallels(np.arange(-90,90,30), labels=[1,0,0,0], labelstyle='+/-', linewidth=0.1, dashes=[1,1], alpha=0.5)
        m.drawmeridians(np.arange(0,360,60), linewidth=0.1, dashes=[1,1], alpha=0.5)
        m.drawmapboundary(linewidth=0.5, fill_color='white')
        plt.scatter(*m(lon_samp, lat_samp), color='g', s=0.1, lw=0)
        S = plt.scatter(*m(lon_map, lat_map), c=dpgmm.log_skymap, cmap='RdPu', s=2, lw=0)
        if injFile is not None: plt.scatter(*m(lon_inj, lat_inj), color='r', s=500, marker='+')
        cbar = m.colorbar(S,location='bottom',pad="5%")
        cbar.set_label(r"$\log(\mathrm{Probability})$")
        plt.savefig(os.path.join(out_dir, 'marg_log_sky_aed_%d.pdf'%(eventID)))
        
        plt.figure()
        m = Basemap(projection='moll', lon_0=round(lon_cen, 2), lat_0=0, resolution='c')
        m.drawcoastlines(linewidth=0.5, color='0.5')
        m.drawparallels(np.arange(-90,90,30), labels=[1,0,0,0], labelstyle='+/-', linewidth=0.1, dashes=[1,1], alpha=0.5)
        m.drawmeridians(np.arange(0,360,60), linewidth=0.1, dashes=[1,1], alpha=0.5)
        m.drawmapboundary(linewidth=0.5, fill_color='white')
        plt.scatter(*m(lon_samp, lat_samp), color='g', s=0.1, lw=0)
        S = plt.scatter(*m(lon_map, lat_map), c=dpgmm.skymap, cmap='RdPu', s=2, lw=0) #OrRd
        if injFile is not None: plt.scatter(*m(lon_inj, lat_inj), color='r', s=500, marker='+')
        cbar = m.colorbar(S,location='bottom',pad="5%")
        cbar.set_label(r"$\mathrm{Probability}$ $\mathrm{density}$")
        plt.savefig(os.path.join(out_dir, 'marg_sky_%d.pdf'%(eventID)))

        if options.plots:
            if options.catalog:
                sys.stderr.write("producing 3 dimensional maps\n")
                theta,phi = eq2ang(dpgmm.ranked_ra,dpgmm.ranked_dec)
                # Create a sphere
                x = dpgmm.ranked_dl*np.sin(theta)*np.cos(phi)
                y = dpgmm.ranked_dl*np.sin(theta)*np.sin(phi)
                z = dpgmm.ranked_dl*np.cos(theta)

                imax = dpgmm.ranked_probability.argmax()
                threshold = dpgmm.heights['0.5']
                (k,) = np.where(dpgmm.ranked_probability>threshold)
                MIN = dpgmm.d_grid[0]
                MAX = dpgmm.d_grid[-1]
                sys.stderr.write("%d galaxies above threshold, plotting\n"%(len(k)))
                myfig = plt.figure()
                from mpl_toolkits.mplot3d import Axes3D
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter([0.0],[0.0],[0.0],c='k',s=500,marker=r'$\bigoplus$',edgecolors='none')
                S = ax.scatter(x[k],y[k],z[k],c=dpgmm.ranked_probability[k],s=500*(dpgmm.ranked_dl[k]/options.dmax)**2,marker='.',alpha=0.5,edgecolors='none')#,norm=matplotlib.colors.LogNorm()
                ax.scatter(x[imax],y[imax],z[imax],c=dpgmm.ranked_probability[imax],s=128,marker='+')#,norm=matplotlib.colors.LogNorm()
                fig.colorbar(S)
                
                ax.plot(np.linspace(-MAX,MAX,100),np.zeros(100),color='k', zdir='y', zs=0.0)
                ax.plot(np.linspace(-MAX,MAX,100),np.zeros(100),color='k', zdir='x', zs=0.0)
                ax.plot(np.zeros(100),np.linspace(-MAX,MAX,100),color='k', zdir='y', zs=0.0)

                ax.scatter(x[k], z[k], c = dpgmm.ranked_probability[k], zdir='y', zs=MAX,marker='.',alpha=0.5,edgecolors='none')
                ax.scatter(y[k], z[k], c = dpgmm.ranked_probability[k], zdir='x', zs=-MAX,marker='.',alpha=0.5,edgecolors='none')
                ax.scatter(x[k], y[k], c = dpgmm.ranked_probability[k], zdir='z', zs=-MAX,marker='.',alpha=0.5,edgecolors='none')


                ax.scatter(x[imax],z[imax],c=dpgmm.ranked_probability[imax], zdir='y', zs=MAX,s=128,marker='+')
                ax.scatter(y[imax],z[imax],c=dpgmm.ranked_probability[imax], zdir='x', zs=-MAX,s=128,marker='+')
                ax.scatter(x[imax],y[imax],c=dpgmm.ranked_probability[imax], zdir='z', zs=-MAX,s=128,marker='+')
                
                ax.set_xlim([-MAX, MAX])
                ax.set_ylim([-MAX, MAX])
                ax.set_zlim([-MAX, MAX])

                ax.set_xlabel(r"$D_L/\mathrm{Mpc}$")
                ax.set_ylabel(r"$D_L/\mathrm{Mpc}$")
                ax.set_zlabel(r"$D_L/\mathrm{Mpc}$")
                
                for ii in xrange(0,360,1):
                    sys.stderr.write("producing frame %03d\r"%ii)
                    ax.view_init(elev=40., azim=ii)
                    plt.savefig(os.path.join(out_dir, 'galaxies_3d_scatter_%03d.png'%ii))
                sys.stderr.write("\n")
                
                # make an animation
                os.system("ffmpeg -f image2 -r 10 -i %s -vcodec mpeg4 -y %s"%(os.path.join(out_dir, 'galaxies_3d_scatter_%03d.png'),os.path.join(out_dir, 'movie.mp4')))
                
                plt.figure()
                lon_gals = np.degrees(dpgmm.ranked_ra[k][::-1]) - np.mean(gmst_deg)
                lat_gals = np.degrees(dpgmm.ranked_dec[k][::-1])
                dl_gals = dpgmm.ranked_dl[k][::-1]
                logProbability = dpgmm.ranked_probability[k][::-1]
                m = Basemap(projection='moll', lon_0=round(lon_cen, 2), lat_0=0, resolution='c')
                m.drawcoastlines(linewidth=0.5, color='0.5')
                m.drawparallels(np.arange(-90,90,30), labels=[1,0,0,0], labelstyle='+/-', linewidth=0.1, dashes=[1,1], alpha=0.5)
                m.drawmeridians(np.arange(0,360,60), linewidth=0.1, dashes=[1,1], alpha=0.5)
                m.drawmapboundary(linewidth=0.5, fill_color='white')
                plt.scatter(*m(lon_map, lat_map), c=dpgmm.log_skymap, cmap='OrRd', s=2, lw=0)
                S = plt.scatter(*m(lon_gals, lat_gals), s=dl_gals, c=logProbability, lw=0, marker='o')

                if injFile is not None: plt.scatter(*m(lon_inj, lat_inj), color='k', s=500, marker='+')
                cbar = m.colorbar(S,location='bottom',pad="5%")
                cbar.set_label(r"$\log(\mathrm{Probability})$")

                plt.savefig(os.path.join(out_dir, 'galaxies_marg_sky_%d.pdf'%(eventID)))