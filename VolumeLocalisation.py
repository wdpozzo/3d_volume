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
matplotlib.use("MACOSX")

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
        
        bins: number of bins in the d,ra,dec directions. default = [10,10,10]
        
        dist_max: maximum radial distance to consider. default = 218 Mpc
        
        nthreads: number of multiprocessing pool workers to use. default = multiprocessing.cpu_count()
        
        injection: the injection file.
        
        catalog: the galaxy catalog for the ranked list of galaxies
        """
    def __init__(self,posterior_samples,dimension=3,max_sticks=16,bins=[10,10,10],dist_max=218,nthreads=None,injection=None,catalog=None):
        np.random.seed(0)
        self.posterior_samples = np.array(posterior_samples)
        self.dims = 3
        self.max_sticks = max_sticks
        if nthreads == None:
            self.nthreads = mp.cpu_count()
        else:
            self.nthreads = nthreads
        self.bins = bins
        self.dist_max = dist_max
        self.pool = mp.Pool(self.nthreads)
        self.injection = injection
        self.distance_max=dist_max
        self.catalog = None
        if catalog is not None:
            self.catalog = readGC(catalog)
        self._initialise_grid()

    def _initialise_dpgmm(self):
        self.model = DPGMM(self.dims)
        for point in self.posterior_samples:
            self.model.add(point)
        self.model.setPrior(scale=np.prod(celestial_to_cartesian(np.array([self.dD,self.dDEC,self.dRA]))))
        self.model.setThreshold(1e-4)
        self.model.setConcGamma(1,1)
    
    def _initialise_grid(self):
        a = np.maximum(0.9*samples[:,0].min(),1.0)
        b = np.minimum(1.1*samples[:,0].max(),self.distance_max)
        self.grid = [np.linspace(a,b,self.bins[0]),np.linspace(-np.pi/2.0,np.pi/2.0,self.bins[1]),np.linspace(0.0,2.0*np.pi,self.bins[2])]
        self.dD = np.diff(self.grid[0])[0]
        self.dDEC = np.diff(self.grid[1])[0]
        self.dRA = np.diff(self.grid[2])[0]

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
        jobs = ((self.density,np.array((d,dec,ra))) for d,dec,ra in zip(self.catalog[:,2],self.catalog[:,1],self.catalog[:,0]))
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

    def logPosterior(self,celestial_coordinates):
        cartesian_vect = celestial_to_cartesian(celestial_coordinates)
        logPs = [np.log(self.density[0][ind])+prob.logProb(cartesian_vect) for ind,prob in enumerate(self.density[1])]
        return logsumexp(logPs)+np.log(Jacobian(cartesian_vect))

    def Posterior(self,celestial_coordinates):
        cartesian_vect = celestial_to_cartesian(celestial_coordinates)
        Ps = [self.density[0][ind]*prob.prob(cartesian_vect) for ind,prob in enumerate(self.density[1])]
        return reduce(np.sum,Ps)*Jacobian(cartesian_vect)
    
    def evaluate_volume_map(self):
        N = self.bins[0]*self.bins[1]*self.bins[2]
        sys.stderr.write("computing log posterior for %d grid poinds\n"%N)
        sample_args = ((self.density,np.array((d,dec,ra))) for d in self.grid[0] for dec in self.grid[1] for ra in self.grid[2])
        results = self.pool.imap(logPosterior, sample_args, chunksize = N/(self.nthreads * 32))
        self.log_volume_map = np.array([r for r in results]).reshape(self.bins[0],self.bins[1],self.bins[2])
        self.volume_map = np.exp(self.log_volume_map)

    def evaluate_sky_map(self):
        dsquared = self.grid[0]**2
        self.skymap = np.trapz(dsquared[:,None,None]*self.volume_map, x=self.grid[0], axis=0)
        self.log_skymap = np.log(self.skymap)
    
    def evaluate_distance_map(self):
        cosdec = np.cos(self.grid[1])
        intermediate = np.trapz(cosdec[None,:,None]*self.volume_map, x=self.grid[1], axis=1)
        self.distance_map = np.trapz(intermediate, x=self.grid[2], axis=1)
        self.log_distance_map = np.log(self.distance_map)
        self.distance_map/=(self.distance_map*np.diff(self.grid[0])[0]).sum()

    def ConfidenceVolume(self, adLevels):
        # create a normalized cumulative distribution
        self.log_volume_map_sorted = np.sort(self.log_volume_map.flatten())[::-1]
        self.log_volume_map_cum = cumulative.fast_log_cumulative(self.log_volume_map_sorted)
        
        # find the indeces  corresponding to the given CLs
        adLevels = np.ravel([adLevels])
        args = [(self.log_volume_map_sorted,self.log_volume_map_cum,level) for level in adLevels]
        adHeights = self.pool.map(FindHeights,args)
        self.heights = {str(lev):hei for lev,hei in zip(adLevels,adHeights)}
        volumes = []
        for height in adHeights:
            (index_d, index_dec, index_ra,) = np.where(self.log_volume_map>=height)
            volumes.append(np.sum([self.grid[0][i_d]**2. *np.cos(self.grid[1][i_dec]) * self.dD * self.dRA * self.dDEC for i_d,i_dec in zip(index_d,index_dec)]))
        self.volume_confidence = np.array(volumes)

        if self.injection!=None:
            ra,dec = self.injection.get_ra_dec()
            distance = self.injection.distance
            logPval = self.logPosterior(np.array((distance,dec,ra)))
            confidence_level = np.exp(self.log_volume_map_cum[np.abs(self.log_volume_map_sorted-logPval).argmin()])
            height = FindHeights((self.log_volume_map_sorted,self.log_volume_map_cum,confidence_level))
            (index_d, index_dec, index_ra,) = np.where(self.log_volume_map>=height)
            searched_volume = np.sum([self.grid[0][i_d]**2. *np.cos(self.grid[1][i_dec]) * self.dD * self.dRA * self.dDEC for i_d,i_dec in zip(index_d,index_dec)])
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

        areas = []
        for height in adHeights:
            (index_dec,index_ra,) = np.where(self.log_skymap>=height)
            areas.append(np.sum([self.dRA*np.cos(self.grid[1][i_dec])*self.dDEC for i_dec in index_dec])*(180.0/np.pi)**2.0)
        self.area_confidence = np.array(areas)
        
        if self.injection!=None:
            ra,dec = self.injection.get_ra_dec()
            id_ra = np.abs(self.grid[2]-ra).argmin()
            id_dec = np.abs(self.grid[1]-dec).argmin()
            logPval = self.log_skymap[id_dec,id_ra]
            confidence_level = np.exp(self.log_skymap_cum[np.abs(self.log_skymap_sorted-logPval).argmin()])
            height = FindHeights((self.log_skymap_sorted,self.log_skymap_cum,confidence_level))
            (index_dec,index_ra,) = np.where(self.log_skymap >= height)
            searched_area = np.sum([self.dRA*np.cos(self.grid[1][i_dec])*self.dDEC for i_dec in index_dec])*(180.0/np.pi)**2.0
            return self.area_confidence,(confidence_level,searched_area)

        del self.log_skymap_sorted
        del self.log_skymap_cum
        return self.area_confidence,None

    def ConfidenceDistance(self, adLevels):
        cumulative_distribution = np.cumsum(self.distance_map*self.dD)
        distances = []
        for cl in adLevels:
            idx = np.abs(cumulative_distribution-cl).argmin()
            distances.append(self.grid[0][idx])
        self.distance_confidence = np.array(distances)

        if self.injection!=None:
            idx = np.abs(self.injection.distance-self.grid[0]).argmin()
            confidence_level = cumulative_distribution[idx]
            searched_distance = self.grid[0][idx]
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

def logPosterior(args):
    density,celestial_coordinates = args
    cartesian_vect = celestial_to_cartesian(celestial_coordinates)
    logPs = [np.log(density[0][ind])+prob.logProb(cartesian_vect) for ind,prob in enumerate(density[1])]
    return logsumexp(logPs)+np.log(Jacobian(cartesian_vect))

def logPosteriorCartesian(args):
    density,cartesian_coordinates = args
    logPs = [np.log(density[0][ind])+prob.logProb(cartesian_coordinates) for ind,prob in enumerate(density[1])]
    return logsumexp(logPs)

def Posterior(args):
    density,celestial_coordinates = args
    cartesian_vect = celestial_to_cartesian(celestial_coordinates)
    Ps = [density[0][ind]*prob.prob(cartesian_vect) for ind,prob in enumerate(density[1])]
    return reduce(np.sum,Ps)*np.abs(np.cos(celestial_coordinates[2]))*celestial_coordinates[0]**2

def solve_dpgmm(args):
    (nc, model) = args
    for _ in xrange(nc-1): model.incStickCap()
    try:
        it = model.solve(iterCap=1024)
        return (model.stickCap, model.nllData(), model)
    except:
        return (model.stickCap, -np.inf, model)

def Jacobian(cartesian_vect):
    d = np.sqrt(cartesian_vect.dot(cartesian_vect))
    d_sin_theta = np.sqrt(cartesian_vect[:-1].dot(cartesian_vect[:-1]))
    return d*d_sin_theta

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

def celestial_to_cartesian(celestial_vect):
    """Convert the spherical coordinate vector [r, dec, ra] to the Cartesian vector [x, y, z]."""
    celestial_vect[1]=np.pi/2. - celestial_vect[1]
    return spherical_to_cartesian(celestial_vect)

def cartesian_to_celestial(cartesian_vect):
    """Convert the Cartesian vector [x, y, z] to the celestial coordinate vector [r, dec, ra]."""
    spherical_vect = cartesian_to_spherical(cartesian_vect)
    spherical_vect[1]=np.pi/2. - spherical_vect[1]
    return spherical_vect

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

def parse_to_list(option, opt, value, parser):
    """
    parse a comma separated string into a list
    """
    setattr(parser.values, option.dest, value.split(','))

#-------------------
# start the program
#-------------------

if __name__=='__main__':
    parser = op.OptionParser()
    parser.add_option("-i", "--input", type="string", dest="input", help="Input file")
    parser.add_option("--inj",type="string",dest="injfile",help="injection file",default=None)
    parser.add_option("-o", "--output", type="string", dest="output", help="Output file")
    parser.add_option("--bins", type="string", dest="bins", help="number of bins in d,dec,ra", action='callback',
                      callback=parse_to_list)
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
    options.bins = np.array(options.bins,dtype=np.int)
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

    # we are going to normalisa the distance between 0 and 1

    if "dist" in samples.dtype.names:
        samples = np.column_stack((samples["dist"],samples["dec"],samples["ra"],samples["time"]))
    else:
        samples = np.column_stack((samples["distance"],samples["dec"],samples["ra"],samples["time"]))

    dRA = 2.0*np.pi/options.bins[2]
    dDEC = np.pi/options.bins[1]
    dD = (options.dmax-1.0)/options.bins[0]
    print 'The number of grid points in the sky is :',options.bins[1]*options.bins[2],'resolution = ',np.degrees(dRA)*np.degrees(dDEC), ' deg^2'
    print 'The number of grid points in distance is :',options.bins[0],'minimum resolution = ',dD,' Mpc'
    print 'Total grid size is :',options.bins[1]*options.bins[2]*options.bins[0]
    print 'Volume resolution is :',dD*dDEC*dRA,' Mpc^3'

    samps = []
    gmst_rad = []

    if options.nsamps is not None:
        idx = np.random.choice(range(0,len(samples[:,0])),size=options.nsamps)
    else: 
        idx = range(0,len(samples[:,0]))

    for k in xrange(len(samples[idx,0])):
        GPSTime=lal.LIGOTimeGPS(samples[k,3])
        gmst_rad.append(lal.GreenwichMeanSiderealTime(GPSTime))
        samps.append(celestial_to_cartesian(np.array((samples[k,0],samples[k,1],samples[k,2]))))

    dpgmm = DPGMMSkyPosterior(samps,dimension=3,
                              max_sticks=options.max_stick,
                              bins=options.bins,
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
    volumes,searched_volume = dpgmm.ConfidenceVolume(CLs)
    dpgmm.evaluate_sky_map()
    dpgmm.evaluate_distance_map()
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
#        hp.visufunc.mollview(dpgmm.log_skymap)
#        if injFile!=None:
#            hp.visufunc.projscatter(eq2ang(ra_inj,dec_inj), c='y', s=256, marker='*')
#        plt.savefig(os.path.join(options.output,'sky_map.pdf'),bbox_inches='tight')
#        plt.figure()
        plt.plot(dpgmm.grid[0],dpgmm.distance_map,color="k",linewidth=2.0)
        plt.hist(samples[:,0],bins=dpgmm.grid[0],normed=True,facecolor="0.9")
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
        lon_cen = np.degrees(np.mean(samples[:,2])) - np.mean(gmst_deg)
        lat_cen = np.degrees(np.mean(samples[:,1]))

    lon_samp = np.degrees(samples[:,2]) - gmst_deg
    lat_samp = np.degrees(samples[:,1])

    ra_map,dec_map = dpgmm.grid[2],dpgmm.grid[1]
    lon_map = np.degrees(ra_map) - np.mean(gmst_deg)
    lat_map = np.degrees(dec_map)
    if options.plots:
        sys.stderr.write("producing sky maps \n")
        plt.figure()
        plt.plot(np.arange(1,dpgmm.max_sticks+1),dpgmm.scores,'.')
        plt.xlabel(r"$\mathrm{number}$ $\mathrm{of}$ $\mathrm{components}$")
        plt.ylabel(r"$\mathrm{marginal}$ $\mathrm{likelihood}$")
        plt.savefig(os.path.join(out_dir, 'scores.pdf'))
        
        from mpl_toolkits.basemap import Basemap,shiftgrid
        # make an orthographic projection map
        plt.figure()
        m = Basemap(projection='ortho', lon_0=round(lon_cen, 2), lat_0=lat_cen, resolution='c')
        m.drawcoastlines(linewidth=0.5, color='0.5')
        m.drawparallels(np.arange(-90,90,30), labels=[1,0,0,0], labelstyle='+/-', linewidth=0.1, dashes=[1,1], alpha=0.5)
        m.drawmeridians(np.arange(0,360,60), linewidth=0.1, dashes=[1,1], alpha=0.5)
        m.drawmapboundary(linewidth=0.5, fill_color='white')
        X,Y = m(*np.meshgrid(lon_map, lat_map))
        plt.scatter(*m(lon_samp, lat_samp), color='k', s=0.1, lw=0)
        S = m.contourf(X,Y,dpgmm.log_skymap,10,linestyles='-', hold='on',origin='lower', cmap='YlOrRd', s=2, lw=0, vmin = -10.0)
        if injFile is not None: plt.scatter(*m(lon_inj, lat_inj), color='r', s=500, marker='+')
        cbar = m.colorbar(S,location='bottom',pad="5%")
        cbar.set_label(r"$\log(\mathrm{Probability})$")
        clevs1 = np.linspace(dpgmm.log_skymap.min(),dpgmm.log_skymap.max(),10)
        cbar.set_ticks(clevs1[::1])
        cbar.ax.set_xticklabels(clevs1[::1],rotation=90)
        plt.savefig(os.path.join(out_dir, 'marg_log_sky_%d.pdf'%(eventID)))
        # make an equatorial equidistant projection map
        plt.figure()
        m = Basemap(projection='hammer', lon_0=round(lon_cen, 2), lat_0=0, resolution='c')
        m.drawcoastlines(linewidth=0.5, color='0.5')
        m.drawparallels(np.arange(-90,90,30), labels=[1,0,0,0], labelstyle='+/-', linewidth=0.1, dashes=[1,1], alpha=0.5)
        m.drawmeridians(np.arange(0,360,60), linewidth=0.1, dashes=[1,1], alpha=0.5)
        m.drawmapboundary(linewidth=0.5, fill_color='white')
        X,Y = m(*np.meshgrid(lon_map, lat_map))
        plt.scatter(*m(lon_samp, lat_samp), color='k', s=0.1, lw=0)
        S = m.contourf(X,Y,dpgmm.log_skymap,10,linestyles='-', hold='on',origin='lower', cmap='YlOrRd', s=2, lw=0, vmin = -10.0)
        if injFile is not None: plt.scatter(*m(lon_inj, lat_inj), color='r', s=500, marker='+')
        cbar = m.colorbar(S,location='bottom',pad="5%")
        cbar.set_label(r"$\log(\mathrm{Probability})$")
        clevs1 = np.linspace(dpgmm.log_skymap.min(),dpgmm.log_skymap.max(),10)
        cbar.set_ticks(clevs1[::1])
        cbar.ax.set_xticklabels(clevs1[::1],rotation=90)
        plt.savefig(os.path.join(out_dir, 'marg_log_sky_hammer_%d.pdf'%(eventID)))
#
        plt.figure()
        m = Basemap(projection='ortho', lon_0=round(lon_cen, 2), lat_0=lat_cen, resolution='c')
        m.drawcoastlines(linewidth=0.5, color='0.5')
        m.drawparallels(np.arange(-90,90,30), labels=[1,0,0,0], labelstyle='+/-', linewidth=0.1, dashes=[1,1], alpha=0.5)
        m.drawmeridians(np.arange(0,360,60), linewidth=0.1, dashes=[1,1], alpha=0.5)
        m.drawmapboundary(linewidth=0.5, fill_color='white')
        X,Y = m(*np.meshgrid(lon_map, lat_map))
        plt.scatter(*m(lon_samp, lat_samp), color='k', s=0.1, lw=0)
        S = m.contourf(X,Y,dpgmm.skymap,10,linestyles='-', hold='on', origin='lower', cmap='YlOrRd', s=2, lw=0, vmin = 0.0)
        if injFile is not None: plt.scatter(*m(lon_inj, lat_inj), color='r', s=500, marker='+')
        cbar = m.colorbar(S,location='bottom',pad="5%")
        cbar.set_label(r"$\mathrm{probability}$ $\mathrm{density}$")
        clevs1 = np.linspace(dpgmm.skymap.min(),dpgmm.skymap.max(),10)
        cbar.set_ticks(clevs1[::1])
        cbar.ax.set_xticklabels(clevs1[::1],rotation=90)
        plt.savefig(os.path.join(out_dir, 'marg_sky_%d.pdf'%(eventID)))

        plt.figure()
        m = Basemap(projection='hammer', lon_0=round(lon_cen, 2), lat_0=0, resolution='c')
        m.drawcoastlines(linewidth=0.5, color='0.5')
        m.drawparallels(np.arange(-90,90,30), labels=[1,0,0,0], labelstyle='+/-', linewidth=0.1, dashes=[1,1], alpha=0.5)
        m.drawmeridians(np.arange(0,360,60), linewidth=0.1, dashes=[1,1], alpha=0.5)
        m.drawmapboundary(linewidth=0.5, fill_color='white')
        X,Y = m(*np.meshgrid(lon_map, lat_map))
        plt.scatter(*m(lon_samp, lat_samp), color='k', s=0.1, lw=0)
        S = m.contourf(X,Y,dpgmm.skymap,10,linestyles='-', hold='on',origin='lower', cmap='YlOrRd', s=2, lw=0, vmin = 0.0)
        if injFile is not None: plt.scatter(*m(lon_inj, lat_inj), color='r', s=500, marker='+')
        cbar = m.colorbar(S,location='bottom',pad="5%")
        cbar.set_label(r"$\mathrm{probability}$ $\mathrm{density}$")
        clevs1 = np.linspace(dpgmm.skymap.min(),dpgmm.skymap.max(),10)
        cbar.set_ticks(clevs1[::1])
        cbar.ax.set_xticklabels(clevs1[::1],rotation=90)
        plt.savefig(os.path.join(out_dir, 'marg_sky_hammer_%d.pdf'%(eventID)))

        if options.plots:
            if options.catalog:
                out_dir = os.path.join(out_dir, 'galaxies_scatter')
                os.system("mkdir -p %s"%out_dir)
                sys.stderr.write("producing 3 dimensional maps\n")
                # Create a sphere
                x = dpgmm.ranked_dl*np.cos(dpgmm.ranked_dec)*np.cos(dpgmm.ranked_ra)
                y = dpgmm.ranked_dl*np.cos(dpgmm.ranked_dec)*np.sin(dpgmm.ranked_ra)
                z = dpgmm.ranked_dl*np.sin(dpgmm.ranked_dec)

                imax = dpgmm.ranked_probability.argmax()
                threshold = dpgmm.heights['0.5']
                (k,) = np.where(dpgmm.ranked_probability>threshold)
                MIN = dpgmm.grid[0][0]
                MAX = dpgmm.grid[0][-1]
                sys.stderr.write("%d galaxies above threshold, plotting\n"%(len(k)))

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

                S = plt.scatter(*m(lon_gals, lat_gals), s=dl_gals, c=logProbability, lw=0, marker='o')

                if injFile is not None: plt.scatter(*m(lon_inj, lat_inj), color='k', s=500, marker='+')
                cbar = m.colorbar(S,location='bottom',pad="5%")
                cbar.set_label(r"$\log(\mathrm{Probability})$")
                plt.savefig(os.path.join(out_dir, 'galaxies_marg_sky_%d.pdf'%(eventID)))
    # try to produce a volume plot
    if 0:
        sys.stderr.write("rendering 3D volume\n")
        from mayavi import mlab
        # Create a cartesian grid
        N = 100
        MAX = dpgmm.grid[0][-1]
        x = np.linspace(-MAX,MAX,N)
        y = np.linspace(-MAX,MAX,N)
        z = np.linspace(-MAX,MAX,N)
        sample_args = ((dpgmm.density,np.array((xi,yi,zi))) for xi in x for yi in y for zi in x)
        results = dpgmm.pool.imap(logPosteriorCartesian, sample_args, chunksize = N**3/(dpgmm.nthreads * 16))
        log_cartesian_map = np.array([r for r in results]).reshape(N,N,N)
        log_cartesian_map[np.isinf(log_cartesian_map)] = np.nan
        min = log_cartesian_map.min()
        max = log_cartesian_map.max()
        mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(1000, 1000))
        mlab.clf()
        X,Y,Z = np.meshgrid(x,y,z)
#            O = mlab.pipeline.scalar_scatter([0.0],[0.0],[0.0], colormap="copper", scale_factor=.25, mode="sphere",opacity=0.5)
#            mlab.contour3d(X,Y,Z,log_cartesian_map,contours=10)
        mlab.pipeline.volume(mlab.pipeline.scalar_field(log_cartesian_map),vmin=min + 0.5 * (max - min),
                             vmax=min + 0.9 * (max - min))
#        axes = mlab.axes(xlabel=r'$D_L$', ylabel=r'$D_L$', zlabel=r'$D_L$')
        mlab.show()
    
    exit()
