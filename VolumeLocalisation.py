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
from utils import *
import matplotlib
import time
from VolumeLocalisationCore import *

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
    parser.add_option("--max-stick", type="int", dest="max_stick", help="maximum number of gaussian components", default=16)
    parser.add_option("-e", type="int", dest="event_id", help="event ID")
    parser.add_option("--threads", type="int", dest="nthreads", help="number of threads to spawn", default=None)
    parser.add_option("--catalog", type="string", dest="catalog", help="galaxy catalog to use", default=None)
    parser.add_option("--plots", type="string", dest="plots", help="produce plots", default=False)
    parser.add_option("-N", type="int", dest="ranks", help="number of ranked galaxies to list in output", default=1000)
    parser.add_option("--nsamps", type="int", dest="nsamps", help="number of posterior samples to utilise", default=None)
    parser.add_option("--cosmology", type="int", dest="cosmology", help="assume a lambda CDM cosmology", default=1)
    parser.add_option("--3d", type="int", dest="threed", help="3d volume map", default=0)
    (options, args) = parser.parse_args()
    np.random.seed(1)
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
    print 'Total grid size is :',options.bins[0]*options.bins[1]*options.bins[2]
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
                              catalog=options.catalog,
                              standard_cosmology=options.cosmology)

    dpgmm.compute_dpgmm()
    pickle.dump(dpgmm.density, open(os.path.join(options.output,'dpgmm_model.p'), 'wb'))
    if dpgmm.catalog is not None:
        dpgmm.rank_galaxies()

        np.savetxt(os.path.join(options.output,'galaxy_ranks.txt'),
                   np.array([np.degrees(dpgmm.ranked_ra[:options.ranks]),np.degrees(dpgmm.ranked_dec[:options.ranks]),dpgmm.ranked_dl[:options.ranks],dpgmm.ranked_zs[:options.ranks],dpgmm.ranked_zp[:options.ranks],dpgmm.ranked_probability[:options.ranks]]).T,
                   fmt='%.9f\t%.9f\t%.9f\t%.9f\t%.9f\t%.9f\t',
                   header='ra[deg]\tdec[deg]\tDL[Mpc]\tz_spec\tz_phot\tlogposterior')

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
        lon_cen = np.degrees(np.mean(samples[idx,2])) - np.mean(gmst_deg)
        lat_cen = np.degrees(np.mean(samples[idx,1]))

    lon_samp = np.degrees(samples[idx,2]) - gmst_deg
    lat_samp = np.degrees(samples[idx,1])

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
#        plt.scatter(*m(lon_samp, lat_samp), color='k', s=0.1, lw=0)
        S = m.contourf(X,Y,dpgmm.skymap,100,linestyles='-', hold='on', origin='lower', cmap='YlOrRd', s=2, lw=0, vmin = 0.0)
        if injFile is not None: plt.scatter(*m(lon_inj, lat_inj), color='r', s=500, marker='+')
        plt.savefig(os.path.join(out_dir, 'marg_sky_%d.pdf'%(eventID)))

        plt.figure()
        m = Basemap(projection='hammer', lon_0=round(lon_cen, 2), lat_0=0, resolution='c')
        m.drawcoastlines(linewidth=0.5, color='0.5')
        m.drawparallels(np.arange(-90,90,30), labels=[1,0,0,0], labelstyle='+/-', linewidth=0.1, dashes=[1,1], alpha=0.5)
        m.drawmeridians(np.arange(0,360,60), linewidth=0.1, dashes=[1,1], alpha=0.5)
        m.drawmapboundary(linewidth=0.5, fill_color='white')
        X,Y = m(*np.meshgrid(lon_map, lat_map))
#        plt.scatter(*m(lon_samp, lat_samp), color='k', s=0.1, lw=0)
        S = m.contourf(X,Y,dpgmm.skymap,100,linestyles='-', hold='on',origin='lower', cmap='YlOrRd', s=2, lw=0, vmin = 0.0)
        if injFile is not None: plt.scatter(*m(lon_inj, lat_inj), color='r', s=500, marker='+')
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

                threshold = dpgmm.heights['0.9']
                (k,) = np.where(dpgmm.ranked_probability>threshold)
                np.savetxt(os.path.join(options.output,'galaxy_0.9.txt'),
                           np.array([np.degrees(dpgmm.ranked_ra[k]),np.degrees(dpgmm.ranked_dec[k]),dpgmm.ranked_dl[k],dpgmm.ranked_zs[k],dpgmm.ranked_zp[k],dpgmm.ranked_probability[k]]).T,
                           fmt='%.9f\t%.9f\t%.9f\t%.9f\t%.9f\t%.9f\t',
                           header='ra[deg]\tdec[deg]\tDL[Mpc]\tz_spec\tz_phot\tlogposterior')

                imax = dpgmm.ranked_probability.argmax()
                threshold = dpgmm.heights['0.5']
                (k,) = np.where(dpgmm.ranked_probability>threshold)
                MIN = dpgmm.grid[0][0]
                MAX = dpgmm.grid[0][-1]
                sys.stderr.write("%d galaxies above threshold, plotting\n"%(len(k)))
                np.savetxt(os.path.join(options.output,'galaxy_0.5.txt'),
                           np.array([np.degrees(dpgmm.ranked_ra[k]),np.degrees(dpgmm.ranked_dec[k]),dpgmm.ranked_dl[k],dpgmm.ranked_zs[k],dpgmm.ranked_zp[k],dpgmm.ranked_probability[k]]).T,
                           fmt='%.9f\t%.9f\t%.9f\t%.9f\t%.9f\t%.9f\t',
                           header='ra[deg]\tdec[deg]\tDL[Mpc]\tz_spec\tz_phot\tlogposterior')
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
                    plt.savefig(os.path.join(out_dir, 'galaxies_3d_scatter_%03d.png'%ii),dpi=200)
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

                S = plt.scatter(*m(lon_gals, lat_gals), s=10, c=dl_gals, lw=0, marker='o')

                if injFile is not None: plt.scatter(*m(lon_inj, lat_inj), color='k', s=500, marker='+')
#                cbar = m.colorbar(S,location='bottom',pad="5%")
#                cbar.set_label(r"$\log(\mathrm{Probability})$")
                plt.savefig(os.path.join(out_dir, 'galaxies_marg_sky_%d.pdf'%(eventID)))

    if options.threed:
        # produce a volume plot

        from skimage import measure
        from mpl_toolkits.mplot3d import Axes3D
        # Create a cartesian grid
        N = 100
        MAX = dpgmm.grid[0][-1]
        x = np.linspace(-MAX,MAX,N)
        y = np.linspace(-MAX,MAX,N)
        z = np.linspace(-MAX,MAX,N)
        sys.stderr.write("producing 3 dimensional maps\n")
        sample_args = ((dpgmm.density,np.array((xi,yi,zi))) for xi in x for yi in y for zi in x)
        results = dpgmm.pool.imap(logPosteriorCartesian, sample_args, chunksize = N**3/(dpgmm.nthreads * 16))
        log_cartesian_map = np.array([r for r in results]).reshape(N,N,N)
        log_cartesian_map[np.isinf(log_cartesian_map)] = np.nan
        # create a normalized cumulative distribution
        log_cartesian_sorted = np.sort(log_cartesian_map.flatten())[::-1]
        log_cartesian_cum = cumulative.fast_log_cumulative(log_cartesian_sorted)
        # find the indeces  corresponding to the given CLs
        adLevels = np.ravel([0.1,0.5,0.9])
        args = [(log_cartesian_sorted,log_cartesian_cum,level) for level in adLevels]
        adHeights = dpgmm.pool.map(FindHeights,args)
        heights = {str(lev):hei for lev,hei in zip(adLevels,adHeights)}
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for lev,cmap,al in zip(['0.1','0.5','0.9'],['BrBG','YlOrBr','BuPu'],[0.9,0.3,0.1]):
            verts, faces = measure.marching_cubes(np.exp(log_cartesian_map), np.exp(heights[lev]), spacing=(np.diff(x)[0],np.diff(y)[0],np.diff(z)[0]))
            ax.plot_trisurf(verts[:, 0], verts[:,1], faces, verts[:, 2],
                        cmap=cmap, alpha = al, antialiased=False, linewidth=0)
#        ax.set_xlim([-MAX, MAX])
#        ax.set_ylim([-MAX, MAX])
#        ax.set_zlim([-MAX, MAX])

        ax.set_xlabel(r"$D_L/\mathrm{Mpc}$")
        ax.set_ylabel(r"$D_L/\mathrm{Mpc}$")
        ax.set_zlabel(r"$D_L/\mathrm{Mpc}$")
        plt.show()



    if 0:
        sys.stderr.write("rendering 3D volume\n")
        from mayavi import mlab
        # Create a cartesian grid
        N = 10
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
    sys.stderr.write("\n")

