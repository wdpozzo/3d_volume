import matplotlib
import matplotlib.pyplot as plt

def init_plotting():
    plt.rcParams['figure.figsize'] = (3.4, 3.4)
    plt.rcParams['font.size'] = 11
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.sans-serif'] = ['Bitstream Vera Sans']
    plt.rcParams['axes.labelsize'] = plt.rcParams['font.size']
    plt.rcParams['axes.titlesize'] = plt.rcParams['font.size']
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['xtick.major.size'] = 3
    plt.rcParams['xtick.minor.size'] = 3
    plt.rcParams['xtick.major.width'] = 1
    plt.rcParams['xtick.minor.width'] = 1
    plt.rcParams['ytick.major.size'] = 3
    plt.rcParams['ytick.minor.size'] = 3
    plt.rcParams['ytick.major.width'] = 1
    plt.rcParams['ytick.minor.width'] = 1
    plt.rcParams['legend.frameon'] = False
    plt.rcParams['legend.loc'] = 'center left'
    plt.rcParams['axes.linewidth'] = 1
    plt.rcParams['contour.negative_linestyle'] = 'solid'
    plt.gca().spines['right'].set_color('none')
    plt.gca().spines['top'].set_color('none')
    plt.gca().xaxis.set_ticks_position('bottom')
    plt.gca().yaxis.set_ticks_position('left')

init_plotting()

def volume_rendering():
    from skimage import measure
    from mpl_toolkits.mplot3d import Axes3D
    # Create a cartesian grid
    N = 200
    MAX = np.max(celestial_to_cartesian(np.array([dpgmm.grid[0][-1],0.0,0.0])))
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
    log_cartesian_cum = fast_log_cumulative(log_cartesian_sorted)
    # find the indeces corresponding to the given CLs
    adLevels = np.ravel([0.05,0.5,0.95])
    args = [(log_cartesian_sorted,log_cartesian_cum,level) for level in adLevels]
    adHeights = dpgmm.pool.map(FindHeights,args)
    heights = {str(lev):hei for lev,hei in zip(adLevels,adHeights)}
    
    fig = plt.figure(figsize=(13.5,13.5))
    ax = fig.add_subplot(111, projection='3d')
    # draw sphere
    u, v = np.mgrid[0:2*np.pi:100j, 0:np.pi:100j]
    xs = MAX*np.cos(u)*np.sin(v)
    ys = MAX*np.sin(u)*np.sin(v)
    zs = MAX*np.cos(v)
    ax.plot_wireframe(xs, ys, zs, color="0.9", alpha=0.5, lw=0.3)
    # initialise the view so that the observer is orthogonal to the volume
    ax.view_init(elev=np.mean(lat_map), azim=np.mean(lon_map))
    for lev,cmap,al in zip(['0.05','0.5','0.95'],['BrBG','YlOrBr','BuPu'],[0.9,0.3,0.1]):
        verts, faces, normals, values = measure.marching_cubes_lewiner(np.exp(log_cartesian_map), np.exp(heights[lev]), spacing=(np.diff(x)[0],np.diff(y)[0],np.diff(z)[0]))
        ax.plot_trisurf(verts[:, 0], verts[:,1], faces, verts[:, 2],
                    cmap=cmap, alpha = al, antialiased=False, linewidth=0)
    # plot the Earth in the origin
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage
    from matplotlib._png import read_png
    im = read_png('earth.png')
    imagebox = OffsetImage(im, zoom=.05)
    xy = [0.0, 0.0]               # coordinates to position this image

    ab = AnnotationBbox(imagebox, xy,
                        xybox=(0., 0.),
                        xycoords='data',
                        boxcoords="offset points",
                        frameon=False)
    ax.add_artist(ab)
#        ax.scatter([0.0],[0.0],[0.0],c='k',s=200,marker=r'$\bigoplus$',edgecolors='none')
    ax.grid(False)
    ax.xaxis.pane.set_edgecolor('black')
    ax.yaxis.pane.set_edgecolor('black')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    [t.set_va('center') for t in ax.get_yticklabels()]
    [t.set_ha('left') for t in ax.get_yticklabels()]
    [t.set_va('center') for t in ax.get_xticklabels()]
    [t.set_ha('right') for t in ax.get_xticklabels()]
    [t.set_va('center') for t in ax.get_zticklabels()]
    [t.set_ha('left') for t in ax.get_zticklabels()]
    ax.xaxis._axinfo['tick']['inward_factor'] = 0
    ax.xaxis._axinfo['tick']['outward_factor'] = 0.4
    ax.yaxis._axinfo['tick']['inward_factor'] = 0
    ax.yaxis._axinfo['tick']['outward_factor'] = 0.4
    ax.zaxis._axinfo['tick']['inward_factor'] = 0
    ax.zaxis._axinfo['tick']['outward_factor'] = 0.4
    ax.zaxis._axinfo['tick']['outward_factor'] = 0.4

    ax.set_xlim(-MAX,MAX)
    ax.set_ylim(-MAX,MAX)
    ax.set_zlim(-MAX,MAX)
    ax.set_xlabel(r"$D_L/\mathrm{Mpc}$")
    ax.set_ylabel(r"$D_L/\mathrm{Mpc}$")
    ax.set_zlabel(r"$D_L/\mathrm{Mpc}$")
    plt.savefig(os.path.join(out_dir, 'posterior_volume.pdf'))
