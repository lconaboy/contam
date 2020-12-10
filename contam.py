"""
TODO
    - get sim info from info.txt file
"""
import gc
import os
import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('agg')

import pynbody
pynbody.config['number_of_threads'] = 1
from pynbody.array import SimArray
from pynbody.filt import Sphere, Cuboid


def main(isnap, path):
    # pynbody.config['number_of_threads'] = 8
    print('Using', pynbody.config['number_of_threads'], 'cores', flush=True)

    # Colourmap
    cmap = matplotlib.cm.get_cmap('jet')
    cmap_points = np.linspace(0.0, 1.0, 15)
    cols = cmap(cmap_points)    

    print('Loading output_{0:05d}'.format(isnap), flush=True)

    s = pynbody.load(path+'output_{0:05d}'.format(isnap))

    print('Filtering snapshot', flush=True)

    cen = (0.5, 0.5, 0.5)
    rad = 0.03

    sub = s[Sphere(rad, cen)]
    sub = sub.d
    pos = sub['pos']
    mass = sub['mass']    
    
    # Calculate distance from the centre for each particle
    # s_cen = s.array(cen, s.info["unit_length"])
    # s_rad = s.array(rad, s.info["unit_length"])
    print('Calculating distances', flush=True)
    pos -= cen
    pos = np.linalg.norm(pos, axis=1)
    pos_norm = pos/rad
    pos_arr = np.asarray(pos_norm)/rad


    # Calculate mass of each particle
    print('Calculating masses', flush=True)
    mass_norm = mass/np.min(mass)
    mass_norm_ref = np.log2(mass_norm)/3
    mass_arr = np.asarray(mass_norm_ref, dtype=int)

    ### PLOTTING ###
    # Hist
    print('Plotting', flush=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim([0, 1])
    bins = np.linspace(0, 1, 100)
    for i in range(int(np.max(mass_arr))+1):
        ax.hist(pos_arr[mass_arr==i], bins=bins, histtype="step", label=r"$\ell - \ell_\mathrm{{max}}$ = {}".format(i), color=cols[i])
    ax.set_xlabel(r"r/r$_0$")
    ax.set_ylabel("N")
    ax.set_yscale("log")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig("contam_{0:05d}.pdf".format(isnap))


def main_eff(isnap, path, ncpu, levelmax, levelmin, omega_b=0.0, cen=None, rad=None, fn='', prof=False):
    """Checks for contamination in a region, in a memory-efficient way
    by reading each CPU of data individually.  Produces a plot of
    particle level distribution as a function of normalised distance
    from the centre of the region.

    :param isnap: (int) which snapshot number to analyse
    :param path: (str) path to snapshots
    :param ncpu: (int) number of CPUs used for simulation
    :param levelmax: (int) maximum particle level (_not_ AMR level)
    :param levelmin: (int) minimum particle level
    :param omega_b: (float) baryon density parameter today
    :param cen: (tuple, floats) centre of region to analyse
    :param rad: (float) radius of region to analyse
    :param fn: (str) output plot filename
    :param prof: (bool) make density profile?

    :returns:

    :rtype: 
    """
    

    print('Working in memory-efficient mode')
    print('Loading output_{0:05d}'.format(isnap))
    print(pynbody.config)
    print('Using', pynbody.config['number_of_threads'], 'cores', flush=True)
    sim_path = path+'output_{0:05d}'.format(isnap)
    cpus = [[x] for x in range(1, ncpu+1)]
    
    if cen is None:
        cen = [0.5, 0.5, 0.5]
    if rad is None:
        rad = 0.05
    print('Centred on', cen, 'with radius', rad)

    # Read in comsological values    
    s = pynbody.load(sim_path, cpus=[0])
    
    omega_m, a, h, L = get_sim_vars(s)
    
    if omega_b == 0.0:
        print('---- omega_b set to zero, assuming no hydro')

    # Get the number of mass levels and minimum particle mass values
    max_l_mass, min_mass = get_mass_vars(levelmax, levelmin, omega_m,
                                         omega_b, h)

    # Set up histogram
    bins = np.linspace(0.0, 1.0, 101)
    hist = [np.zeros(100) for x in range(max_l_mass)]
    n_part = 0

    for cpu in cpus:
        print('Working on CPU', cpu[0])
        s = pynbody.load(sim_path, cpus=cpu)
        print('---- filtering snapshot', flush=True)
        sub = s.d[Sphere(rad, cen)]
        ls = len(sub)
        n_part += ls
        
        # Check whether we have anything
        if ls > 0:
            print('---- found', ls, 'particles', flush=True)
            pos_arr, mass_arr = check_particles(sub, cen, rad, min_mass)
            # Bin for this CPU
            for i in range(max_l_mass):
                # print(pos_arr[mass_arr==i])
                tmp_hist, tmp_bin_edges = np.histogram(pos_arr[mass_arr==i], bins=bins)
                # Add to the main histogram for this mass level
                hist[i] += tmp_hist

        else:
            print('---- no particles found in region', flush=True)

        # Clean up
        gc.collect()
        del s
        del sub

    # Save data
    bin_cen = 0.5*(bins[1:] + bins[:-1])
    np.savetxt(fn+'bins_{0:05d}.dat'.format(isnap), bin_cen)
    
    for i in range(max_l_mass):
        np.savetxt(fn+'mass_{0:03d}_{1:05d}.dat'.format(i, isnap), hist[i])

    # Plot data
    s = pynbody.load(sim_path, cpus=[1])  # for sim info
    cols = get_cols(max_l_mass)
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Plot each mass bin in a loop
    for i in range(max_l_mass):
        ax.step(bin_cen, hist[i], c=cols[i], label=r"$\ell_\mathrm{{part}}$ = {}".format(levelmax - i))

    ax.set_xlabel('r/r$_0$')
    # ax.set_ylabel('log$_{{10}}$[n(r) + 1]')
    ax.set_ylabel('N')
    ax.set_title('z = {0:.2f};\n centre = ({1:.3f}, {2:.3f}, {3:.3f}); r$_0$ = {4:.3f} Mpc/h'.format(1.0/s.properties['a'] - 1.0, cen[0]*L, cen[1]*L, cen[2]*L, rad*L))
    ax.set_yscale('log')
    _, top = ax.get_ylim()
    ax.set_ylim([0.5, top])
    ax.legend()
    fig.savefig(fn+'contam_{0:05d}.pdf'.format(isnap))

    print('Found ', n_part, ' particles in total')

    if prof:
        print('Plotting profile')
        d = bin_cen * rad * L / h
        m = np.zeros(100)
        fac = 1.0
        for i in range(max_l_mass-1, 0, -1):
            m += hist[i] * fac
            fac *= 8.0  # the next lot of particles are 8 times more massive

        m *= get_part_mass(levelmax, L, omega_m, omega_b) / h  # Msol

        # Volume of each annulus
        bins *= rad * L  # cMpc/h
        bins /= h        # cMpc
        v = (4./3.) * np.pi *(bins[1:]**3 - bins[:-1]**3)  # cMpc^3

        rho = m / v

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(d, rho, c='k')
        ax.set_xlabel('r (Mpc)')
        ax.set_ylabel('$\rho$ (M$_\odot$/Mpc$^3$)')
        ax.set_yscale('log')
        fig.savefig(fn+'prof_{0:05d}.pdf'.format(isnap))
        
    print('Done')


def halo(isnap, path, rockstar_path, levelmax, levelmin, omega_b=0.0):
    

def read_rockstar(path, iout):
    """Reads in a rockstar out.list file.

    :param path: (str) path to directory containing out.list files
    :param iout: (int) index of out.list file to look at

    :returns: halo index, mvir (Msol/h), rvir (code units), x (code
              units), y (code units), z (code units)

    :rtype: (int), (float), (float), (float), (float), (float)
    """
    
    path = os.path.join(path, 'out_{0:d}.list'.format(iout))

    # Read box size from rockstar header
    l = ''
    with open(path, 'r') as f:
        while 'Box' not in l:
            l = f.readline()
            
    l = l.split().strip('\n')
    b = float(l[-2])  # box size in Mpc/h

    print('---- read rockstar box size', b, 'Mpc/h')
    
    idx, mvir, rvir, x, y, z = np.loadtxt(path, usecols=(0, 2, 5, 8, 9, 10),
                                          unpack=True)

    # Convert positions to code units, rockstar positions are in
    # Mpc/h, while sizes are in kpc/h
    rvir = rvir / b / 1e3
    x = x / b
    y = y / b
    z = z / b

    return idx, mvir, rvir, x, y, z


def get_cols(n):
    import matplotlib
    # Colourmap
    cmap = matplotlib.cm.get_cmap('plasma')
    cmap_points = np.linspace(0.0, 1.0, n)
    cols = cmap(cmap_points)

    return cols


def get_part_mass(l, L, omega_m, omega_b):
    """Calculates the particle mass at a given level l, using the box
    length in cMpc/h, L, and the matter and baryon density
    parameters. Returns mass in Msol/h.
    """

    # Set up constants
    rhom = 2.77519737e11  # (Msol/h) / (Mpc/h)^3
    dx = L / (2**l)  # Mpc/h
    mp = (omega_m - omega_b) * rhom * dx * dx * dx  # Msol/h
    
    return mp

def get_mass_vars(levelmax, levelmin, omega_m, omega_b, h):
    """Get the mass variables.

    :param levelmax: (float) maximum particle level
    :param levelmin: (float) minimum particle level
    :param omega_m: (float) matter density parameter today
    :param omega_b: (float) baryon density parameter today
    :param h: (float) little h (H0/100)
    :returns: number of mass levels, minimum particle mass
    :rtype: (int), (float)

    """
    # How many mass levels do we want to go down?
    max_l_mass = levelmax - levelmin + 1
    min_mass = get_part_mass(levelmax, L, omega_m, omega_b) / h

    return max_l_mass, min_mass


def get_sim_vars(s):
    """Read in the simulation parameters from the snapshot object.

    :param s: snapshot object

    :returns: omega_m, scale factor, H0/100, boxsize (cMpc/h)

    :rtype: (float), (float), (float), (float)
    """
    
    print('-- reading in parameters')
    omega_m = s.properties['omegaM0']
    # omega_b = s.properties['omegaB0']
    a = s.properties['a']
    h = s.properties['h']
    L = s.properties['boxsize'].in_units('Mpc') * h / a
    print('---- h :', h)
    print('---- a :', a, 'z :', 1./a - 1.)
    print('---- omega_m           :', omega_m)
    print('---- omega_b (not read):', omega_b)
    print('---- box size (cMpc/h) :', L)


    return omega_m, a, h, L


def check_particles(sub, cen, rad, min_mass):
    """Checks the particles within the subsnap.  Normalises masses to
    the smallest particle mass in the simuation, and distances to cen.

    :param sub: the subsnap to analyse
    :param cen: centre of region to analyse in code units
    :param rad: radius to analyse over in code units
    :param min_mass: minimum particle mass in Msol/h
    """
    
    pos = sub['pos']
    mass = sub['mass']
    mass.convert_units('Msol')    
    
    # Calculate distance from the centre
    print('---- calculating distances', flush=True)
    pos -= SimArray(cen, pos.units)
    pos = np.linalg.norm(pos, axis=1)
    pos_arr = pos/rad
    print('-------- min: {0:.3f}  max: {1:.3f}'.format(min(pos_arr), max(pos_arr)))
    # pos_arr = np.asarray(pos_norm)

    # Calculate mass of each particle
    print('---- calculating masses', flush=True)
    # Normalise to smallest possible mass at levelmax
    mass_norm = mass/min_mass
    mass_norm_ref = np.log2(mass_norm)/3
    mass_norm_ref_rnd = np.around(mass_norm_ref)
    diff = mass_norm_ref_rnd - mass_norm_ref
    av_diff = np.mean(np.abs(diff))
    print('-------- min diff      :', np.min(diff))
    print('-------- max diff      :', np.max(diff))
    print('-------- avg |diff|    :', np.mean(np.abs(diff)))
    print('-------- min mass_norm :', np.min(mass_norm_ref_rnd))
    print('-------- max mass_norm :', np.max(mass_norm_ref_rnd))
            
    assert av_diff < 1.0e-2, 'Average difference > 1.0e-2, stopping'
            
    mass_arr = np.asarray(mass_norm_ref_rnd, dtype=int)
    
    del mass_norm_ref
    del mass_norm_ref_rnd
    del diff

    return pos_arr, mass_arr


if __name__=='__main__':
    ncpu = 20*28 #140 #64
    isnap = 11
    path = '/snap7/scratch/dp004/dc-cona1/lg/runs/100Mpch_256_4096/'
    # the following are for particles, not grids
    levelmax = 12
    levelmin = 8

    # LG zoom region
    # ix = [928., 784., 1056.]
    # nx = [224., 224., 224.]
    # nl = 2.**11

    # rad = max(nx) / nl / 2.
    # cen = [(iix + (inx/2.)) / nl for iix, inx in zipix, nx)]
    
    # LG most massive halo
    rad = 0.00198
    cen = [0.47823645, 0.52603906, 0.47954712]
    fn = 'halo1_'

    main_eff(isnap, path, ncpu, levelmax, levelmin, rad=rad, cen=cen, fn=fn)

    # Next largest
    rad = 0.00144
    cen = [0.51149074, 0.49567393, 0.457671]
    fn = 'halo2_'

    main_eff(isnap, path, ncpu, levelmax, levelmin, rad=rad, cen=cen, fn=fn)

    # Full zoom
    # rad = 0.0208 # smallest ni
    # rad = 104/2048
    # cen = np.array([1080, 888, 1135])/2048.
    # fn = 'full'

