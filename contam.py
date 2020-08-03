"""
TODO
    - get sim info from info.txt file
"""
import gc
import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import pynbody
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


def main_eff(isnap, path, ncpu, levelmax, levelmin, omega_b=0.0, cen=None, rad=None, fn='', prof=True):

    print('Working in memory-efficient mode')
    print('Loading output_{0:05d}'.format(isnap))
    print('Using', pynbody.config['number_of_threads'], 'cores', flush=True)
    sim_path = path+'output_{0:05d}'.format(isnap)
    cpus = [[x] for x in range(1, ncpu+1)]
    
    if cen is None:
        cen = [0.5, 0.5, 0.5]
    if rad is None:
        rad = 0.05
    print('Centred on', cen, 'with radius', rad)

    # Read in comsological values
    print('Reading in parameters')
    s = pynbody.load(sim_path, cpus=[0])
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
    
    if omega_b == 0.0:
        print('---- omega_b set to zero, assuming no hydro')

    # How many mass levels do we want to go down?
    max_l_mass = levelmax - levelmin + 1
    min_mass = get_part_mass(levelmax, L, omega_m, omega_b) / h

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
    ax.set_ylim([1.0, top])
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


if __name__=='__main__':
    ncpu = 64
    isnap = 11
    path = '/lustre/scratch/astro/lc589/lg/runs/lg_oldwrite/'
    levelmax = 11
    levelmin = 8

    # LG zoom region
    # ix = [928., 784., 1056.]
    # nx = [224., 224., 224.]
    # nl = 2.**11

    # rad = max(nx) / nl / 2.
    # cen = [(iix + (inx/2.)) / nl for iix, inx in zip(ix, nx)]
    
    # # LG most massive halo
    rad = 0.00207220
    cen = [0.47796, 0.52700, 0.47867]

    main_eff(isnap, path, ncpu, levelmax, levelmin, rad=rad, cen=cen, fn='m31')
