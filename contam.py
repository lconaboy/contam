"""
TODO
    - get sim info from info.txt file
"""
import gc

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


def main_eff(isnap, path, ncpu, levelmax, levelmin, L, omega_m, omega_b, h):


    print('Working in memory-efficient mode')
    print('Loading output_{0:05d}'.format(isnap))
    print('Using', pynbody.config['number_of_threads'], 'cores', flush=True)
    sim_path = path+'output_{0:05d}'.format(isnap)
    cpus = [[x] for x in range(1, ncpu+1)]
    cen = [0.5, 0.5, 0.5]
    rad = 0.05
    print('Centred on', cen, 'with radius', rad)

    # How many mass levels do we want to go down?
    max_l_mass = levelmax - levelmin + 1
    min_mass = get_part_mass(levelmax, L, omega_m, omega_b) / h

    # Set up histogram
    bins = np.linspace(0.0, 1.0, 101)
    hist = [np.zeros(100) for x in range(max_l_mass)]
    
    for cpu in cpus:
        print('Working on CPU', cpu[0])
        s = pynbody.load(sim_path, cpus=cpu)
        print('---- filtering snapshot', flush=True)
        sub = s.d[Sphere(rad, cen)]
        ls = len(sub)
        
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
                print(pos_arr[mass_arr==i])
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
    np.savetxt('bins', bin_cen)
    
    for i in range(max_l_mass):
        np.savetxt('mass_{0:03d}'.format(i), hist[i])

    # Plot data
    s = pynbody.load(sim_path, cpus=[1])  # for sim info
    cols = get_cols(max_l_mass)
    for i in range(max_l_mass):
        plt.plot(bin_cen, hist[i], c=cols[i], label=r"$\ell - \ell_\mathrm{{max}}$ = {}".format(i))

    plt.xlabel('x/r')
    plt.ylabel('n(r)')
    plt.title('z = {0:5.3f}'.format(1.0/s.properties['a'] - 1.0))
    plt.legend()
    plt.savefig('contam_{0:05d}.png'.format(isnap))

    print('Done')

def get_cols(n):
    import matplotlib
    # Colourmap
    cmap = matplotlib.cm.get_cmap('jet')
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
    ncpu = 720
    isnap = 18
    path = '/gpfs/scratch/userexternal/lconaboy/bd/drft_run/runs/zoom_972_100Mpc_512_16384_7248/unbiased/'
    levelmax = 14
    levelmin = 9
    L = 100.0
    omega_m = 0.314
    omega_b = 0.045
    h = 0.67
    
    main_eff(isnap, path, ncpu, levelmax, levelmin, L, omega_m, omega_b, h)
