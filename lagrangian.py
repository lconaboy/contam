import numpy as np

def write_namelist(ii, ex, lma):
    """Writes a partial namelist for use with cubic-mask"""

    with open('zoom4096.param', 'w') as f:
        f.write('&mask_params\n')
        f.write('levelmin = 8 \n')
        f.write('levelmax = {0:d}\n'.format(lma))
        f.write('ix = {0:d}\n'.format(ii[0]))
        f.write('iy = {0:d}\n'.format(ii[1]))
        f.write('iz = {0:d}\n'.format(ii[2]))
        f.write('nx = {0:d}\n'.format(ex[0]))
        f.write('ny = {0:d}\n'.format(ex[1]))
        f.write('nz = {0:d}\n'.format(ex[2]))
        f.write('npad = 4\n')
        f.write("path = '/store/erebos/spilipenko/ramses_test/fields/' \n")
        f.write('/\n')


def run_refmask(cen=None, rad=None, region=True):
    import subprocess
    
    inf = '/snap7/scratch/dp004/dc-cona1/lg/runs/oldest/output_00011'
    ini = '/snap7/scratch/dp004/dc-cona1/lg/runs/oldest/output_00001'
    
    # Run based on the region, or list of particle IDs
    if region:
        exe = '/cosma/home/dp004/dc-cona1/music/tools/ramses/get_music_refmask'
        rad = str(rad)
        xc = str(cen[0])
        yc = str(cen[1])
        zc = str(cen[2])
        subprocess.run([exe, '-inf', inf, '-ini', ini, '-xc', xc, '-yc', yc,
                        '-zc', zc, '-rad', rad])
    else:
        exe = '/cosma/home/dp004/dc-cona1/scripts/lg/get_partid_refmask'
        subprocess.run([exe, '-inf', inf, '-ini', ini])


def update_lims(old_lim):
    """Checks tmp_lim against new limits from the refmask run, and updates
    if they've changed."""
    
    dat = np.loadtxt('./music_region_file.txt')
    new_lim = np.zeros((3, 2))

    for i in range(3):
        # tmp_lim[i, :] = ([dat[:, i].min(), dat[:, i].max()])
        dmin = dat[:, i].min()
        dmax = dat[:, i].max()

        # print('dminmax')
        # print(dmin)
        # print(dmax)

        if dmin < old_lim[i, 0]:
            new_lim[i, 0] = dmin
        else:
            new_lim[i, 0] = old_lim[i, 0]
            
        if dmax > old_lim[i, 1]:
            new_lim[i, 1] = dmax
        else:
            new_lim[i, 1] = old_lim[i, 1]

    return new_lim

def dilate_region(lim):
    # Dilate region by 2% on each edge to account for padding
    lim = np.array([lim[:, 0] * 0.98,
                    lim[:, 1] * 1.02]).T

    return lim

# Highest level of refinement
lma = 12

# Max refinement region size (fine grid cells)
max_ref = 256

# Factor to expand region around centres
fac = 1.5 # 3.0

# Coordinates of halo centres (kpc/h)
# x = np.array([[47796.445, 52700.705, 47867.080],
#               [50452.671, 49535.092, 45333.135],
#               [50677.313, 49877.175, 45676.869]])  # kpc/h

x = np.array([[47823, 52603, 47954],
              [50983, 49567, 45720]], dtype=float)  # kpc/h

r = np.array([198.9, 143.0], dtype=float)

# Masses of haloes
m = np.array([1.8536e12, 7.044e11])  # Msol/h
M = m.sum()

# Factor to convert x into code units (0, 1)
ub = 100000.0  # kpc/h

# Factor to convert from code units to cells
uc = 2. ** lma

# Convert to code units
xx = x / ub
rr = r / ub

# Convert x
x /= ub
x *= uc

# Average x
# cen = np.mean(x, axis=0)

# Calculate centre of mass (kpc/h) and the difference between x and
# cen
cen = np.zeros(3, dtype=int)
dif_avg = np.zeros_like(x)

for i in range(3):
    cen[i] = (np.sum(x[:, i] * m.T) / M).astype(int)
    dif_avg[:, i] = x[:, i] - cen[i]

dif_avg = np.max(np.abs(dif_avg))
# cen = cen.astype(int)

# Find the largest difference between all of the halo centres, by
# rolling elements
dif_raw = np.zeros((x.shape[0], 3))
for i in range(x.shape[0] - 1):
    # dif_raw = np.max([np.abs(x-x[[2, 0, 1], :]), np.abs(x-x[[1, 2, 0], :])])  # TODO - generalise to nD
    x0 = x
    x1 = np.roll(x, 1, axis=0)  # rotate rows by one
    dif_raw = np.max([np.abs(dif_raw), np.abs(x-x1)])  # TODO - generalise to nD
    x0 = x1

# dif_avg = np.max(np.abs(x - [cen, cen, cen]))
dif_raw = np.ceil(dif_raw)
dif_avg = np.ceil(dif_avg)

print('---- averaged centre (fine cells): ', cen)
print('---- largest diff to raw coords:   ', dif_raw)
print('---- largest diff to avg coords:   ', dif_avg)

# Take the largest difference
dif = np.max([dif_raw, dif_avg])
dif = np.int(fac * dif)

# Check if larger than max_ref
ex = np.min([dif, max_ref])

print('---- extent (fine cells):          ', ex)

# Go from centre to corners
ii = cen - (ex // 2)

print('---- corners (fine cells):         ', ii)
print('---- running get_music_refmask')



# Run the music refinement mask code to find which particles are in
# the Lagrangian region
# run_refmask(cen/uc, dif/uc)

lim = []
ni = []
ii = []

# tmp_lim = np.zeros(shape=(3, 2))
# tmp_lim[:, 0] = 1e30   # xmin
# tmp_lim[:, 1] = -1e30  # xmax

# Instead of just choosing large values we can start off with Sergey's
# region
tmp_lim = np.array([[0.453125, 0.55078125],
                    [0.3828125, 0.484375],
                    [0.515625, 0.625]])

tmp_ni = np.zeros(3)
tmp_ii = np.zeros(3)

for j in range(xx.shape[0]):
    run_refmask(xx[j, :], rr[j] * 8.5, region=True)  # Onorbe et al (2014)

    print('xx rr')
    print(xx[j, :], rr[j])

    tmp_lim = update_lims(tmp_lim)

    # This file contains the coordinates of which will end up insde the
    # specified zoom region
        # tmp_ni[i] = int((tmp_lim[i, 1] - tmp_lim[i, 0]) * uc)
        # tmp_ii[i] = int(tmp_lim[i, 0] * uc)

    # lim.append(tmp_lim)
    # ni.append(tmp_ni)
    # ii.append(tmp_ii)

# Now run based on list of particle IDs from AHF
run_refmask(region=False)
tmp_lim = update_lims(tmp_lim)
tmp_lim = dilate_region(tmp_lim)

tmp_ni = ((tmp_lim[:, 1] - tmp_lim[:, 0]) * uc).astype(int)
tmp_ii = (tmp_lim[:, 0] * uc).astype(int)

# print(tmp_lim)
# print(tmp_ni)
# print(tmp_ii)

# print(ni)
# print(ii)
# with open('l.dat', 'w+') as f:
#     for i, c in enumerate('xyz'):
#         f.write('n{0} = {1}'.format(c, ni[i]))
#         f.write('i{0} = {1}'.format(c, ii[i]))

print('---- writing namelist')

# Take the largest of ni, but make sure it's not bigger than max_ref
# write_namelist(ii, min([max(ni), max_ref]), lma)
write_namelist(tmp_ii, tmp_ni, lma)
