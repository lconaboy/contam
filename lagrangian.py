import numpy as np

def write_namelist(ii, ex, lma):
    """Writes a partial namelist for use with cubic-mask"""

    with open('zoom.param', 'w') as f:
        f.write('&mask_params\n')
        f.write('levelmin = 8 \n')
        f.write('levelmax = {0:d}\n'.format(lma))
        f.write('ix = {0:d}\n'.format(ii[0]))
        f.write('iy = {0:d}\n'.format(ii[1]))
        f.write('iz = {0:d}\n'.format(ii[2]))
        f.write('nx = {0:d}\n'.format(ex))
        f.write('ny = {0:d}\n'.format(ex))
        f.write('nz = {0:d}\n'.format(ex))
        f.write('npad = 4\n')
        f.write("path = '' \n")
        f.write('/\n')


def run_refmask(cen, rad):
    import subprocess

    exe = ''
    inf = ''
    ini = ''
    rad = str(rad)
    xc = str(cen[0])
    yc = str(cen[1])
    zc = str(cen[2])

    subprocess.run([exe, '-inf', inf, '-ini', ini, '-xc', xc, '-yc', yc,
                    '-zc', zc, '-rad', rad])

# Highest level of refinement
lma = 11

# Max refinement region size (fine grid cells)
max_ref = 220

# Factor to expand region around centres
fac = 1.5 # 3.0

# Coordinates of halo centres (kpc/h)
# x = np.array([[47796.445, 52700.705, 47867.080],
#               [50452.671, 49535.092, 45333.135],
#               [50677.313, 49877.175, 45676.869]])  # kpc/h

x = np.array([[47823, 52603, 47954],
              [50983, 49567, 45720]], dtype=float)  # kpc/h

# Masses of haloes
m = np.array([1.8536e12, 7.044e11])  # Msol/h
M = m.sum()

# Factor to convert x into code units (0, 1)
ub = 100000.0  # kpc/h

# Factor to convert from code units to cells
uc = 2. ** lma

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
run_refmask(cen/uc, dif/uc)

# This file contains the coordinates of which will end up insde the
# specified zoom region
dat = np.loadtxt('./music_region_file.txt')
lim = []
ni = []
ii = []

for i in range(dat.shape[1]):
    lim.append([dat[:, i].min(), dat[:, i].max()])
    ni.append(int((lim[i][1] - lim[i][0]) * uc))
    ii.append(int(lim[i][0] * uc))

# with open('l.dat', 'w+') as f:
#     for i, c in enumerate('xyz'):
#         f.write('n{0} = {1}'.format(c, ni[i]))
#         f.write('i{0} = {1}'.format(c, ii[i]))

print('---- writing namelist')

# Take the largest of ni, but make sure it's not bigger than max_ref
write_namelist(ii, min([max(ni), max_ref]), lma)
