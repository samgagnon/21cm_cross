from utils import *
from config import *

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--fgd_model', type=str, default='moderate',
                    help='Foreground model to use')
parser.add_argument('--sz', type=float, default=1e-2,
                    help='SZ effect parameter')
parser.add_argument('--lya', type=float, default=42.25,
                    help='Lya luminosity limit')
args = parser.parse_args()

fgd_model = args.fgd_model
sz = args.sz
lya_lim = args.lya
fov = 1 # TODO what does this mean?

# load in BT field
hdf = h5py.File(f'../../subaru/data/id_{ID}.h5', 'r')
los_z = dict(hdf['lightcones'].attrs)['lightcone_redshifts']
lc_sidelength = 300
lc_side_voxels = 200
los_dc = Planck18.comoving_distance(los_z).value - Planck18.comoving_distance(5.00).value
side_dc = np.linspace(0, lc_sidelength, lc_side_voxels)
dL = Planck18.luminosity_distance(los_z).to('cm').value
bt = hdf['lightcones']['brightness_temp'][:]

# load astro parameters
ALPHA_STAR = hdf['simulation_parameters']['astro_params'].attrs['ALPHA_STAR']
SIGMA_STAR = hdf['simulation_parameters']['astro_params'].attrs['SIGMA_STAR']
UPPER_STELLAR_TURNOVER_MASS = hdf['simulation_parameters']['astro_params'].attrs['UPPER_STELLAR_TURNOVER_MASS']
M_TURN = hdf['simulation_parameters']['astro_params'].attrs['M_TURN']
UPPER_STELLAR_TURNOVER_INDEX = hdf['simulation_parameters']['astro_params'].attrs['UPPER_STELLAR_TURNOVER_INDEX']
F_STAR10 = hdf['simulation_parameters']['astro_params'].attrs['F_STAR10']
SIGMA_SFR_LIM = hdf['simulation_parameters']['astro_params'].attrs['SIGMA_SFR_LIM']
SIGMA_SFR_INDEX = hdf['simulation_parameters']['astro_params'].attrs['SIGMA_SFR_INDEX']
t_STAR = hdf['simulation_parameters']['astro_params'].attrs['t_STAR']

# load varying parameters
ALPHA_ESC = hdf['simulation_parameters']['varying_params'].attrs['ALPHA_ESC']
ALPHA_STAR = hdf['simulation_parameters']['varying_params'].attrs['ALPHA_STAR']
F_ESC10 = hdf['simulation_parameters']['varying_params'].attrs['F_ESC10']
L_X = hdf['simulation_parameters']['varying_params'].attrs['L_X']
NU_X_THRESH = hdf['simulation_parameters']['varying_params'].attrs['NU_X_THRESH']
SIGMA_8 = hdf['simulation_parameters']['varying_params'].attrs['SIGMA_8']
random_seed = hdf['simulation_parameters']['varying_params'].attrs['random_seed']

# populate galaxy light
ngal = np.zeros_like(bt)
with np.load(f'../data/{ID}/ngal.npz') as data:
    # 1. View all array names stored in the file
    print("Available arrays:", data.files)
    for n_coeval in data.files:
        print(f"Processing coeval {n_coeval}...")
        coeval_data = data[n_coeval]
        x, y, z = coeval_data
        ngal[x, y, z] += int(n_coeval)

ngal /= 1.5**2 # convert to cMpc^-3