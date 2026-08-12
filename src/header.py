"""
Generic header for main scripts

Samuel Gagnon-Hartman, 2026.
Scuola Normale Superiore, Pisa, Italy.
"""

import os
import h5py
import pickle

import py21cmfast as p21c
import numpy as np

from astropy.cosmology import Planck18, z_at_value
from astropy import units as u
from astropy.constants import c, k_B, m_p

from astropy.cosmology.units import littleh
from powerbox import get_power
from py21cmsense import GaussianBeam, Observatory, Observation, PowerSpectrum

import astropy.units as u
from astropy.constants import c


# NOTE the following are questionable
from scipy.optimize import curve_fit

def sigmoid(x, a, b, c):
    return a/(1 + np.exp(-b*(x - c)))

from tqdm import tqdm