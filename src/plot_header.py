"""
Generic header for plotting scripts

Samuel Gagnon-Hartman, 2026.
Scuola Normale Superiore, Pisa, Italy.
"""

import matplotlib.pyplot as plt
rc = {"font.family" : "serif", 
      "mathtext.fontset" : "stix"}
plt.rcParams.update(rc)
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
plt.rcParams.update({'font.size': 14})
# plt.style.use('default')
import matplotlib as mpl
label_size = 20
font_size = 30
mpl.rcParams['xtick.labelsize'] = label_size
mpl.rcParams['ytick.labelsize'] = label_size