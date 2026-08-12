from header import *

# define global parameters
hlittle = Planck18.H0.value / 100

def t0_of_z(z):
    """
    A function to compute the characteristic 21-cm brightness temperature at a given redshift.
    TODO: find source for this equation
    """
    z = float(z)
    omegam = 0.315823
    omegab = 0.049387

    t0_of_z = 38.6 * hlittle * (omegab / 0.045) * \
        np.sqrt(0.27 / omegam * (1+z) / 10)
    return t0_of_z

def purge_inf(psn, ps21, psg, psx, kper, kpar):
    """
    A function to remove infinite-valued entries from the power spectrum.
    """
    psn_isinf = np.isinf(psn)
    col_isallinf = np.sum(psn_isinf, axis=0) == psn.shape[0]
    row_isallinf = np.sum(psn_isinf, axis=1) == psn.shape[1]
    col_isallinf_idcs = np.array(list(range(len(col_isallinf))))[col_isallinf]
    row_isallinf_idcs = np.array(list(range(len(row_isallinf))))[row_isallinf]
    psn = np.delete(psn, col_isallinf_idcs, axis=1)
    psn = np.delete(psn, row_isallinf_idcs, axis=0)
    ps21 = np.delete(ps21, col_isallinf_idcs, axis=1)
    ps21 = np.delete(ps21, row_isallinf_idcs, axis=0)
    psg = np.delete(psg, col_isallinf_idcs, axis=1)
    psg = np.delete(psg, row_isallinf_idcs, axis=0)
    psx = np.delete(psx, col_isallinf_idcs, axis=1)
    psx = np.delete(psx, row_isallinf_idcs, axis=0)
    kpar = np.delete(kpar, col_isallinf_idcs)
    kper = np.delete(kper, row_isallinf_idcs)
    return psn, ps21, psg, psx, kper, kpar

def centers_to_edges_log(centers):
    """
    A function to compute bin edges from bin centers in log space.
    """
    edges = np.zeros(len(centers) + 1)
    logsep = np.abs(np.log10(centers[1]) - np.log10(centers[0]))
    edges[0] = 10**(np.log10(centers[0]) - logsep/2)
    edges[1:] = 10**(np.log10(centers) + logsep/2)
    return edges

def purge_nan(psn, ps21, psg, psx, kper, kpar):
    """
    A function to remove NaN-valued entries from the power spectrum.
    """
    ps21_isnan = np.isnan(ps21)
    col_isallnan = np.sum(ps21_isnan, axis=0) == ps21.shape[0]
    row_isallnan = np.sum(ps21_isnan, axis=1) == ps21.shape[1]
    col_isallnan_idcs = np.array(list(range(len(col_isallnan))))[col_isallnan]
    row_isallnan_idcs = np.array(list(range(len(row_isallnan))))[row_isallnan]
    psn = np.delete(psn, col_isallnan_idcs, axis=1)
    psn = np.delete(psn, row_isallnan_idcs, axis=0)
    ps21 = np.delete(ps21, col_isallnan_idcs, axis=1)
    ps21 = np.delete(ps21, row_isallnan_idcs, axis=0)
    psg = np.delete(psg, col_isallnan_idcs, axis=1)
    psg = np.delete(psg, row_isallnan_idcs, axis=0)
    psx = np.delete(psx, col_isallnan_idcs, axis=1)
    psx = np.delete(psx, row_isallnan_idcs, axis=0)
    kpar = np.delete(kpar, col_isallnan_idcs)
    kper = np.delete(kper, row_isallnan_idcs)
    return psn, ps21, psg, psx, kper, kpar

def get_ps(obj, nbins, obj2=None, res=1.5):
    """
    Compute the cylindrical power spectrum of a 3D field.
    Parameters
    ----------
    obj : array_like
        The 3D field to compute the power spectrum of.
    nbins : tuple
        The number of bins in the kperp and kpar directions.
    obj2 : array_like, optional
        The second 3D field to compute the cross power spectrum with.
    res : float, optional
        Resolution of obj in cMpc/voxel side length.
    """

    box_len_side, _, box_len_los = obj.shape

    # let's try defining the bin edges
    kperp_bin_edges = np.logspace(np.log10(2*np.pi/box_len_side), \
                        np.log10(2*np.pi/1.5), nbins[0] + 1)

    # compute the cylindrical power spectrum
    ps_2d, kper_bins, var, _, kpar_bins = get_power(
        deltax=obj,
        boxlength=(box_len_side*res, box_len_side*res, box_len_los*res),
        deltax2=obj2,
        res_ndim=2,
        bins=kperp_bin_edges,
        dimensionless=False,
        log_bins=True,
        nthreads=1,
        return_sumweights=True,
        get_variance=True
    )

    kper_bins = np.array(kper_bins)
    kpar_bins = np.array(kpar_bins)[0]
    ps_2d = np.array(ps_2d)

    # remove NaN and zero bins
    m = kpar_bins > 1e-10
    kpar_bins = kpar_bins[m]
    ps_2d = ps_2d[:, m]
    var = var[:, m]
    mkperp = ~np.isnan(kper_bins)
    kper_bins = kper_bins[mkperp]
    ps_2d = ps_2d[mkperp, :]
    var = var[mkperp, :]

    # resample the power spectrum to a uniform grid in logspace
    kpar_new_edges = np.logspace(np.log10(kpar_bins.min()), \
                                np.log10(kpar_bins.max()), \
                                nbins[1] + 1)
    kpar_new_bins = kpar_new_edges[:-1] + np.diff(kpar_new_edges) / 2

    new_ps = np.zeros((kper_bins.shape[0], nbins[1]))
    new_var = np.zeros((kper_bins.shape[0], nbins[1]))

    modes = np.zeros(nbins[1])
    idxs = np.digitize(kpar_bins, kpar_new_edges)
    for i in range(len(kpar_new_edges) - 1):
        m = idxs == i
        new_ps[..., i] = np.nanmean(ps_2d[..., m], axis=-1)
        new_var[..., i] = np.nanmean(var[..., m], axis=-1)
        modes[i] = np.sum(m)

    kpar_bins = kpar_new_bins
    ps_2d = new_ps
    var = new_var

    return ps_2d, kper_bins, kpar_bins, modes, var