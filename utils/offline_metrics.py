import pdb

import numpy as np
import scipy.stats as stats

def mse(x, y, rowvar=False):
    '''
    Calculates mean-squared error between given input and true signal
    inputs:
        x   -   1 or 2d array of observations and variables. By defaults, each row is an observation, each column a
                variable
        y   -   Additional set of observations and variables to compare to. should be the same size as x
        rowvar (optional)   -   If rowvar is True, then each row is a variable, each column an observation

    outputs:
        mse -   scalar or 1d array of mse for each variable across all observations
    '''
    if len(x.shape) > 1:
        ax = 1 if rowvar else 0
        mse = (np.square(x - y)).mean(axis=ax)
    else:
        mse = (np.square(x - y)).mean()

    return mse
def vaf(y, yhat, rowvar=False):
    '''
    Calculates variance-accounted-for between true signal (y) and another signal (yhat)
    inputs:
        y   -   1 or 2d NUMPY array of observations and variables. By defaults, each row is an observation, each column a
                variable
        yhat   -   Additional set of observations and variables to compare to. should be the same size as y
        rowvar (optional)   -   If rowvar is True, then each row is a variable, each column an observation

    outputs:
        vaf -   scalar or 1d array of vaf for each variable across all observations
    '''
    if len(y.shape) > 1:
        ax = 1 if rowvar else 0
        vaf = 1 - (np.sum(np.square(y - yhat), axis=ax) / np.sum(np.square(y - y.mean(axis=ax)), axis=ax))
    else:
        vaf = 1 - (np.sum(np.square(y - yhat)) / np.sum(np.square(y - y.mean())))

    return vaf
def corrcoef(x, y, rowvar=False):
    '''
    Calculates pairwise correlation coefficients for two (NxM) ndarrays.
    inputs:
        x   -   1 or 2d array of observations and variables. By defaults, each row is an observation, each column a
                variable
        y   -   Additional set of observations and variables to compare to. should be the same size as x
        rowvar (optional)   -   If rowvar is True, then each row is a variable, each column an observation
    outputs:
        cc  -   scalar or 1d array of correlation coefficients for each pair of variables in x and y
    '''
    if len(x.shape) > 1:
        num_vars = x.shape[0] if rowvar else x.shape[1]
        cc = np.diag(np.corrcoef(x, y, rowvar=rowvar)[:num_vars, num_vars:])
    else:
        cc = np.corrcoef(x, y)[0, 1]

    return cc
def kldiv(f, g):
    '''
    Calculate the Kullback-Leibler Divergence between two different PMFs. Assumes that both distributions share equal
    'bin' sizes, doesn't adjust for this.

    Some notes: KL divergence is _only_ defined when g(x) > 0 and f(x) > 0. Also, KL divergence is not a true
    distance metric, because its not symmetric (kldiv(f,g) is not guaranteed to be kldiv(g, f)). It also doesn't really
    care about number of samples, assuming everything is a true distribution. Here, computing with natural log, not base
    2.

    Inputs:
        f (ndarray):
            nx1 ndarray representing the pmf of a function across multiple bins (think these should be evenly spaced).
            In general, this should be your observed distribution (ground truth)
        g (ndarray):
            nx1 ndarray representing the pmf of a funciton across multiple bins (think these should be evenly spaced).
            Should match the size of f. In general, this should be your model (predicted distribution).

    Outputs:
        kldiv (float):
            Returns the KL-divergence between the two pmfs, taken as : Sig_x(f(x) * log(f(x)/g(x)) where x is an element
            of [0, n).
    '''
    zero_mask = np.logical_and(f != 0, g != 0)
    kld = np.sum(f[zero_mask] * np.log(f[zero_mask]/g[zero_mask]))
    return kld

def bin_psd(psd, f, numbins):
    binned_psd = np.zeros((numbins, psd.shape[1], psd.shape[2]))
    bins = np.linspace(np.max(f)/numbins, np.max(f), numbins)
    for i in np.arange(len(bins)):
        binmask = np.where(f <= bins[i])[0]
        binned_psd[i,:,:] = np.mean(psd[binmask,:,:], axis=0)

    return binned_psd, np.linspace(0, np.max(f), numbins+1)

def kl_div_gaussian(pred, binedges, binsize):
    mean = np.mean(pred)
    std = np.std(pred)
    fitted_gaussian = stats.norm.pdf(binedges[:-1] + binsize / 2, loc=mean, scale=std)
    pred_hist, _ = np.histogram(pred, bins=binedges, density=True)
    pred_hist /= np.sum(pred_hist)
    fitted_gaussian /= np.sum(fitted_gaussian)
    kl_div_gaussian = kldiv(pred_hist, fitted_gaussian)

    return kl_div_gaussian