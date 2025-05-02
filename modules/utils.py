import numpy as np
import numba as nb

import scipy.linalg
import math

@nb.njit
def nb_skew(data):
    """
    Calculate the skewness of a 1D array using Numba for JIT compilation.

    Parameters
    ----------
    data : np.ndarray
        1D array of data points.

    Returns
    -------
    float
        The skewness of the data.
    """

    n = len(data)
    mean = np.mean(data)
    std = np.std(data)

    den = n*std**3
    if den == 0:
        return np.nan
    return np.sum((data - mean)**3)/den

@nb.njit
def nb_kurtosis(data):
    """
    Calculate the kurtosis of a 1D array using Numba for JIT compilation.
    
    Parameters
    ----------
    data : np.ndarray
        1D array of data points.

    Returns
    -------
    float
        The kurtosis of the data.
    """

    n = len(data)
    mean = np.mean(data)
    std = np.std(data)
    den = n*std**4
    if den == 0:
        return np.nan
    return np.sum((data - mean)**4)/den - 3

@nb.njit
def bimodality_coefficient(data):
    """
    Calculate the bimodality coefficient of a 1D array using Numba for JIT compilation.

    Parameters
    ----------
    data : np.ndarray
        1D array of data points.

    Returns
    -------
    float
        The bimodality coefficient of the data.
    """

    n = len(data)
    s = nb_skew(data)
    k = nb_kurtosis(data)

    den = (k + 3*(n-1)**2/((n-2)*(n-3)))
    if den == 0:
        return np.nan
    return (s**2 + 1)/den

@nb.njit
def numba_pad(X, m):
    """
    Pad the array X with m elements at the beginning and end.
    This is used to avoid edge effects when calculating differences.

    Parameters
    ----------
    X : np.ndarray
        1D array to be padded.
    m : int
        Number of elements to pad at the beginning and end.

    Returns
    -------
    np.ndarray
        Padded 1D array.
    """

    Xnew = np.zeros(X.size + 2*m)
    Xnew[m:-m] = X
    Xnew[:m] = X[0]
    Xnew[-m:] = X[-1]
    return Xnew

@nb.njit
def differential_entropy_1D(X, verbose = False):
    """
    Calculate the differential entropy of a 1D array using Numba for JIT compilation.

    Parameters
    ----------
    X : np.ndarray
        1D array of data points.

    verbose : bool, optional
        If True, print the number of remaining values after removing NaN and Inf. Default is False.

    Returns
    -------
    float
        The differential entropy of the data.
    """

    X = X[~np.isnan(X)]
    X = X[~np.isinf(X)]

    if verbose:
        print("Removing nan and inf values, remaining values: ", X.size)
    
    n = X.size
    m = int(np.floor(np.sqrt(n) + 0.5))
    X = np.sort(X)

    X = numba_pad(X, m)
    differences = X[..., 2 * m:] - X[..., : -2 * m:]
    logs = np.log(n/(2*m) * differences)

    return np.mean(logs)/np.log(2)

@nb.njit
def sample_multivariate_gaussian(mean, cov, n_samples):
    """
    Sample from a multivariate Gaussian distribution using Cholesky decomposition.

    Parameters
    ----------
    mean : np.ndarray
        Mean vector of the Gaussian distribution.
    cov : np.ndarray
        Covariance matrix of the Gaussian distribution.
    n_samples : int
        Number of samples to generate.

    Returns
    -------
    np.ndarray
        Generated samples from the multivariate Gaussian distribution.
    """

    L = np.linalg.cholesky(cov)
    d = len(mean)
    z = np.random.randn(n_samples, d)
    return mean + np.dot(z, L.T)

@nb.njit
def get_gaussian_entropy_1D(sigma):
    """
    Calculate the differential entropy of a Gaussian distribution in 1D.

    Parameters
    ----------
    sigma : float
        Standard deviation of the Gaussian distribution.

    Returns
    -------
    float
        The differential entropy of the Gaussian distribution.
    """

    return 0.5*(1 + np.log(2*np.pi*sigma))/np.log(2)

@nb.njit
def get_MI_1D_output(output_samples, sigma_output):
    """
    Calculate the mutual information between the output samples and the input samples.
    The mutual information is defined as the difference between the differential entropy of the output samples
    and the differential entropy of a Gaussian distribution with a given standard deviation,
    which is expected to be the standard deviation associated with the output self-interactions.

    Used for fast processing units.

    Parameters
    ----------
    output_samples : np.ndarray
        2D array of output samples, where each row is a sample and each column is a feature.
    sigma_output : float
        Standard deviation associated with the output self-interactions.

    Returns
    -------
    float
        The mutual information between the output samples and the input samples.
    """

    H_outputs = differential_entropy_1D(output_samples)
    H1g2 = get_gaussian_entropy_1D(sigma_output)
    return H_outputs - H1g2

@nb.njit
def get_MI_1D_output_condentropy(output_samples):
    """
    Calculate the mutual information between the output samples and the input samples,
    using the conditional entropy of the output samples.
    The mutual information is defined as the difference between the differential entropy of the output samples
    and the average differential entropy of the output samples conditioned on each sample.

    Used for slow processing units.

    Parameters
    ----------
    output_samples : np.ndarray
        2D array of output samples, where each row is a sample and each column is a feature.

    Returns
    -------
    float
        The mutual information between the output samples and the input samples.
    """
    NSamples_input, _ = output_samples.shape
    
    H_OgI = np.zeros(NSamples_input)

    for i in range(NSamples_input):
        H_OgI[i] = differential_entropy_1D(output_samples[i])

    return differential_entropy_1D(output_samples.flatten()) - np.mean(H_OgI)

def generate_covariance_array(sigma_array, NRep, M):
    """
    Generate a covariance matrix array for a given set of standard deviations.
    The covariance matrices are generated using the Lyapunov equation.

    Parameters
    ----------
    sigma_array : np.ndarray
        1D array of standard deviations.
    NRep : int
        Number of repetitions for each standard deviation.
    M : int
        Size of the covariance matrix.

    Returns
    -------
    tuple
        A tuple containing:
        - SigmaMat_array: 3D array of covariance matrices.
        - AMat_array: 3D array of matrices used to generate the covariance matrices.
    """

    SigmaMat_array = np.zeros((sigma_array.size, NRep, M, M))
    AMat_array = np.zeros((sigma_array.size, NRep, M, M))

    for idx_sigma, sigma in enumerate(sigma_array):
        for idx_rep in range(NRep):
            A = np.random.randn(M, M)*sigma/np.sqrt(M)
            A[np.diag_indices(M)] = 1
            SigmaMat = scipy.linalg.solve_continuous_lyapunov(A, 2*np.eye(M))

            while not np.all(np.linalg.eigvals(SigmaMat) > 0):
                A = np.random.randn(M, M)*sigma/np.sqrt(M)
                A[np.diag_indices(M)] = 1
                SigmaMat = scipy.linalg.solve_continuous_lyapunov(A, 2*np.eye(M))
            
            SigmaMat_array[idx_sigma, idx_rep] = SigmaMat
            AMat_array[idx_sigma, idx_rep] = A

    return SigmaMat_array, AMat_array