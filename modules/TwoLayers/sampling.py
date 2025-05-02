import numpy as np
import numba as nb

import sys
sys.path.append("..")
import utils

@nb.njit
def sample_integrated(NSamples, Sigma_input, Sigma_output, A_output, A_OI, g_OI):
    """
    Function to sample from an input-output system with nonlinear integration.

    Parameters
    ----------
    NSamples : int
        Number of samples to generate.
    Sigma_input : np.ndarray
        Covariance matrices of the input.
    Sigma_output : np.ndarray
        Covariance matrices of the output.
    A_output : np.ndarray
        Interaction matrix of the output.
    A_OI : np.ndarray
        Interaction matrix of the output with the input.
    g_OI : float
        Coupling strength of the output with the input.

    Returns
    -------
    input_samples : np.ndarray
        Samples from the input.
    output_samples : np.ndarray
        Samples from the output.
    """

    MOutput = A_output.shape[0]
    MInput = Sigma_input.shape[0]   

    L_output = np.linalg.cholesky(Sigma_output)

    input_samples = utils.sample_multivariate_gaussian(np.zeros(MInput), Sigma_input, NSamples).T
    mean_output = g_OI * np.dot(np.linalg.inv(A_output), np.tanh(np.dot(A_OI, input_samples)/MInput))

    output_samples = mean_output.T + np.dot(np.random.randn(NSamples, MOutput), L_output.T)
    
    return input_samples, output_samples.T

@nb.njit
def sample_nonlinearsum(NSamples, Sigma_input, Sigma_output, A_output, A_OI, g_OI):
    """
    Function to sample from an input-output system with nonlinear summation.

    Parameters
    ----------
    NSamples : int
        Number of samples to generate.
    Sigma_input : np.ndarray
        Covariance matrices of the input.
    Sigma_output : np.ndarray
        Covariance matrices of the output.
    A_output : np.ndarray
        Interaction matrix of the output.
    A_OI : np.ndarray
        Interaction matrix of the output with the input.
    g_OI : float
        Coupling strength of the output with the input.

    Returns
    -------
    input_samples : np.ndarray
        Samples from the input.
    output_samples : np.ndarray
        Samples from the output.
    """
    MOutput = A_output.shape[0]
    MInput = Sigma_input.shape[0]   

    L_output = np.linalg.cholesky(Sigma_output)

    input_samples = utils.sample_multivariate_gaussian(np.zeros(MInput), Sigma_input, NSamples).T
    mean_output = g_OI * np.dot(np.linalg.inv(A_output), np.dot(A_OI, np.tanh(input_samples))/MInput)
    
    output_samples = mean_output.T + np.dot(np.random.randn(NSamples, MOutput), L_output.T)
    
    return input_samples, output_samples.T