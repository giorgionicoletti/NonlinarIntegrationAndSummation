import numpy as np
import numba as nb
import math

import sys
sys.path.append("../..")

import utils

@nb.njit
def get_Vfun_Plus(n, arg, var):
    """
    Nonlinear function appearing in the tanh integral approximation.

    Parameters
    ----------
    n : int
        The index of the term in the series expansion.
    arg : float
        The argument of the function.
    var : float
        The variance of the distribution.

    Returns
    -------
    V : float
        The value of the function at the given argument and variance.
    """

    V = np.exp(2*n**2*var - 2*n*arg)*math.erfc((2*n*var - arg)/(np.sqrt(2*var)))
    return V

@nb.njit
def get_Vfun_Minus(n, arg, var):
    """
    Nonlinear function appearing in the tanh integral approximation.

    Parameters
    ----------
    n : int
        The index of the term in the series expansion.
    arg : float
        The argument of the function.
    var : float
        The variance of the distribution.

    Returns
    -------
    V : float
        The value of the function at the given argument and variance.
    """

    V = np.exp(2*n**2*var + 2*n*arg)*math.erfc((arg + 2*n*var)/(np.sqrt(2*var)))
    return V

@nb.njit
def sample_nonlinearsum(NSamples,
                        Sigma_input, Sigma_processing, Sigma_output,
                        A_input, A_processing, A_output,
                        A_PI, A_OP, g_PI, g_OP,
                        auto_NMax = True, tanh_threshold = 1e-2, NMax_approx = 5,
                        verbose = False):
    """
    Function to sample from an input-processing-output system with nonlinear summation and
    a fast processing unit.
    The function uses a series expansion to approximate the integral of the nonlinear interactions.

    Parameters
    ----------
    NSamples : int
        Number of samples to generate.
    Sigma_input : np.ndarray
        Covariance matrices of the input.
    Sigma_processing : np.ndarray
        Covariance matrices of the processing unit.
    Sigma_output : np.ndarray
        Covariance matrices of the output.
    A_input : np.ndarray
        Interaction matrix of the input.
    A_processing : np.ndarray
        Interaction matrix of the processing unit.
    A_output : np.ndarray
        Interaction matrix of the output.
    A_PI : np.ndarray
        Interaction matrix of the input with the processing unit.
    A_OP : np.ndarray
        Interaction matrix of the processing unit with the output.
    g_PI : float
        Coupling strength of the input with the processing unit.
    g_OP : float
        Coupling strength of the processing unit with the output.
    auto_NMax : bool, optional
        If True, the function will automatically determine the number of terms in the series expansion
        based on the threshold. The default is True.
    tanh_threshold : float, optional
        Threshold for the series expansion. The default is 1e-2.
    NMax_approx : int, optional
        Maximum number of terms in the series expansion. The default is 5.
        Only used if auto_NMax is False.
    verbose : bool, optional
        If True, the function will print the number of terms in the series expansion. The default is False.

    Returns
    -------
    input_samples : np.ndarray
        Samples from the input.
    processing_samples : np.ndarray
        Samples from the processing unit.
    output_samples : np.ndarray
        Samples from the output.
    """

    MInput = Sigma_input.shape[0]
    MProcessing = Sigma_processing.shape[0]
    MOutput = Sigma_output.shape[0]

    L_input = np.linalg.cholesky(Sigma_input)
    L_output_T = np.linalg.cholesky(Sigma_output).T
    L_processing_T = np.linalg.cholesky(Sigma_processing).T

    input_samples = np.dot(L_input, np.random.randn(MInput, NSamples))
    v_input = np.dot(A_PI, np.tanh(input_samples))/MInput
    mean_processing = g_PI * np.dot(np.linalg.inv(A_processing), v_input)

    processing_samples = np.zeros((MProcessing, NSamples))
    output_samples = np.zeros((MOutput, NSamples))

    A_output_inv = np.linalg.inv(A_output)

    for idx in range(NSamples):
        processing_samples[:, idx] = mean_processing[:, idx] + np.dot(np.random.randn(MProcessing), L_processing_T)

        app_tanh = np.zeros(MProcessing)
        for k in range(MProcessing):
            app_tanh[k] = math.erf(mean_processing[k, idx]/(np.sqrt(2*Sigma_processing[k, k])))

            if auto_NMax:
                n = 1
                while True:
                    temp_sum  = get_Vfun_Plus(n, mean_processing[k, idx], Sigma_processing[k, k])
                    temp_sum -= get_Vfun_Minus(n, mean_processing[k, idx], Sigma_processing[k, k])

                    if np.abs(temp_sum) < tanh_threshold or np.isnan(temp_sum) or np.isinf(temp_sum):
                        if verbose:
                            print(f"Threshold reached at n = {n}")
                        break
                    else:
                        app_tanh[k] += (-1)**n*temp_sum
                        n += 1

            else:
                for n in range(1, NMax_approx):
                    temp_sum  = get_Vfun_Plus(n, mean_processing[k, idx], Sigma_processing[k, k])
                    temp_sum -= get_Vfun_Minus(n, mean_processing[k, idx], Sigma_processing[k, k])
                    app_tanh[k] += (-1)**n*temp_sum
            
        mean_output = g_OP/MProcessing * A_output_inv @ A_OP @ app_tanh
        output_samples[:, idx] = mean_output + np.dot(np.random.randn(MOutput), L_output_T)
            
    return input_samples, processing_samples, output_samples

@nb.njit
def sample_nonlinearsum_wmean(NSamples, thetaP,
                              Sigma_input, Sigma_processing, Sigma_output,
                              A_input, A_processing, A_output,
                              A_PI, A_OP, g_PI, g_OP,
                              auto_NMax = True, tanh_threshold = 1e-2, NMax_approx = 5,
                              verbose = False):
    """
    Function to sample from an input-processing-output system with nonlinear summation and
    a fast processing unit. In this case, the processing unit has a constant bias or mean.
    The function uses a series expansion to approximate the integral of the nonlinear interactions.

    Parameters
    ----------
    NSamples : int
        Number of samples to generate.
    thetaP : np.ndarray
        Mean of the processing unit.
    Sigma_input : np.ndarray
        Covariance matrices of the input.
    Sigma_processing : np.ndarray
        Covariance matrices of the processing unit.
    Sigma_output : np.ndarray
        Covariance matrices of the output.
    A_input : np.ndarray
        Interaction matrix of the input.
    A_processing : np.ndarray
        Interaction matrix of the processing unit.
    A_output : np.ndarray
        Interaction matrix of the output.
    A_PI : np.ndarray
        Interaction matrix of the input with the processing unit.
    A_OP : np.ndarray
        Interaction matrix of the processing unit with the output.
    g_PI : float
        Coupling strength of the input with the processing unit.
    g_OP : float
        Coupling strength of the processing unit with the output.
    auto_NMax : bool, optional
        If True, the function will automatically determine the number of terms in the series expansion
        based on the threshold. The default is True.
    tanh_threshold : float, optional
        Threshold for the series expansion. The default is 1e-2.
    NMax_approx : int, optional
        Maximum number of terms in the series expansion. The default is 5.
        Only used if auto_NMax is False.
    verbose : bool, optional
        If True, the function will print the number of terms in the series expansion. The default is False.

    Returns
    -------
    input_samples : np.ndarray
        Samples from the input.
    processing_samples : np.ndarray
        Samples from the processing unit.
    output_samples : np.ndarray
        Samples from the output.
    """
    
    MInput = Sigma_input.shape[0]
    MProcessing = Sigma_processing.shape[0]
    MOutput = Sigma_output.shape[0]

    L_input = np.linalg.cholesky(Sigma_input)
    L_output_T = np.linalg.cholesky(Sigma_output).T
    L_processing_T = np.linalg.cholesky(Sigma_processing).T

    input_samples = np.dot(L_input, np.random.randn(MInput, NSamples))
    v_input = np.dot(A_PI, np.tanh(input_samples))/MInput
    mean_processing = g_PI * np.dot(np.linalg.inv(A_processing), v_input) - thetaP[..., None]

    processing_samples = np.zeros((MProcessing, NSamples))
    output_samples = np.zeros((MOutput, NSamples))

    A_output_inv = np.linalg.inv(A_output)

    for idx in range(NSamples):
        processing_samples[:, idx] = mean_processing[:, idx] + np.dot(np.random.randn(MProcessing), L_processing_T)

        app_tanh = np.zeros(MProcessing)
        for k in range(MProcessing):
            app_tanh[k] = math.erf(mean_processing[k, idx]/(np.sqrt(2*Sigma_processing[k, k])))

            if auto_NMax:
                n = 1
                while True:
                    temp_sum  = get_Vfun_Plus(n, mean_processing[k, idx], Sigma_processing[k, k])
                    temp_sum -= get_Vfun_Minus(n, mean_processing[k, idx], Sigma_processing[k, k])

                    if np.abs(temp_sum) < tanh_threshold or np.isnan(temp_sum) or np.isinf(temp_sum):
                        if verbose:
                            print(f"Threshold reached at n = {n}")
                        break
                    else:
                        app_tanh[k] += (-1)**n*temp_sum
                        n += 1

            else:
                for n in range(1, NMax_approx):
                    temp_sum  = get_Vfun_Plus(n, mean_processing[k, idx], Sigma_processing[k, k])
                    temp_sum -= get_Vfun_Minus(n, mean_processing[k, idx], Sigma_processing[k, k])
                    app_tanh[k] += (-1)**n*temp_sum
            
        mean_output = g_OP/MProcessing * A_output_inv @ A_OP @ app_tanh
        output_samples[:, idx] = mean_output + np.dot(np.random.randn(MOutput), L_output_T)
            
    return input_samples, processing_samples, output_samples

@nb.njit
def sample_integrated(NSamples,
                      Sigma_input, Sigma_processing, Sigma_output,
                      A_input, A_processing, A_output,
                      A_PI, A_OP, g_PI, g_OP,
                      auto_NMax = True, tanh_threshold = 1e-2, NMax_approx = 5,
                      verbose = False):
    """
    Function to sample from an input-processing-output system with nonlinear integration and
    a fast processing unit.
    The function uses a series expansion to approximate the integral of the nonlinear interactions.

    Parameters
    ----------
    NSamples : int
        Number of samples to generate.
    Sigma_input : np.ndarray
        Covariance matrices of the input.
    Sigma_processing : np.ndarray
        Covariance matrices of the processing unit.
    Sigma_output : np.ndarray
        Covariance matrices of the output.
    A_input : np.ndarray
        Interaction matrix of the input.
    A_processing : np.ndarray
        Interaction matrix of the processing unit.
    A_output : np.ndarray
        Interaction matrix of the output.
    A_PI : np.ndarray
        Interaction matrix of the input with the processing unit.
    A_OP : np.ndarray
        Interaction matrix of the processing unit with the output.
    g_PI : float
        Coupling strength of the input with the processing unit.
    g_OP : float
        Coupling strength of the processing unit with the output.
    auto_NMax : bool, optional
        If True, the function will automatically determine the number of terms in the series expansion
        based on the threshold. The default is True.
    tanh_threshold : float, optional
        Threshold for the series expansion. The default is 1e-2.
    NMax_approx : int, optional
        Maximum number of terms in the series expansion. The default is 5.
        Only used if auto_NMax is False.
    verbose : bool, optional
        If True, the function will print the number of terms in the series expansion. The default is False.

    Returns
    -------
    input_samples : np.ndarray
        Samples from the input.
    processing_samples : np.ndarray
        Samples from the processing unit.
    output_samples : np.ndarray
        Samples from the output.
    """

    MInput = Sigma_input.shape[0]
    MProcessing = Sigma_processing.shape[0]
    MOutput = Sigma_output.shape[0]
    
    L_input = np.linalg.cholesky(Sigma_input)
    L_output_T = np.linalg.cholesky(Sigma_output).T
    L_processing_T = np.linalg.cholesky(Sigma_processing).T

    A_output_inv = np.linalg.inv(A_output)

    input_samples = np.dot(L_input, np.random.randn(MInput, NSamples))
    v_input = np.tanh(np.dot(A_PI, input_samples)/MInput)
    mean_processing = g_PI * np.dot(np.linalg.inv(A_processing), v_input)

    processing_samples = np.zeros((MProcessing, NSamples))
    output_samples = np.zeros((MOutput, NSamples))

    for idx in range(NSamples):
        processing_samples[:, idx] = mean_processing[:, idx] + np.dot(np.random.randn(MProcessing), L_processing_T)

        app_tanh = np.zeros(MOutput)
        mean_change = 1/MProcessing * np.dot(A_OP, mean_processing[:, idx])
        sigma_change = 1/MProcessing**2 * np.diag(np.dot(A_OP, np.dot(Sigma_processing, A_OP.T)))

        for j in range(MOutput):
            app_tanh[j] = math.erf(mean_change[j]/(np.sqrt(2*sigma_change[j])))

            if auto_NMax:
                n = 1
                while True:
                    temp_sum  = get_Vfun_Plus(n, mean_change[j], sigma_change[j])
                    temp_sum -= get_Vfun_Minus(n, mean_change[j], sigma_change[j])

                    if np.abs(temp_sum) < tanh_threshold or np.isnan(temp_sum) or np.isinf(temp_sum):
                        if verbose:
                            print(f"Threshold reached at n = {n}")
                        break
                    else:
                        app_tanh[j] += (-1)**n*temp_sum
                        n += 1

            else:
                for n in range(1, NMax_approx):
                    temp_sum  = get_Vfun_Plus(n, mean_change[j], sigma_change[j])
                    temp_sum -= get_Vfun_Minus(n, mean_change[j], sigma_change[j])
                    app_tanh[j] += (-1)**n*temp_sum
            
        mean_output = g_OP * A_output_inv @ app_tanh
        output_samples[:, idx] = mean_output + np.dot(np.random.randn(MOutput), L_output_T)
            
    return input_samples, processing_samples, output_samples

@nb.njit
def sample_integrated_wmean(NSamples, thetaP,
                            Sigma_input, Sigma_processing, Sigma_output,
                            A_input, A_processing, A_output,
                            A_PI, A_OP, g_PI, g_OP,
                            auto_NMax = True, tanh_threshold = 1e-2, NMax_approx = 5,
                            verbose = False):
    """
    Function to sample from an input-processing-output system with nonlinear integration and
    a fast processing unit. In this case, the processing unit has a constant bias or mean.
    The function uses a series expansion to approximate the integral of the nonlinear interactions.

    Parameters
    ----------
    NSamples : int
        Number of samples to generate.
    thetaP : np.ndarray
        Mean of the processing unit.
    Sigma_input : np.ndarray
        Covariance matrices of the input.
    Sigma_processing : np.ndarray
        Covariance matrices of the processing unit.
    Sigma_output : np.ndarray
        Covariance matrices of the output.
    A_input : np.ndarray
        Interaction matrix of the input.
    A_processing : np.ndarray
        Interaction matrix of the processing unit.
    A_output : np.ndarray
        Interaction matrix of the output.
    A_PI : np.ndarray
        Interaction matrix of the input with the processing unit.
    A_OP : np.ndarray
        Interaction matrix of the processing unit with the output.
    g_PI : float
        Coupling strength of the input with the processing unit.
    g_OP : float
        Coupling strength of the processing unit with the output.
    auto_NMax : bool, optional
        If True, the function will automatically determine the number of terms in the series expansion
        based on the threshold. The default is True.
    tanh_threshold : float, optional
        Threshold for the series expansion. The default is 1e-2.
    NMax_approx : int, optional
        Maximum number of terms in the series expansion. The default is 5.
        Only used if auto_NMax is False.
    verbose : bool, optional
        If True, the function will print the number of terms in the series expansion. The default is False.

    Returns
    -------
    input_samples : np.ndarray
        Samples from the input.
    processing_samples : np.ndarray
        Samples from the processing unit.
    output_samples : np.ndarray
        Samples from the output.
    """

    MInput = Sigma_input.shape[0]
    MProcessing = Sigma_processing.shape[0]
    MOutput = Sigma_output.shape[0]
    
    L_input = np.linalg.cholesky(Sigma_input)
    L_output_T = np.linalg.cholesky(Sigma_output).T
    L_processing_T = np.linalg.cholesky(Sigma_processing).T

    A_output_inv = np.linalg.inv(A_output)

    input_samples = np.dot(L_input, np.random.randn(MInput, NSamples))
    v_input = np.tanh(np.dot(A_PI, input_samples)/MInput)
    mean_processing = g_PI * np.dot(np.linalg.inv(A_processing), v_input)

    processing_samples = np.zeros((MProcessing, NSamples))
    output_samples = np.zeros((MOutput, NSamples))

    for idx in range(NSamples):
        processing_samples[:, idx] = mean_processing[:, idx] + np.dot(np.random.randn(MProcessing), L_processing_T)

        app_tanh = np.zeros(MOutput)
        mean_change = 1/MProcessing * (np.dot(A_OP, mean_processing[:, idx]) - np.dot(A_OP, thetaP))
        sigma_change = 1/MProcessing**2 * np.diag(np.dot(A_OP, np.dot(Sigma_processing, A_OP.T)))

        for j in range(MOutput):
            app_tanh[j] = math.erf(mean_change[j]/(np.sqrt(2*sigma_change[j])))

            if auto_NMax:
                n = 1
                while True:
                    temp_sum  = get_Vfun_Plus(n, mean_change[j], sigma_change[j])
                    temp_sum -= get_Vfun_Minus(n, mean_change[j], sigma_change[j])

                    if np.abs(temp_sum) < tanh_threshold or np.isnan(temp_sum) or np.isinf(temp_sum):
                        if verbose:
                            print(f"Threshold reached at n = {n}")
                        break
                    else:
                        app_tanh[j] += (-1)**n*temp_sum
                        n += 1

            else:
                for n in range(1, NMax_approx):
                    temp_sum  = get_Vfun_Plus(n, mean_change[j], sigma_change[j])
                    temp_sum -= get_Vfun_Minus(n, mean_change[j], sigma_change[j])
                    app_tanh[j] += (-1)**n*temp_sum
            
        mean_output = g_OP * A_output_inv @ app_tanh
        output_samples[:, idx] = mean_output + np.dot(np.random.randn(MOutput), L_output_T)
            
    return input_samples, processing_samples, output_samples


@nb.njit(parallel = True, fastmath = True)
def sample_gXg_grid(g_PI_array, g_OP_array, NSamples,
                    Sigma_input_array, Sigma_processing_array, A_processing_array,
                    Sigma_output, A_output,
                    tanh_threshold,
                    sigma_OP = 1, sigma_PI = 1):
    """
    Function to sample from an input-processing-output system with nonlinear integration and
    a fast processing unit. The function uses a grid of g_PI and g_OP values to sample the system,
    where g_PI is the coupling strength of the input with the processing unit and g_OP is the coupling
    strength of the processing unit with the output.
    The function uses a series expansion to approximate the integral of the nonlinear interactions.

    Parameters
    ----------
    g_PI_array : np.ndarray
        Array of g_PI values.
    g_OP_array : np.ndarray
        Array of g_OP values.
    NSamples : int
        Number of samples to generate.
    Sigma_input_array : np.ndarray
        Covariance matrices of the input.
    Sigma_processing_array : np.ndarray
        Covariance matrices of the processing unit.
    A_processing_array : np.ndarray
        Interaction matrices of the processing unit.
    Sigma_output : np.ndarray
        Covariance matrices of the output.
    A_output : np.ndarray
        Interaction matrix of the output.
    tanh_threshold : float
        Threshold for the series expansion. The default is 1e-2.
    sigma_OP : float, optional
        Standard deviation of the interaction matrix from the processing unit to the output.
        The default is 1.
    sigma_PI : float, optional
        Standard deviation of the interaction matrix from the input to the processing unit.
        The default is 1.

    Returns
    -------
    MI_IO_INT : np.ndarray
        Mutual information between the input and output with integrated processing.
    MI_IO_NS : np.ndarray
        Mutual information between the input and output with nonlinear summation.
    """
    
    MInput = np.shape(Sigma_input_array)[-1]
    MProcessing = np.shape(Sigma_processing_array)[-1]
    MOutput = np.shape(Sigma_output)[0]

    NRep = Sigma_input_array.shape[0]

    MI_IO_INT = np.zeros((NRep, g_PI_array.size, g_OP_array.size), dtype = np.float64)
    MI_IO_NS = np.zeros((NRep, g_PI_array.size, g_OP_array.size), dtype = np.float64)

    L_output = np.linalg.cholesky(Sigma_output)
    A_output_inv = np.linalg.inv(A_output)
    H1g2 = utils.get_gaussian_entropy_1D(Sigma_output[0,0])

    for idx_rep in nb.prange(NRep):
        print(f"Mpro: {MProcessing}, rep: {idx_rep}/{NRep}")
        Sigma_input = Sigma_input_array[idx_rep]
        Sigma_processing = Sigma_processing_array[idx_rep]

        L_input = np.linalg.cholesky(Sigma_input)

        A_processing = A_processing_array[idx_rep]
        A_processing_inv = np.linalg.inv(A_processing)
        #L_processing = np.linalg.cholesky(Sigma_processing)

        A_PI = np.random.randn(MProcessing, MInput)*sigma_PI
        A_OP = np.random.randn(MOutput, MProcessing)*sigma_OP

        input_samples = np.dot(L_input, np.random.randn(MInput, NSamples))
        output_samples = np.dot(np.random.randn(NSamples, MOutput), L_output.T).T

        v_input_NS = np.dot(A_PI, np.tanh(input_samples))/MInput
        v_input_INT = np.tanh(np.dot(A_PI, input_samples)/MInput)

        sigma_change = 1/MProcessing**2 * np.diag(np.dot(A_OP, np.dot(Sigma_processing, A_OP.T)))

        for idx_g_in in range(g_PI_array.size):
            g_PI = g_PI_array[idx_g_in]

            mean_processing_NS = g_PI * np.dot(A_processing_inv, v_input_NS)
            mean_processing_INT = g_PI * np.dot(A_processing_inv, v_input_INT)

            app_tanh_NS = np.zeros((MProcessing, NSamples))
            app_tanh_INT = np.zeros((MOutput, NSamples))

            for i in range(NSamples):
                for k in range(MProcessing):
                    app_tanh_NS[k, i] = math.erf(mean_processing_NS[k, i]/(np.sqrt(2*Sigma_processing[k, k])))

                    n = 1
                    while True:
                        temp_sum  = get_Vfun_Plus(n, mean_processing_NS[k, i], Sigma_processing[k, k])
                        temp_sum -= get_Vfun_Minus(n, mean_processing_NS[k, i], Sigma_processing[k, k])

                        if np.abs(temp_sum) < tanh_threshold or np.isnan(temp_sum) or np.isinf(temp_sum):
                            break
                        else:
                            app_tanh_NS[k, i] += (-1)**n*temp_sum
                            n += 1

                mean_change_INT = 1/MProcessing * np.dot(A_OP, mean_processing_INT[:, i])
                for j in range(MOutput):
                    app_tanh_INT[j, i] = math.erf(mean_change_INT[j]/(np.sqrt(2*sigma_change[j])))

                    n = 1
                    while True:
                        temp_sum  = get_Vfun_Plus(n, mean_change_INT[j], sigma_change[j])
                        temp_sum -= get_Vfun_Minus(n, mean_change_INT[j], sigma_change[j])

                        if np.abs(temp_sum) < tanh_threshold or np.isnan(temp_sum) or np.isinf(temp_sum):
                            break
                        else:
                            app_tanh_INT[j, i] += (-1)**n*temp_sum
                            n += 1

            for idx_g_pro in range(g_OP_array.size):
                g_OP = g_OP_array[idx_g_pro]

                mean_output_NS = g_OP/MProcessing * A_output_inv @ A_OP @ app_tanh_NS
                mean_output_INT = g_OP * A_output_inv @ app_tanh_INT

                output_samples_NS = mean_output_NS + output_samples
                output_samples_INT = mean_output_INT + output_samples

                MI_IO_INT[idx_rep, idx_g_in, idx_g_pro] = utils.differential_entropy_1D(output_samples_INT[0, ~np.isnan(output_samples_INT[0])])
                MI_IO_NS[idx_rep, idx_g_in, idx_g_pro] = utils.differential_entropy_1D(output_samples_NS[0, ~np.isnan(output_samples_NS[0])])
                
    return MI_IO_INT - H1g2, MI_IO_NS - H1g2