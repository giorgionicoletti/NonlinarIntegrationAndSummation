import numpy as np
import numba as nb
import math

import sys
sys.path.append("../..")

import utils


@nb.njit
def sample_nonlinearsum(NSamples,
                        Sigma_input, Sigma_processing, Sigma_output,
                        A_input, A_processing, A_output,
                        A_PI, A_OP, g_PI, g_OP):
    MInput = Sigma_input.shape[0]
    MProcessing = Sigma_processing.shape[0]
    MOutput = Sigma_output.shape[0]

    L_input = np.linalg.cholesky(Sigma_input)
    L_output = np.linalg.cholesky(Sigma_output)
    L_processing = np.linalg.cholesky(Sigma_processing)

    A_output_inv = np.linalg.inv(A_output)

    input_samples = np.dot(L_input, np.random.randn(MInput, NSamples))
    v_input = np.dot(A_PI, np.tanh(input_samples))/MInput
    mean_processing = g_PI * np.dot(np.linalg.inv(A_processing), v_input)

    processing_samples = mean_processing + np.dot(L_processing, np.random.randn(MProcessing, NSamples))
    v_output = np.dot(A_OP, np.tanh(processing_samples))/MProcessing
    mean_output = g_OP * np.dot(A_output_inv, v_output)

    output_samples = mean_output + np.dot(L_output, np.random.randn(MOutput, NSamples))
            
    return input_samples, processing_samples, output_samples


@nb.njit
def sample_integrated(NSamples,
                      Sigma_input, Sigma_processing, Sigma_output,
                      A_input, A_processing, A_output,
                      A_PI, A_OP, g_PI, g_OP):
    MInput = Sigma_input.shape[0]
    MProcessing = Sigma_processing.shape[0]
    MOutput = Sigma_output.shape[0]

    L_input = np.linalg.cholesky(Sigma_input)
    L_output = np.linalg.cholesky(Sigma_output)
    L_processing = np.linalg.cholesky(Sigma_processing)

    A_output_inv = np.linalg.inv(A_output)

    input_samples = np.dot(L_input, np.random.randn(MInput, NSamples))
    v_input = np.tanh(np.dot(A_PI, input_samples)/MInput)
    mean_processing = g_PI * np.dot(np.linalg.inv(A_processing), v_input)

    processing_samples = mean_processing + np.dot(L_processing, np.random.randn(MProcessing, NSamples))
    v_output = np.tanh(np.dot(A_OP, processing_samples)/MProcessing)
    mean_output = g_OP * np.dot(A_output_inv, v_output)

    output_samples = mean_output + np.dot(L_output, np.random.randn(MOutput, NSamples))

    return input_samples, processing_samples, output_samples

@nb.njit(fastmath = True)
def sample_nonlinearsum_with_conditional(NSamples_input, NSamples_per_input,
                                         Sigma_input, Sigma_processing, Sigma_output,
                                         A_input, A_processing, A_output,
                                         A_PI, A_OP, g_PI, g_OP):
    MInput = Sigma_input.shape[0]
    MProcessing = Sigma_processing.shape[0]
    MOutput = Sigma_output.shape[0]

    L_input = np.linalg.cholesky(Sigma_input)
    L_output = np.linalg.cholesky(Sigma_output)
    L_processing = np.linalg.cholesky(Sigma_processing)

    A_output_inv = np.linalg.inv(A_output)
    A_processing_inv = np.linalg.inv(A_processing)

    input_samples = np.dot(L_input, np.random.randn(MInput, NSamples_input))
    v_input = np.dot(A_PI, np.tanh(input_samples)/MInput)
    mean_processing = g_PI * np.dot(A_processing_inv, v_input)

    processing_samples = np.zeros((MProcessing, NSamples_input, NSamples_per_input))
    output_samples = np.zeros((MOutput, NSamples_input, NSamples_per_input))

    for idx_input in range(NSamples_input):
        processing_samples[:, idx_input] = mean_processing[:, idx_input][:, None] + np.dot(L_processing, np.random.randn(MProcessing, NSamples_per_input))
        v_output = np.dot(A_OP, np.tanh(processing_samples[:, idx_input])/MProcessing)

        mean_output = g_OP * np.dot(A_output_inv, v_output)
        output_samples[:, idx_input] = mean_output + np.dot(L_output, np.random.randn(MOutput, NSamples_per_input))

    return input_samples, processing_samples, output_samples

@nb.njit(fastmath = True)
def sample_integrated_with_conditional(NSamples_input, NSamples_per_input,
                                       Sigma_input, Sigma_processing, Sigma_output,
                                       A_input, A_processing, A_output,
                                       A_PI, A_OP, g_PI, g_OP):
    MInput = Sigma_input.shape[0]
    MProcessing = Sigma_processing.shape[0]
    MOutput = Sigma_output.shape[0]

    L_input = np.linalg.cholesky(Sigma_input)
    L_output = np.linalg.cholesky(Sigma_output)
    L_processing = np.linalg.cholesky(Sigma_processing)

    A_output_inv = np.linalg.inv(A_output)
    A_processing_inv = np.linalg.inv(A_processing)

    input_samples = np.dot(L_input, np.random.randn(MInput, NSamples_input))
    v_input = np.tanh(np.dot(A_PI, input_samples)/MInput)
    mean_processing = g_PI * np.dot(A_processing_inv, v_input)

    processing_samples = np.zeros((MProcessing, NSamples_input, NSamples_per_input))
    output_samples = np.zeros((MOutput, NSamples_input, NSamples_per_input))

    for idx_input in range(NSamples_input):
        processing_samples[:, idx_input] = mean_processing[:, idx_input][:, None] + np.dot(L_processing, np.random.randn(MProcessing, NSamples_per_input))
        v_output = np.tanh(np.dot(A_OP, processing_samples[:, idx_input])/MProcessing)

        mean_output = g_OP * np.dot(A_output_inv, v_output)
        output_samples[:, idx_input] = mean_output + np.dot(L_output, np.random.randn(MOutput, NSamples_per_input))

    return input_samples, processing_samples, output_samples

@nb.njit(parallel = True, fastmath = True)
def sample_gXg_grid(g_PI_array, g_OP_array,
                    NSamples_input, NSamples_per_input,
                    Sigma_input_array, Sigma_processing_array, A_processing_array,
                    Sigma_output, A_output,
                    sigma_OP = 1, sigma_PI = 1):
    """
    Function to sample from an input-processing-output system with nonlinear integration and
    a slow processing unit. The function uses a grid of g_PI and g_OP values to sample the system,
    where g_PI is the coupling strength of the input with the processing unit and g_OP is the coupling
    strength of the processing unit with the output.

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
    
    for idx_rep in nb.prange(NRep):
        print(f"Mpro: {MProcessing}, rep: {idx_rep}/{NRep}")
        Sigma_input = Sigma_input_array[idx_rep]
        Sigma_processing = Sigma_processing_array[idx_rep]
        A_processing = A_processing_array[idx_rep]

        L_input = np.linalg.cholesky(Sigma_input)
        A_processing_inv = np.linalg.inv(A_processing)
        L_processing = np.linalg.cholesky(Sigma_processing)

        A_PI = np.random.randn(MProcessing, MInput)*sigma_PI
        A_OP = np.random.randn(MOutput, MProcessing)*sigma_OP

        input_samples = np.dot(L_input, np.random.randn(MInput, NSamples_input))

        v_input_NS = np.dot(A_PI, np.tanh(input_samples)/MInput)
        v_input_INT = np.tanh(np.dot(A_PI, input_samples)/MInput)

        for idx_g_in in range(g_PI_array.size):
            g_PI = g_PI_array[idx_g_in]

            mean_processing_NS = g_PI * np.dot(A_processing_inv, v_input_NS)
            mean_processing_INT = g_PI * np.dot(A_processing_inv, v_input_INT)

            output_samples_NS = np.zeros((MOutput, NSamples_input, NSamples_per_input, g_OP_array.size))
            output_samples_INT = np.zeros((MOutput, NSamples_input, NSamples_per_input, g_OP_array.size))

            H_OgI_NS = np.zeros((NSamples_input, g_OP_array.size))
            H_OgI_INT = np.zeros((NSamples_input, g_OP_array.size))

            for idx_input in range(NSamples_input):
                processing_samples_rand = np.dot(L_processing, np.random.randn(MProcessing, NSamples_per_input))
                output_samples_rand = np.dot(L_output, np.random.randn(MOutput, NSamples_per_input))

                processing_samples_NS = np.zeros((MProcessing, NSamples_per_input))
                processing_samples_INT = np.zeros((MProcessing, NSamples_per_input))

                # this is done in order to avoid broadcasting errors in numba
                for idx_sample in range(NSamples_per_input):
                    processing_samples_NS[:, idx_sample] = mean_processing_NS[:, idx_input] + processing_samples_rand[:, idx_sample]
                    processing_samples_INT[:, idx_sample] = mean_processing_INT[:, idx_input] + processing_samples_rand[:, idx_sample]

                v_output_NS = np.dot(A_OP, np.tanh(processing_samples_NS)/MProcessing)
                v_output_INT = np.tanh(np.dot(A_OP, processing_samples_INT)/MProcessing)

                for idx_g_pro in range(g_OP_array.size):
                    g_OP = g_OP_array[idx_g_pro]

                    mean_output_NS = g_OP * np.dot(A_output_inv, v_output_NS)
                    mean_output_INT = g_OP * np.dot(A_output_inv, v_output_INT)

                    output_samples_NS[:, idx_input, :, idx_g_pro] = mean_output_NS + output_samples_rand
                    output_samples_INT[:, idx_input, :, idx_g_pro] = mean_output_INT + output_samples_rand

                    H_OgI_NS[idx_input, idx_g_pro] = utils.differential_entropy_1D(output_samples_NS[0, idx_input, :, idx_g_pro])
                    H_OgI_INT[idx_input, idx_g_pro] = utils.differential_entropy_1D(output_samples_INT[0, idx_input, :, idx_g_pro])
            
            for idx_g_pro in range(g_OP_array.size):
                MI_IO_INT[idx_rep, idx_g_in, idx_g_pro] = utils.differential_entropy_1D(output_samples_INT[0, :, :, idx_g_pro].flatten()) - np.mean(H_OgI_INT[:, idx_g_pro])
                MI_IO_NS[idx_rep, idx_g_in, idx_g_pro] = utils.differential_entropy_1D(output_samples_NS[0, :, :, idx_g_pro].flatten()) - np.mean(H_OgI_NS[:, idx_g_pro])

    return MI_IO_INT, MI_IO_NS