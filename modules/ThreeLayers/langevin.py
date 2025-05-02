import numpy as np
import numba as nb


@nb.njit(fastmath=True)
def simulate_nonlinearsum(NSteps, dt,
                          tau_input, tau_processing, tau_output,
                          A_input, A_processing, A_output,
                          A_PI, A_OP, g_PI, g_OP):
    """
    Function to simulate the Langevin dynamics of an input-processing-output system with
    nonlinear summation.

    Parameters
    ----------
    NSteps : int
        Number of time steps to simulate.
    dt : float
        Time step size.
    tau_input : float
        Timescale of the input unit.
    tau_processing : float
        Timescale of the processing unit.
    tau_output : float
        Timescale of the output unit.
    A_input : np.ndarray
        Interaction matrix of the input unit.
    A_processing : np.ndarray
        Interaction matrix of the processing unit.
    A_output : np.ndarray
        Interaction matrix of the output unit.
    A_PI : np.ndarray
        Interaction matrix of from the input to the processing unit.
    A_OP : np.ndarray
        Interaction matrix of from the processing to the output unit.
    g_PI : float
        Coupling strength from the input to the processing unit.
    g_OP : float
        Coupling strength from the processing to the output unit.

    Returns
    -------
    x_input : np.ndarray
        Simulated input unit activity over time.
    x_processing : np.ndarray
        Simulated processing unit activity over time.
    x_output : np.ndarray
        Simulated output unit activity over time.
    """

    MInput = A_input.shape[0]
    MProcessing = A_processing.shape[0]
    MOutput = A_output.shape[0]

    x_input = np.zeros((NSteps, MInput))
    x_processing = np.zeros((NSteps, MProcessing))
    x_output = np.zeros((NSteps, MOutput))

    x_input[0] = np.random.randn(MInput)
    x_processing[0] = np.random.randn(MProcessing)
    x_output[0] = np.random.randn(MOutput)

    sqdt_in = np.sqrt(2*dt/tau_input)
    sqdt_pr = np.sqrt(2*dt/tau_processing)
    sqdt_out = np.sqrt(2*dt/tau_output)
    dt_in = dt/tau_input
    dt_pr = dt/tau_processing
    dt_out = dt/tau_output

    for t in range(NSteps-1):
        x_input[t+1] = x_input[t] + dt_in * (- A_input @ x_input[t]) + sqdt_in*np.random.randn(MInput)
        x_processing[t+1] = x_processing[t] + dt_pr * (- A_processing @ x_processing[t] + g_PI/MInput*A_PI @ np.tanh(x_input[t]))
        x_processing[t+1] += sqdt_pr*np.random.randn(MProcessing)
        x_output[t+1] = x_output[t] + dt_out * (- A_output @ x_output[t] + g_OP/MProcessing*A_OP @ np.tanh(x_processing[t]))
        x_output[t+1] += sqdt_out*np.random.randn(MOutput)

    return x_input, x_processing, x_output

@nb.njit(fastmath=True)
def simulate_integration(NSteps, dt,
                         tau_input, tau_processing, tau_output,
                         A_input, A_processing, A_output,
                         A_PI, A_OP, g_PI, g_OP):
    """
    Function to simulate the Langevin dynamics of an input-processing-output system with
    nonlinear integration.

    Parameters
    ----------
    NSteps : int
        Number of time steps to simulate.
    dt : float
        Time step size.
    tau_input : float
        Timescale of the input unit.
    tau_processing : float
        Timescale of the processing unit.
    tau_output : float
        Timescale of the output unit.
    A_input : np.ndarray
        Interaction matrix of the input unit.
    A_processing : np.ndarray
        Interaction matrix of the processing unit.
    A_output : np.ndarray
        Interaction matrix of the output unit.
    A_PI : np.ndarray
        Interaction matrix of from the input to the processing unit.
    A_OP : np.ndarray
        Interaction matrix of from the processing to the output unit.
    g_PI : float
        Coupling strength from the input to the processing unit.
    g_OP : float
        Coupling strength from the processing to the output unit.

    Returns
    -------
    x_input : np.ndarray
        Simulated input unit activity over time.
    x_processing : np.ndarray
        Simulated processing unit activity over time.
    x_output : np.ndarray
        Simulated output unit activity over time.
    """

    MInput = A_input.shape[0]
    MProcessing = A_processing.shape[0]
    MOutput = A_output.shape[0]

    x_input = np.zeros((NSteps, MInput))
    x_processing = np.zeros((NSteps, MProcessing))
    x_output = np.zeros((NSteps, MOutput))

    x_input[0] = np.random.randn(MInput)
    x_processing[0] = np.random.randn(MProcessing)
    x_output[0] = np.random.randn(MOutput)

    sqdt_in = np.sqrt(2*dt/tau_input)
    sqdt_pr = np.sqrt(2*dt/tau_processing)
    sqdt_out = np.sqrt(2*dt/tau_output)
    dt_in = dt/tau_input
    dt_pr = dt/tau_processing
    dt_out = dt/tau_output


    for t in range(NSteps-1):
        x_input[t+1] = x_input[t] + dt_in * (- A_input @ x_input[t]) + sqdt_in*np.random.randn(MInput)
        x_processing[t+1] = x_processing[t] + dt_pr * (- A_processing @ x_processing[t] + g_PI*np.tanh(A_PI @ x_input[t]/MInput))
        x_processing[t+1] += sqdt_pr*np.random.randn(MProcessing)
        x_output[t+1] = x_output[t] + dt_out * (- A_output @ x_output[t] + g_OP*np.tanh(A_OP @ x_processing[t]/MProcessing))
        x_output[t+1] += sqdt_out*np.random.randn(MOutput)

    return x_input, x_processing, x_output

@nb.njit(fastmath=True)
def simulate_NS_INT(NSteps, dt,
                    tau_input, tau_processing, tau_output,
                    A_input, A_processing, A_output,
                    A_PI, A_OP, g_PI, g_OP,
                    NBurn = 100000):
    """
    Function to simulate the Langevin dynamics of an input-processing-output system with
    nonlinear summation and integration.

    Parameters
    ----------
    NSteps : int
        Number of time steps to simulate.
    dt : float
        Time step size.
    tau_input : float
        Timescale of the input unit.
    tau_processing : float
        Timescale of the processing unit.
    tau_output : float
        Timescale of the output unit.
    A_input : np.ndarray
        Interaction matrix of the input unit.
    A_processing : np.ndarray
        Interaction matrix of the processing unit.
    A_output : np.ndarray
        Interaction matrix of the output unit.
    A_PI : np.ndarray
        Interaction matrix of from the input to the processing unit.
    A_OP : np.ndarray
        Interaction matrix of from the processing to the output unit.
    g_PI : float
        Coupling strength from the input to the processing unit.
    g_OP : float
        Coupling strength from the processing to the output unit.
    NBurn : int
        Number of burn-in steps to discard.
        Default is 100000.
    
    Returns
    -------
    x_input : np.ndarray
        Simulated input unit activity over time.
    x_processing : np.ndarray
        Simulated processing unit activity over time.
    x_output : np.ndarray
        Simulated output unit activity over time.
    """

    MInput = A_input.shape[0]
    MProcessing = A_processing.shape[0]
    MOutput = A_output.shape[0]

    x_input = np.random.randn(MInput)
    x_processing_NS = np.random.randn(MProcessing)
    x_processing_INT = np.random.randn(MProcessing)
    x_output_NS = np.zeros((NSteps, MOutput))
    x_output_INT = np.zeros((NSteps, MOutput))

    x_output_NS_0 = np.random.randn(MOutput)
    x_output_INT_0 = np.random.randn(MOutput)

    sqdt_in = np.sqrt(2*dt/tau_input)
    sqdt_pr = np.sqrt(2*dt/tau_processing)
    sqdt_out = np.sqrt(2*dt/tau_output)
    dt_in = dt/tau_input
    dt_pr = dt/tau_processing
    dt_out = dt/tau_output

    for t in range(NBurn):
        x_output_NS_0 = x_output_NS_0 + dt_out * (- A_output @ x_output_NS_0 + g_OP/MProcessing*A_OP @ np.tanh(x_processing_NS))
        x_output_NS_0 += sqdt_out*np.random.randn(MOutput)

        x_output_INT_0 = x_output_INT_0 + dt_out * (- A_output @ x_output_INT_0 + g_OP*np.tanh(A_OP @ x_processing_INT/MProcessing))
        x_output_INT_0 += sqdt_out*np.random.randn(MOutput)

        x_processing_NS += dt_pr * (- A_processing @ x_processing_NS + g_PI/MInput*A_PI @ np.tanh(x_input))
        x_processing_NS += sqdt_pr*np.random.randn(MProcessing)

        x_processing_INT += + dt_pr * (- A_processing @ x_processing_INT + g_PI*np.tanh(A_PI @ x_input/MInput))
        x_processing_INT += sqdt_pr*np.random.randn(MProcessing)

        x_input += dt_in * (- A_input @ x_input) + sqdt_in*np.random.randn(MInput)

    x_output_NS[0] = x_output_NS_0
    x_output_INT[0] = x_output_INT_0

    for t in range(NSteps-1):
        x_output_NS[t+1] = x_output_NS[t] + dt_out * (- A_output @ x_output_NS[t] + g_OP/MProcessing*A_OP @ np.tanh(x_processing_NS))
        x_output_NS[t+1] += sqdt_out*np.random.randn(MOutput)

        x_output_INT[t+1] = x_output_INT[t] + dt_out * (- A_output @ x_output_INT[t] + g_OP*np.tanh(A_OP @ x_processing_INT/MProcessing))
        x_output_INT[t+1] += sqdt_out*np.random.randn(MOutput)

        x_processing_NS += dt_pr * (- A_processing @ x_processing_NS + g_PI/MInput*A_PI @ np.tanh(x_input))
        x_processing_NS += sqdt_pr*np.random.randn(MProcessing)

        x_processing_INT += + dt_pr * (- A_processing @ x_processing_INT + g_PI*np.tanh(A_PI @ x_input/MInput))
        x_processing_INT += sqdt_pr*np.random.randn(MProcessing)

        x_input += dt_in * (- A_input @ x_input) + sqdt_in*np.random.randn(MInput)

    return x_output_NS, x_output_INT

@nb.njit(parallel=True, fastmath=True)
def trajectory_statistics_NS_INT(NRep, NSteps, dt,
                                tau_input, tau_processing, tau_output,
                                A_input, A_processing, A_output,
                                A_PI, A_OP, g_PI, g_OP,
                                NBurn = 100000):
    """
    Function to simulate the Langevin dynamics of an input-processing-output system with
    nonlinear summation and integration, and collect statistics over multiple trajectories.

    Parameters
    ----------
    NRep : int
        Number of repetitions to simulate.
    NSteps : int
        Number of time steps to simulate.
    dt : float
        Time step size.
    tau_input : float
        Timescale of the input unit.
    tau_processing : float
        Timescale of the processing unit.
    tau_output : float
        Timescale of the output unit.
    A_input : np.ndarray
        Interaction matrix of the input unit.
    A_processing : np.ndarray
        Interaction matrix of the processing unit.
    A_output : np.ndarray
        Interaction matrix of the output unit.
    A_PI : np.ndarray
        Interaction matrix of from the input to the processing unit.
    A_OP : np.ndarray
        Interaction matrix of from the processing to the output unit.
    g_PI : float
        Coupling strength from the input to the processing unit.
    g_OP : float
        Coupling strength from the processing to the output unit.
    NBurn : int
        Number of burn-in steps to discard.
        Default is 100000.
    
    Returns
    -------
    x_output_NS : np.ndarray
        Simulated output unit activity over time for nonlinear summation.
    x_output_INT : np.ndarray
        Simulated output unit activity over time for nonlinear integration.
    """

    MOutput = A_output.shape[0]

    x_output_NS = np.zeros((NRep, NSteps, MOutput))
    x_output_INT = np.zeros((NRep, NSteps, MOutput))

    for rep in nb.prange(NRep):
        res = simulate_NS_INT(NSteps, dt,
                              tau_input, tau_processing, tau_output,
                              A_input, A_processing, A_output,
                              A_PI, A_OP, g_PI, g_OP,
                              NBurn = NBurn)
        x_output_NS[rep] = res[0]
        x_output_INT[rep] = res[1]

    return x_output_NS, x_output_INT