import numpy as np
import numba as nb


@nb.njit
def simulate_nonlinearsum(NSteps, dt,
                          tau_input, tau_output,
                          A_input, A_output,
                          A_OI, g_OI):
    """
    Function to simulate the Langevin dynamics of an input-output system with
    nonlinear summation.

    Parameters
    ----------
    NSteps : int
        Number of time steps to simulate.
    dt : float
        Time step size.
    tau_input : float
        Timescale of the input unit.
    tau_output : float
        Timescale of the output unit.
    A_input : np.ndarray
        Interaction matrix of the input unit.
    A_output : np.ndarray
        Interaction matrix of the output unit.
    A_OI : np.ndarray
        Interaction matrix of from the input to the output unit.
    g_OI : float
        Coupling strength from the input to the output unit.

    Returns
    -------
    x_input : np.ndarray
        Simulated input unit activity over time.
    x_output : np.ndarray
        Simulated output unit activity over time.
    """

    MInput = A_input.shape[0]
    MOutput = A_output.shape[0]

    x_input = np.zeros((NSteps, MInput))
    x_output = np.zeros((NSteps, MOutput))

    x_input[0] = np.random.randn(MInput)
    x_output[0] = np.random.randn(MOutput)

    sqdt_in = np.sqrt(2*dt/tau_input)
    sqdt_out = np.sqrt(2*dt/tau_output)
    dt_in = dt/tau_input
    dt_out = dt/tau_output

    for t in range(NSteps-1):
        x_input[t+1] = x_input[t] + dt_in * (- A_input @ x_input[t]) + sqdt_in*np.random.randn(MInput)
        x_output[t+1] = x_output[t] + dt_out * (- A_output @ x_output[t] + g_OI/MInput*A_OI @ np.tanh(x_input[t])) + sqdt_out*np.random.randn(MOutput)

    return x_input, x_output

@nb.njit
def simulate_integrated(NSteps, dt, 
                        tau_input, tau_output,
                        A_input, A_output,
                        A_OI, g_OI):
    """
    Function to simulate the Langevin dynamics of an input-output system with
    nonlinear integration.

    Parameters
    ----------
    NSteps : int
        Number of time steps to simulate.
    dt : float
        Time step size.
    tau_input : float
        Timescale of the input unit.
    tau_output : float
        Timescale of the output unit.
    A_input : np.ndarray
        Interaction matrix of the input unit.
    A_output : np.ndarray
        Interaction matrix of the output unit.
    A_OI : np.ndarray
        Interaction matrix of from the input to the output unit.
    g_OI : float
        Coupling strength from the input to the output unit.

    Returns
    -------
    x_input : np.ndarray
        Simulated input unit activity over time.
    x_output : np.ndarray
        Simulated output unit activity over time.
    """

    MInput = A_input.shape[0]
    MOutput = A_output.shape[0]

    x_input = np.zeros((NSteps, MInput))
    x_output = np.zeros((NSteps, MOutput))

    x_input[0] = np.random.randn(MInput)
    x_output[0] = np.random.randn(MOutput)

    sqdt_in = np.sqrt(2*dt/tau_input)
    sqdt_out = np.sqrt(2*dt/tau_output)
    dt_in = dt/tau_input
    dt_out = dt/tau_output

    for t in range(NSteps-1):
        x_input[t+1]   = x_input[t] + dt_in * (- A_input @ x_input[t]) + sqdt_in*np.random.randn(MInput)
        x_output[t+1]  = x_output[t] + dt_out * (- A_output @ x_output[t] + g_OI*np.tanh(A_OI @ x_input[t]/MInput)) 
        x_output[t+1] += sqdt_out*np.random.randn(MOutput)

    return x_input, x_output