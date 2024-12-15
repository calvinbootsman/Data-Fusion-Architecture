import numpy as np
import random
import pandas as pd
from numba import njit
import cupy as cp

def log_likelihood(Z_n, omega_n, R, C, sigma):
    """
    Calculates the likelihood of observing the measured impedance data given 
    a specific circuit model and its parameters.

    Args:
        Z_i (complex): Measured impedance at frequency omega_i.
        omega_i (float): Angular frequency.
        R (list): List of resistance values in the circuit model.
        C (list): List of capacitance values in the circuit model.
        sigma (float): Standard deviation of the measurement noise.

    Returns:
        float: Likelihood value.
    """
    
    # Calculate impedance from the circuit model
    z_omega_i = circuit_impedance(omega_n, R, C)
    log_likelihood_value = -(1 / (2 * sigma**2)) * abs(Z_n - z_omega_i)**2 - 0.5 * cp.log(2 * cp.pi * sigma**2)

    return log_likelihood_value  # Return the log-likelihood

def circuit_impedance(omega_n, R, C):
    """
    Calculates the impedance of the circuit model at given frequencies.

    Args:
        omega_n (list or numpy array): Angular frequencies.
        R (list): List of resistance values in the circuit model.
        C (list): List of capacitance values in the circuit model.

    Returns:
        numpy array: Impedances of the circuit model for each frequency.
    """
    
    # omega_n = cp.array(omega_n)  # Convert to NumPy array for vectorization
    # R = cp.array(R)
    # C = cp.array(C)
    omega_n = cp.array(omega_n) 
    z_omega_n = cp.zeros_like(omega_n, dtype=complex)  # Initialize an array to store impedances

    for i in range(len(R)):
        z_omega_n += R[i] / (1 + 1j * omega_n * R[i] * C[i])  # Vectorized calculation

    return z_omega_n

def log_prior(R, C, R_min=100, R_max=10e3, C_min=10e-9, C_max=10e-6):
    """
    Calculates the prior probability of the circuit parameters.

    Args:
        R (list): List of resistance values.
        C (list): List of capacitance values.
        R_min (float, optional): Minimum resistance value. Defaults to 100.
        R_max (float, optional): Maximum resistance value. Defaults to 10e3.
        C_min (float, optional): Minimum capacitance value. Defaults to 10e-9.
        C_max (float, optional): Maximum capacitance value. Defaults to 10e-6.

    Returns:
        float: Prior probability value.
    """
    
    for r in R:
        if not (R_min <= r <= R_max):
            return -cp.inf  # Prior is 0 if resistance is outside the allowed range
    for c in C:
        if not (C_min <= c <= C_max):
            return -cp.inf  # Prior is 0 if capacitance is outside the allowed range
    
    # If all parameters are within the allowed range, use uniform prior
    return 0  # Log of a uniform distribution is 0  

def log_posterior(omega_n, Z_n, K, R, C, sigma):
    """
    Calculates the posterior probability of the circuit parameters given the 
    observed data and the prior.

    Args:
        omega_n (list): List of angular frequencies.
        Z_n (list): List of measured impedances.
        K (int): Number of RC parallel circuits in the model.
        R (list): List of resistance values.
        C (list): List of capacitance values.
        sigma (float): Standard deviation of the measurement noise.

    Returns:
        float: Posterior probability value.
    """   
    log_prior_prob = log_prior(R, C)
    if log_prior_prob == -cp.inf:
        return -cp.inf  # If log-prior is -inf, log-posterior is also -inf

    log_likelihood_prob = 0
    
    log_likelihood_prob += cp.sum(log_likelihood(Z_n, omega_n, R, C, sigma))

    log_posterior_prob = log_likelihood_prob + log_prior_prob
    return log_posterior_prob  # Return the un-logged posterior probability


def metropolis(omega_n, Z_n, K, sigma, n_iterations=10000, burn_in=1000, 
              R_min=100, R_max=10e3, C_min=10e-9, C_max=10e-6):
    """
    Performs Metropolis sampling to estimate the circuit parameters.

    Args:
        omega_n (list): List of angular frequencies.
        Z_n (list): List of measured impedances.
        K (int): Number of RC parallel circuits in the model.
        sigma (float): Standard deviation of the measurement noise.
        n_iterations (int, optional): Number of MCMC iterations. Defaults to 10000.
        burn_in (int, optional): Number of initial samples to discard. Defaults to 1000.
        R_min (float, optional): Minimum resistance value. Defaults to 100.
        R_max (float, optional): Maximum resistance value. Defaults to 10e3.
        C_min (float, optional): Minimum capacitance value. Defaults to 10e-9.
        C_max (float, optional): Maximum capacitance value. Defaults to 10e-6.

    Returns:
        tuple: A tuple containing the accepted R and C samples.
    """

    # Initialize parameters
    R_current = [cp.random.uniform(R_min, R_max) for _ in range(K)]
    C_current = [cp.random.uniform(C_min, C_max) for _ in range(K)]
    accepted_proposals = 0
    # Store accepted samples
    R_samples = []
    C_samples = []
    
    # omega_n = cp.array(omega_n)
    for _ in range(n_iterations):
        # Propose new parameters
        R_proposal = [r + cp.random.uniform(-10, 10) for r in R_current]
        C_proposal = [c + cp.random.uniform(-10e-9, 10e-9) for c in C_current]  # REPORT: change it normal distribution and compare.

        # R_proposal = [r + cp.random.normal(0, 10) for r in R_current]
        # C_proposal = [c + cp.random.normal(0, 1e-7) for c in C_current]  # REPORT: change it normal distribution and compare.

        # Calculate posterior probabilities
        log_posterior_current = log_posterior(omega_n, Z_n, K, R_current, C_current, sigma)
        log_posterior_proposal = log_posterior(omega_n, Z_n, K, R_proposal, C_proposal, sigma)

        # Calculate log of acceptance ratio
        log_acceptance_ratio = log_posterior_proposal - log_posterior_current
        acceptance_ratio = min(1, log_acceptance_ratio)  # 0 if log_acceptance_ratio is positive, else log_acceptance_ratio
        
        # Accept or reject proposal (using exp(acceptance_ratio) since acceptance_ratio is in log space)
        if cp.log(random.uniform(0, 1)) < acceptance_ratio:
            R_current = R_proposal
            C_current = C_proposal
            accepted_proposals += 1
        # Store accepted samples after burn-in
        if _ > burn_in:
            R_samples.append(R_current)
            C_samples.append(C_current)

    return R_current, C_current, accepted_proposals, R_samples, C_samples

def wbic(omega_n, Z_n, K, R_samples, C_samples, sigma):
    """
    Calculates the Widely Applicable Bayesian Information Criterion (WBIC) 
    for a given circuit model and its parameters.

    Args:
        omega_n (list): List of angular frequencies.
        Z_n (list): List of measured impedances.
        K (int): Number of RC parallel circuits in the model.
        R_samples (list): List of accepted resistance samples from MCMC.
        C_samples (list): List of accepted capacitance samples from MCMC.
        sigma (float): Standard deviation of the measurement noise.

    Returns:
        float: WBIC value.
    """
    
    n = len(omega_n)
    beta = 1 / cp.log(n)
    log_likelihood_sum = 0

    R_samples = cp.array(R_samples)
    C_samples = cp.array(C_samples)

    for i in range(n):
        Z_i = Z_n[i]  # Get the current Z_i
        omega_i = omega_n[i]  # Get the current omega_i

        # Vectorized calculation of impedance for all samples at once
        z_omega_i = cp.sum(R_samples / (1 + 1j * omega_i * R_samples * C_samples), axis=1)  

        # Vectorized calculation of log-likelihood for all samples at once
        log_likelihood_sum += cp.sum(-(1 / (2 * sigma**2)) * cp.abs(Z_i - z_omega_i)**2 - cp.log(cp.sqrt(2 * cp.pi * sigma**2)))

    wbic_value = - beta * log_likelihood_sum 
    return wbic_value

import matplotlib.pyplot as plt

def find_best_fit_model(frequencies, Z_measured, sigma = 10, K_max=2, n_itterations=10000, burn_in=1000):
    wbic_values = []
    R_samples_n = []
    C_samples_n = []

    for k in range(1, K_max + 1):  # Check models with K=1, 2
        R_n, C_n, accepted_proposals, R_samples, C_samples = metropolis(frequencies, Z_measured, k, sigma, n_iterations=n_itterations, burn_in=burn_in)
        wbic_value = wbic(frequencies, Z_measured, k, R_samples, C_samples, sigma)
        wbic_values.append(wbic_value)
        R_samples_n.append(R_samples)
        C_samples_n.append(C_samples)
        
    # Select the best model
    best_model = cp.argmin(wbic_values) + 1  # +1 because K starts from 1
    return best_model, wbic_values, R_samples_n, C_samples_n

def single_mcmc(data_filename, K, sigma=10, itterations=10000, burn_in=1000, print_results=False):
    data = pd.read_csv(data_filename)
    frequency_list = data['Frequency']  # Hz
    Z_complex = cp.array(data['Z_real']) + 1j * cp.array( data['Z_imag'])

    R_samples = []
    C_samples = []
    R_n, C_n, accepted_proposals, R_samples, C_samples = metropolis(frequency_list, Z_complex, K, sigma, n_iterations=itterations, burn_in=burn_in)
    acceptance_rate = accepted_proposals / itterations * 100
    if print_results:
        print(f"Acceptance rate: {acceptance_rate:.2f}%")
        if len(R_n) == 1:
            print(f"Estimated R value: {cp.mean(R_n[0]):.2f}")
            print(f"Estimated C value: {cp.mean(C_n[0]):.2e}")
        else:
            print(f"Estimated R value: {cp.mean(R_n[0]):.2f}\t{cp.mean(R_n[1]):.2f}")
            print(f"Estimated C value: {cp.mean(C_n[0]):.2e}\t{cp.mean(C_n[1]):.2e}")
        print('')

    results = {
        'R_n': R_n,
        'C_n': C_n,
        'R_samples': R_samples,
        'C_samples': C_samples,
        'acceptance_rate': acceptance_rate,
        'frequency_list': frequency_list,
        'Z_complex': Z_complex
    }
    return results

def impedance_plot(frequencies, Z_measured, R, C, ax=None): 
    if ax is None:
        ax = plt.gca()  # Get the current axes if none is provided

    # Z_model = [circuit_impedance(omega, R, C) for omega in frequencies]
    Z_model = circuit_impedance(frequencies, R, C)
    Z_model = cp.asnumpy(Z_model)
    Z_measured = cp.asnumpy(Z_measured)
    frequencies = cp.asnumpy(frequencies)

    ax.plot(frequencies, np.real(Z_model), label='Calculated Z_real')
    ax.plot(frequencies, np.imag(Z_model), label='Calculated Z_imag')
    ax.plot(frequencies, np.real(Z_measured), 'o', label='Measured Z_real')
    ax.plot(frequencies, np.imag(Z_measured), 'o', label='Measured Z_imag')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Impedance')
    ax.set_title('Impedance vs Frequency')
    ax.legend()
    ax.grid()

def R_C_plot(R_samples, C_samples, ax_r, ax_c):
    if ax_r is None:
        return
    if ax_c is None:
        return
    R_samples_np = np.array([cp.asnumpy(r[0]) for r in R_samples]) 
    C_samples_np = np.array([cp.asnumpy(c[0]) for c in C_samples])

    ax_r.plot(R_samples_np, label='Resistance (Ohms)')
    ax_r.set_title('R_n values')
    ax_r.set_xlabel('Iteration')
    ax_r.set_ylabel('Value')
    ax_r.legend()

    ax_c.plot(C_samples_np, label='Capacitance (Farads)')    
    ax_c.set_title('C_n values')
    ax_c.set_xlabel('Iteration')
    ax_c.set_ylabel('Value')
    ax_c.legend()

def sort_samples(R_samples, C_samples):
    """
    Sorts the sublists in R_samples and rearranges the corresponding sublists in C_samples accordingly.
    
    Parameters:
        R_samples (list of lists): The main list to be sorted.
        C_samples (list of lists): The secondary list whose elements follow the sorting of R_samples.

    Returns:
        tuple: Sorted R_samples and adjusted C_samples.
    """
    sorted_R_samples = []
    sorted_C_samples = []

    for foo, bar in zip(R_samples, C_samples):
        # Combine the lists as pairs, sort by the first element, and ucpack
        combined = sorted(zip(foo, bar), key=lambda x: x[0])
        # Separate them back into two lists
        sorted_foo, sorted_bar = zip(*combined)
        sorted_R_samples.append(list(sorted_foo))
        sorted_C_samples.append(list(sorted_bar))

    return sorted_R_samples, sorted_C_samples

from tqdm import tqdm
import matplotlib.pyplot as plt
from multiprocessing import Pool

# ... (your find_best_fit_model and sort_samples functions) ...

custom_data_files = [
    "circuit1_data.csv",
    "circuit2_data.csv",
    "generated_data_1.csv",
    "generated_data_2.csv",
    "generated_data_3.csv",
]
pos = -1
def process_file(file):
    global pos
    custom_data = pd.read_csv(file)
    custom_frequencies = cp.array(custom_data["Frequency"])  # Hz
    custom_Z_measured = cp.array(custom_data["Z_real"]) + 1j * cp.array(
        custom_data["Z_imag"]
    )

    wbic_values_n = []
    R_samples_n = []
    C_samples_n = []

    K_max = 3

    if file == "circuit1_data.csv" or file == "circuit2_data.csv":
        K_max = 2
    K_n = cp.zeros(K_max)

    # Use tqdm with position argument
    for i in tqdm(cp.arange(500), desc=f"Finding best fit model for {file}", position=pos): 
        best_model, wbic_values, R_samples, C_samples = find_best_fit_model(
            custom_frequencies,
            custom_Z_measured,
            sigma=10,
            n_itterations=50_000,
            K_max=K_max,
            burn_in=1000,
        )
        K_n[best_model - 1] += 1
        R_samples_n.append(R_samples)
        C_samples_n.append(C_samples)

    K_best = cp.argmax(K_n) + 1

    R_samples_n, C_samples_n = sort_samples(R_samples_n, C_samples_n)
    
    # ... (rest of your plotting and printing code) ...

    return K_best, R_samples_n, C_samples_n, K_n

if __name__ == '__main__': 
    with Pool() as pool:
        pos += 1
        results = pool.map(process_file, custom_data_files)

    # Access results for each file
    for i, file in enumerate(custom_data_files):
        K_best, R_samples_n, C_samples_n, K_n = results[i]
        print(f"Best fit model for {file}: K = {K_best}\n")