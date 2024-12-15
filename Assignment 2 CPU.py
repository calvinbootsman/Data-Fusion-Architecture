import numpy as np
import random
import pandas as pd
from numba import njit
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool, Manager
import cupy as cp
import pickle 
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
    z_omega_i = circuit_impedance(omega_n, R, C)
    log_likelihood_value = -(1 / (2 * sigma**2)) * abs(Z_n - z_omega_i)**2 - 0.5 * np.log(2 * np.pi * sigma**2)
    return log_likelihood_value

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
    z_omega_n = np.zeros_like(omega_n, dtype=complex)  # Initialize an array to store impedances

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
            return -np.inf
    for c in C:
        if not (C_min <= c <= C_max):
            return -np.inf
    return 0

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
    if log_prior_prob == -np.inf:
        return -np.inf

    log_likelihood_prob = np.sum(log_likelihood(Z_n, omega_n, R, C, sigma))
    return log_likelihood_prob + log_prior_prob

from multiprocessing import Pool

def single_chain_update(args):
    """
    Updates a single chain at a specific temperature.

    Args:
        omega_n (list): Angular frequencies.
        Z_n (list): Measured impedances.
        K (int): Number of RC circuits.
        R (list): Current resistances in the chain.
        C (list): Current capacitances in the chain.
        sigma (float): Noise standard deviation.
        temp (float): Current temperature of the chain.

    Returns:
        dict: Updated chain with R, C, and log_posterior values.
    """
    omega_n, Z_n, K, R, C, sigma, temp = args

    proposed_R = [r + np.random.normal(0, 10) for r in R]
    proposed_C = [c + np.random.normal(0, 5.77e-9) for c in C]

    current_log_posterior = log_posterior(omega_n, Z_n, K, R, C, sigma)
    proposed_log_posterior = log_posterior(omega_n, Z_n, K, proposed_R, proposed_C, sigma)

    log_acceptance_ratio = (proposed_log_posterior - current_log_posterior) / temp

    if np.log(random.uniform(0, 1)) < log_acceptance_ratio:
        return {"R": proposed_R, "C": proposed_C, "log_posterior": proposed_log_posterior}
    else:
        return {"R": R, "C": C, "log_posterior": current_log_posterior}

def parallel_tempering_gpu(omega_n, Z_n, K, sigma, n_iterations=10000, burn_in=1000, 
                           n_temps=5, temp_min=1, R_min=100, R_max=10e3, 
                           C_min=10e-9, C_max=10e-6):
    """
    GPU-accelerated Parallel Tempering MCMC.

    Args:
        Same as before, but arrays are processed on the GPU.

    Returns:
        tuple: Accepted samples for R and C from the coldest chain.
    """
    # Z_n = np.array(Z_n)
    # omega_n = np.array(omega_n)

    temperatures = np.logspace(np.log10(temp_min), np.log10(10 * temp_min), n_temps)
    chains = [{
        "R": np.random.uniform(R_min, R_max, K),
        "C": np.random.uniform(C_min, C_max, K),
        "log_posterior": None
    } for _ in range(n_temps)]

    

    for chain in chains:
        chain["log_posterior"] = np.sum(log_likelihood(Z_n, omega_n, chain["R"], chain["C"], sigma))

    R_samples, C_samples = [], []

    for _ in range(n_iterations):
        for t in range(n_temps):
            current_chain = chains[t]
            proposed_R = current_chain["R"] + np.random.normal(0, 10, K)
            proposed_C = current_chain["C"] + np.random.normal(0, 5.77e-9, K)

            log_posterior_proposal = np.sum(log_likelihood(Z_n, omega_n, proposed_R, proposed_C, sigma))
            log_acceptance_ratio = (log_posterior_proposal - current_chain["log_posterior"]) / temperatures[t]

            if np.log(np.random.uniform(0, 1)) < log_acceptance_ratio:
                current_chain["R"] = proposed_R
                current_chain["C"] = proposed_C
                current_chain["log_posterior"] = log_posterior_proposal

        if _ >= burn_in:
            # R_samples.append(cp.asnumpy(chains[0]["R"]))
            # C_samples.append(cp.asnumpy(chains[0]["C"]))
            R_samples.append(chains[0]["R"])
            C_samples.append(chains[0]["C"])

    return chains[0]["R"], chains[0]["C"], R_samples, C_samples

def parallel_tempering_multithreaded(omega_n, Z_n, K, sigma, n_iterations=10000, burn_in=1000, 
                       n_temps=6, temp_min=1, R_min=100, R_max=10e3, 
                       C_min=10e-9, C_max=10e-6):
    """
    Performs Parallel Tempering MCMC with multiprocessing to parallelize updates across chains.

    Args:
        omega_n (list): Angular frequencies.
        Z_n (list): Measured impedances.
        K (int): Number of RC circuits.
        sigma (float): Noise standard deviation.
        n_iterations (int): Number of MCMC iterations.
        burn_in (int): Number of initial samples to discard.
        n_temps (int): Number of temperature chains.
        temp_min (float): Minimum temperature.
        R_min, R_max, C_min, C_max: Parameter bounds.

    Returns:
        tuple: Final samples for R and C, along with their histories.
    """
    temperatures = np.logspace(np.log10(temp_min), np.log10(10 * temp_min), n_temps)
    chains = [{
        "R": [np.random.uniform(R_min, R_max) for _ in range(K)],
        "C": [np.random.uniform(C_min, C_max) for _ in range(K)],
        "log_posterior": None
    } for _ in range(n_temps)]

    for chain in chains:
        chain["log_posterior"] = log_posterior(omega_n, Z_n, K, chain["R"], chain["C"], sigma)

    R_samples = []
    C_samples = []

    with Pool(processes=n_temps) as pool:
        for iteration in range(n_iterations):
            # Prepare arguments for multiprocessing
            args = [
                (omega_n, Z_n, K, chains[t]["R"], chains[t]["C"], sigma, temperatures[t]) 
                for t in range(n_temps)
            ]
            
            # Update all chains in parallel
            updated_chains = pool.map(single_chain_update, args)
            
            for t, updated_chain in enumerate(updated_chains):
                chains[t] = updated_chain

            # Perform swaps between adjacent temperature chains
            if iteration % 100 == 0 and n_temps > 1:
                for t in range(n_temps - 1):
                    chain_t, chain_t1 = chains[t], chains[t + 1]
                    delta = ((chain_t1["log_posterior"] / temperatures[t + 1]) -
                             (chain_t["log_posterior"] / temperatures[t]))
                    if np.log(random.uniform(0, 1)) < delta:
                        chains[t], chains[t + 1] = chains[t + 1], chains[t]

            # Record samples after burn-in
            if iteration >= burn_in:
                R_samples.append(chains[0]["R"])
                C_samples.append(chains[0]["C"])

    return chains[0]["R"], chains[0]["C"], R_samples, C_samples

def wbic(omega_n, Z_n, K, R_samples, C_samples, sigma):
    n = len(omega_n)
    beta = 1 / np.log(n)
    log_likelihood_sum = 0

    # Ensure inputs are CuPy arrays
    omega_n = np.array(omega_n)  # Ensure angular frequencies are on GPU
    Z_n = np.array(Z_n)          # Measured impedances
    R_samples = np.array(R_samples)
    C_samples = np.array(C_samples)

    # Loop over the data points
    for i in range(n):
        Z_i = Z_n[i]
        omega_i = omega_n[i]
        z_omega_i = np.sum(R_samples / (1 + 1j * omega_i * R_samples * C_samples), axis=1)
        log_likelihood_sum += np.sum(-(1 / (2 * sigma**2)) * np.abs(Z_i - z_omega_i)**2 - np.log(np.sqrt(2 * np.pi * sigma**2)))

    # Return WBIC value
    return -beta * log_likelihood_sum

def find_best_fit_model(frequencies, Z_measured, sigma=10, K_max=2, n_iterations=10000, burn_in=1000):
    wbic_values = []
    R_samples_n = []
    C_samples_n = []

    for k in range(1, K_max + 1):
        R_n, C_n, R_samples, C_samples = parallel_tempering_multithreaded(frequencies, Z_measured, k, sigma, 
                                                             n_iterations=n_iterations, burn_in=burn_in)
        wbic_value = wbic(frequencies, Z_measured, k, R_samples, C_samples, sigma)
        # wbic_values.append(cp.asnumpy(wbic_value))
        # R_samples_n.append(cp.asnumpy(R_samples))
        # C_samples_n.append(cp.asnumpy(C_samples))
        wbic_values.append(wbic_value)
        R_samples_n.append(R_samples)
        C_samples_n.append(C_samples)

    
    # wbic_values_np = np.array(wbic_values)  # Convert list to NumPy array
    best_model = np.argmin(wbic_values) + 1
    return best_model, wbic_values, R_samples_n, C_samples_n


def single_mcmc(data_filename, K, sigma=10, itterations=10000, burn_in=1000, print_results=False):
    pass
    # data = pd.read_csv(data_filename)
    # frequency_list = data['Frequency']  # Hz
    # Z_complex = np.array(data['Z_real']) + 1j * np.array( data['Z_imag'])

    # R_samples = []
    # C_samples = []
    # R_n, C_n, accepted_proposals, R_samples, C_samples = metropolis(frequency_list, Z_complex, K, sigma, n_iterations=itterations, burn_in=burn_in)
    # acceptance_rate = accepted_proposals / itterations * 100
    # if print_results:
    #     print(f"Acceptance rate: {acceptance_rate:.2f}%")
    #     if len(R_n) == 1:
    #         print(f"Estimated R value: {np.mean(R_n[0]):.2f}")
    #         print(f"Estimated C value: {np.mean(C_n[0]):.2e}")
    #     else:
    #         print(f"Estimated R value: {np.mean(R_n[0]):.2f}\t{np.mean(R_n[1]):.2f}")
    #         print(f"Estimated C value: {np.mean(C_n[0]):.2e}\t{np.mean(C_n[1]):.2e}")
    #     print('')

    # results = {
    #     'R_n': R_n,
    #     'C_n': C_n,
    #     'R_samples': R_samples,
    #     'C_samples': C_samples,
    #     'acceptance_rate': acceptance_rate,
    #     'frequency_list': frequency_list,
    #     'Z_complex': Z_complex
    # }
    # return results

def impedance_plot(frequencies, Z_measured, R, C, ax=None): 
    if ax is None:
        ax = plt.gca()  # Get the current axes if none is provided

    Z_model = [circuit_impedance(omega, R, C) for omega in frequencies]

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
    ax_r.plot(R_samples, label='Resistance (Ohms)')
    ax_r.set_title('R_n values')
    ax_r.set_xlabel('Iteration')
    ax_r.set_ylabel('Value')
    ax_r.legend()

    ax_c.plot(C_samples, label='Capacitance (Farads)')    
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

    for r, c in zip(R_samples, C_samples):
        # Combine the lists as pairs, sort by the first element, and unpack
        combined = sorted(zip(r, c), key=lambda x: x[0])
        # Separate them back into two lists
        sorted_foo, sorted_bar = zip(*combined)
        sorted_R_samples.append(list(sorted_foo))
        sorted_C_samples.append(list(sorted_bar))

    return sorted_R_samples, sorted_C_samples

import numpy as np
def parallel_run(file, custom_frequencies, custom_Z_measured, sigma, K_max, n_iterations, burn_in, rund_idx):
    best_model, wbic_values, R_samples, C_samples = find_best_fit_model(
        custom_frequencies, custom_Z_measured, sigma=sigma, K_max=K_max,
        n_iterations=n_iterations, burn_in=burn_in
    )
    return best_model, R_samples, C_samples

from concurrent.futures import ProcessPoolExecutor
from functools import partial
import csv

# Updated main block for parallel execution
if __name__ == '__main__':
    custom_data_files = ["circuit1_data.csv", "circuit2_data.csv", "generated_data_1.csv", "generated_data_2.csv", "generated_data_3.csv"]
    
    for file in custom_data_files:
        print("############################################################################################################")
        print(f"Processing {file}")
        custom_data = pd.read_csv(file)
        custom_frequencies = np.array(custom_data['Frequency'])  # Hz
        custom_Z_measured = np.array(custom_data['Z_real']) + 1j * np.array(custom_data['Z_imag'])

        
        K_max = 3
        if file in ["circuit1_data.csv", "circuit2_data.csv"]:
            K_max = 2
        
        wbic_values_n = []
        R_samples_n = []
        C_samples_n = []
        
        num_parallel = 5  # Number of parallel runs
        K_n = np.zeros(K_max)

        # Define the function with partial arguments
        run_function = partial(parallel_run, file, custom_frequencies, custom_Z_measured, 10, K_max, 80_000, 70_000)

        for _ in range(10):
            # Use ProcessPoolExecutor for parallel processing
            with ProcessPoolExecutor() as executor:
                # Submit tasks in parallel
                results = list(tqdm(executor.map(run_function, range(num_parallel)), total=num_parallel, desc=f"Parallel Runs for {file}"))
            
            # Collect results
            for best_model, R_samples, C_samples in results:
                K_n[best_model - 1] += 1
                R_samples_n.append(R_samples)
                C_samples_n.append(C_samples)

        # Normalize K_n
        K_n = K_n / np.sum(K_n)
        print(f"Best fit model for {file}: K = {np.argmax(K_n) + 1}\n")
        
        K_best = np.argmax(K_n) + 1

        output_filename = f"{file.split('.')[0]}_K{K_best}_samples.pkl"

        # Save as a pickle file for arbitrary Python objects
        with open(output_filename, "wb") as f:
            pickle.dump({"R_samples": R_samples_n, "C_samples": C_samples_n}, f)
        print(f"Saved R and C samples of K_best to {output_filename}")

        # # Save the R and C samples of K_best to a CSV file
        # csv_filename = f"{file.split('.')[0]}_K{K_best}_samples.csv"
        # file1 = open(csv_filename, "w")
        # file1.write("R_samples, C_samples\n")
        # for i in range(len(R_samples_n[K_best-1])):
        #         file1.write(f"{R_samples_n[K_best-1][i]},{C_samples_n[K_best-1][i]}\n")
        # file1.close()

        # df.to_csv(csv_filename, index=False)
        # print(f"Saved R and C samples of K_best to {csv_filename}")
        # np.savez(output_filename, R_samples=R_samples_n[K_best-1], C_samples=C_samples_n[K_best-1])
        # print(f"Saved R and C samples of K_best to {output_filename}")

        if len(R_samples_n) > 1:
            R_samples_n, C_samples_n = sort_samples(R_samples_n, C_samples_n)
        print(f"Best fit model for {file}: K = {K_best}\n")
        for j in range(K_max):
            for i in range(j + 1):
                print(f"Estimated R value for R[{i}]: {np.mean(R_samples_n[j][i]):.2f}")
                print(f"Estimated C value for C[{i}]: {np.mean(C_samples_n[j][i]):.2e}")
            print('')
            
        # Plotting the bar chart of K_n
        # plt.bar(range(1, K_max + 1), K_n)
        # plt.xlabel('K (Number of RC circuits)')
        # plt.ylabel('Frequency')
        # plt.title(f'Frequency of Best Fit Model Selection for {file}')
        # plt.xticks(range(1, K_max + 1))  # Ensure only integer values are shown on x-axis
        # plt.show()