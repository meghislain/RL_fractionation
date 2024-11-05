import numpy as np
from scipy.stats import norm
from scipy.integrate import quad
import glob
from scipy.interpolate import interp1d
from scipy.stats import multivariate_normal
from scipy import interpolate


def load_dvh(name):
    
    file_list = glob.glob('dose_volume_data/dvh_rectal/*.npz')
    
    all_volumes = []
    all_dose_bins = []
    
    for file in file_list:
        if not name in file:
            continue
        data = np.load(file)
        volumes = data['volumes']
        dose_bins = data['dose_bins']
        all_volumes.append(volumes)
        all_dose_bins.append(dose_bins)
    
    # Find the dose bins from the DVH with the maximum number of bins
    master_dose_bins = max(all_dose_bins, key=len)
    
    # Interpolate all DVHs to the master dose bins
    for i in range(len(all_volumes)):
        interp_func = interpolate.interp1d(all_dose_bins[i], all_volumes[i], fill_value=0.0, bounds_error=False)
        all_volumes[i] = np.maximum(interp_func(master_dose_bins), 0.0)
    
    
    all_volumes = np.array(all_volumes)
    return all_volumes, master_dose_bins

def Lyman_Kutcher_Burman(gEUD, TD50, m):
    # Lyman-Kutcher-Burman Model
    # uses gEUD in LKB model
    t = (gEUD - TD50) / (m * TD50)
    ntcp, _ = quad(lambda x: np.exp(-0.5 * x**2), -np.inf, t)
    return ntcp / np.sqrt(2 * np.pi)

def compute_gEUD(volumes, doses, n):
    
    # Calculating the gEUD
    gEUD = sum(v * (d**(1/n)) for v, d in zip(volumes, doses))
    gEUD = gEUD**n
    
    return gEUD



def compute_NTCP_monte_carlo(all_volumes, master_dose_bins, TD50_mean, TD50_std, m_mean, m_std, n_iterations):

    ntcps = np.zeros(n_iterations)

    # Define the distributions of the NTCP model parameters
    TD50_distribution = norm(loc=TD50_mean, scale=TD50_std)
    m_distribution = norm(loc=m_mean, scale=m_std)
    
    # Compute the mean and standard deviation of the volumes at each dose bin
    volume_means = np.mean(all_volumes, axis=0)
    volume_stds = np.std(all_volumes, axis=0)
    
    # Compute the covariance matrix of the volumes
    volume_cov = np.cov(all_volumes, rowvar=False)
    
    # Add a small constant to the diagonal to make the covariance matrix positive definite
    volume_cov += np.eye(volume_cov.shape[0]) * 1e-6
    
    
    
    # Define the multivariate normal distribution for the DVH
    DVH_distribution = multivariate_normal(mean=volume_means, cov=volume_cov)

    # Perform the Monte Carlo simulation
    for i in range(n_iterations):
        print(f"Monte Carlo iteration n°{i}")
        TD50_sample = TD50_distribution.rvs()
        m_sample = m_distribution.rvs()
        volumes_sample = DVH_distribution.rvs()
        volumes_sample = np.maximum(volumes_sample, 0)
        volumes_sample = np.minimum(volumes_sample, 1.0)

        gEUD = compute_gEUD(volumes_sample, master_dose_bins, n = 0.15)

        # Compute the NTCP and store it in the array
        ntcps[i] = Lyman_Kutcher_Burman(gEUD, TD50_sample, m_sample)

    # Compute the mean and standard deviation of the NTCP distribution
    ntcp_mean = np.mean(ntcps)
    ntcp_std = np.std(ntcps)

    return ntcp_mean, ntcp_std

simulated_volumes_rl, simulated_dose_bins_rl = load_dvh(name='baseline') 


# Set the mean and standard deviation for the NTCP parameters to arbitrary values
TD50_mean = 80.1  # Gy
TD50_std = 2.2  # Gy
m_mean = 0.15
m_std = 0.025

# Set the number of Monte Carlo iterations
n_iterations = 10000

# Compute the NTCP using the Monte Carlo method
ntcp_mean, ntcp_std = compute_NTCP_monte_carlo(simulated_volumes_rl, simulated_dose_bins_rl, TD50_mean, TD50_std, m_mean, m_std, n_iterations)

print(ntcp_mean, ntcp_std)
