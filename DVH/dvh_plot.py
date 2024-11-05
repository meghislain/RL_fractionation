import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy import interpolate
import glob





def load_dvh(name):
    
    file_list = glob.glob('dose_volume_data/dvh_parotid/*.npz')
    
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

    
    
simulated_volumes_rl, simulated_dose_bins_rl = load_dvh(name='dvh_rl')
average_volumes_rl = np.mean(simulated_volumes_rl, axis=0)
std_volumes_rl = np.std(simulated_volumes_rl, axis=0)

simulated_volumes_baseline, simulated_dose_bins_baseline = load_dvh(name='dvh_baseline')
average_volumes_baseline = np.mean(simulated_volumes_baseline, axis=0)
std_volumes_baseline = np.std(simulated_volumes_baseline, axis=0)

simulated_volumes_rl_pvt, simulated_dose_bins_rl_pvt = load_dvh(name='PVT_rl')
average_volumes_rl_pvt = np.mean(simulated_volumes_rl_pvt, axis=0)
std_volumes_rl_pvt = np.std(simulated_volumes_rl_pvt, axis=0)

simulated_volumes_baseline_pvt, simulated_dose_bins_baseline_pvt = load_dvh(name='PVT_baseline')
average_volumes_baseline_pvt = np.mean(simulated_volumes_baseline_pvt, axis=0)
std_volumes_baseline_pvt = np.std(simulated_volumes_baseline_pvt, axis=0)




fig, ax = plt.subplots(1, 1, figsize = (28,18))  # Create 2 subplots
light_orange = '#FFD280'
# Subplot for baseline treatment
ax.plot(simulated_dose_bins_baseline, average_volumes_baseline, label='Mean DVH using baseline treatment', color='blue', linewidth=3)
ax.fill_between(simulated_dose_bins_baseline, np.minimum(np.maximum(average_volumes_baseline - std_volumes_baseline, 0), 1.0), 
                np.minimum(average_volumes_baseline + std_volumes_baseline, 1.0), color='lightblue', alpha=0.5, interpolate=True)
#axs[1].plot(simulated_dose_bins_baseline_pvt, average_volumes_baseline_pvt, label='Mean PVT using baseline treatment', color='blue', linewidth=3)
#axs[1].fill_between(simulated_dose_bins_baseline_pvt, np.minimum(np.maximum(average_volumes_baseline_pvt - std_volumes_baseline_pvt, 0), 1.0), 
#                np.minimum(average_volumes_baseline_pvt + std_volumes_baseline_pvt, 1.0), color='lightblue', alpha=0.5, interpolate=True)

# Subplot for RL treatment
ax.plot(simulated_dose_bins_rl, average_volumes_rl, label='Mean DVH using RL-based treatment', color='orange', linewidth=3)
ax.fill_between(simulated_dose_bins_rl, np.minimum(np.maximum(average_volumes_rl - std_volumes_rl, 0), 1.0), 
                np.minimum(average_volumes_rl + std_volumes_rl, 1.0), color=light_orange, alpha=0.5, interpolate=True)
#axs[1].plot(simulated_dose_bins_rl_pvt, average_volumes_rl_pvt, label='Mean PVT using RL-based treatment', color='orange', linewidth=3)
#axs[1].fill_between(simulated_dose_bins_rl_pvt, np.minimum(np.maximum(average_volumes_rl_pvt - std_volumes_rl_pvt, 0), 1.0), 
#                np.minimum(average_volumes_rl_pvt + std_volumes_rl_pvt, 1.0), color=light_orange, alpha=0.5, interpolate=True)


ax.set_xlabel('Dose (Gy)', fontsize = 55)
ax.set_ylabel('Volume Fraction (-)', fontsize = 55)
ax.legend(fontsize=50)
ax.tick_params(axis='both', labelsize=42)

# Add a title for the whole figure
fig.suptitle('Dose-Volume Histogram for parotid tumor', fontsize = 55)
plt.grid(alpha=0.5)
plt.savefig("comparative_dvh_rl_baseline.svg")
plt.show()
