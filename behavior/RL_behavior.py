import glob
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

"""
path = 'Rectum/'
mean_NTCP = 0.0007604265270609573
ntcp_data = np.load(path + 'NTCP.npz')
ntcp_weights = ntcp_data["NTCP"] / mean_NTCP
"""

path = 'Lung/RL/'


# Load behavior data
file_list = glob.glob(path + 'behavior_*.npz')
all_times = []
all_doses_per_hour = []
all_healthy_arr = []
all_cancer_arr = []
all_ntcp = []

for file in file_list:
    data = np.load(file)
    all_times.append(data['time'])
    all_doses_per_hour.append(data['doses_per_hour'].astype(np.float64))
    all_healthy_arr.append(data['healthy_arr'][1:])
    all_cancer_arr.append(data['cancer_arr'][1:])
    all_ntcp.append(float(data['NTCP']))

print(f"Max NTCP : {np.max(all_ntcp)} at index {np.argmax(all_ntcp)}")
print(f"Min NTCP : {np.min(all_ntcp)} at index {np.argmin(all_ntcp)}")

# Determine the length of the master time array
master_time_length = max(len(time) for time in all_times)
nb_div = np.zeros(master_time_length)
for i in range(len(all_times)):
    for j in range(len(all_doses_per_hour[i])):
        nb_div[j] += 1

def mean_std_arrays(array):
    nb = 5
    
    for i in range(len(all_times)):
        array[i] = array[i][:np.sum(nb_div >= nb)]
        
    master_time_length = np.sum(nb_div >= nb)
    sum_per_hour = np.zeros(master_time_length)
    counts = np.zeros(master_time_length)
    
    for i in range(len(all_times)):
        per_hour = array[i]
        padded = np.pad(per_hour, (0, master_time_length - len(per_hour)), 'constant')
        sum_per_hour += padded
        counts += (padded > 0)
    
    mean_per_hour = sum_per_hour / counts
    mean_per_hour = np.nan_to_num(mean_per_hour)
    
    
    sum_squared_diffs_per_hour = np.zeros(master_time_length)
    
    for i in range(len(all_times)):
        per_hour = array[i]
        padded = np.pad(per_hour, (0, master_time_length - len(per_hour)), 'constant')
        squared_diffs = (padded - mean_per_hour) ** 2 * (padded > 0)
        sum_squared_diffs_per_hour += squared_diffs
    
    variance_per_hour = sum_squared_diffs_per_hour / counts
    variance_per_hour = np.nan_to_num(variance_per_hour)
    std_per_hour = np.sqrt(variance_per_hour)
    
    return mean_per_hour, std_per_hour

mean_doses_per_hour, std_doses_per_hour = mean_std_arrays(all_doses_per_hour)
mean_doses_per_hour = np.clip(mean_doses_per_hour, 1.0, 4.0)

min_doses_per_hour = all_doses_per_hour[np.argmin(all_ntcp)]
max_doses_per_hour = all_doses_per_hour[np.argmax(all_ntcp)]


mean_healthy_per_hour, std_healthy_per_hour = mean_std_arrays(all_healthy_arr)
mean_cancer_per_hour, std_cancer_per_hour = mean_std_arrays(all_cancer_arr)


# Define a new time array that represents the time intervals at which the measurements were taken
time_intervals = np.arange(0, len(mean_doses_per_hour) * 24, 24)
min_time_intervals = np.arange(0, len(min_doses_per_hour) * 24, 24)
max_time_intervals = np.arange(0, len(max_doses_per_hour) * 24, 24)

fig, ax = plt.subplots(1, 1, figsize=[32,16])
ax.plot(time_intervals, mean_doses_per_hour, label=r'$\mu_{dose}$', linewidth=5)


#Parotid
h_arr = [3856, 3742, 3808, 4003, 3994]
c_arr = [1899, 304, 33.4, 4.0, 1.81]
idx = [0, 2, 4, 7, 10]

from adjustText import adjust_text

texts = []
for j, (h,c) in enumerate(zip(h_arr, c_arr)):
    texts.append(ax.text(time_intervals[idx[j]], mean_doses_per_hour[idx[j]], f"{h:.1f}\n{c:.1f}", fontsize=40))

adjust_text(texts)


#Rectum
"""
h_arr = [3769, 3677, 3728, 3904, 3961, 3931]
c_arr = [1233.6, 569, 62, 11.2, 3.34, 2.67]
idx = [0, 1, 3, 5, 7, 10]

from adjustText import adjust_text

texts = []
for j, (h,c) in enumerate(zip(h_arr, c_arr)):
    texts.append(ax.text(time_intervals[idx[j]], mean_doses_per_hour[idx[j]], f"{h:.1f}\n{c:.1f}", fontsize=40))

adjust_text(texts)
"""

#Lung
"""
h_arr = [3648, 3581, 3772, 3873, 4047, 3873]
c_arr = [1868.2, 62.2, 11.6, 2.6, 2.7, 1.7]
idx = [0, 4, 6, 10, 14, 16]

from adjustText import adjust_text

texts = []
for j, (h,c) in enumerate(zip(h_arr, c_arr)):
    texts.append(ax.text(time_intervals[idx[j]], mean_doses_per_hour[idx[j]], f"{h:.1f}\n{c:.1f}", fontsize=40))

adjust_text(texts)
"""
ax.fill_between(time_intervals, mean_doses_per_hour - std_doses_per_hour, np.minimum(mean_doses_per_hour + std_doses_per_hour, 4.0), alpha=0.25, label=r'$\sigma_{dose}$')
ax.scatter(time_intervals[idx], mean_doses_per_hour[idx], marker='o', s=500)
# ax.set_title('Radiation Dose provided by the RL-based treatment for parotid tumor \n', fontsize = 50)
ax.set_xlabel('Time (hours)', fontsize = 50)
ax.set_ylabel('Radiation Dose (Gy)', fontsize = 50)
ax.set_ylim(1., 4.25)
ax.legend(fontsize = 45, loc=0)
plt.xticks(fontsize=40)
plt.yticks(fontsize=40)
plt.grid(alpha=0.5)

plt.tight_layout()
plt.savefig("parotid_dose_behavior.svg")
plt.show()

np.savez(f"Treatments/mean_parotid.npz", treatment=mean_doses_per_hour)


nb_stages_cancer = 50
nb_stages_healthy = 5

from math import ceil

def ccell_state(count):
    if count <= 10:
        return count
    if count < 500:
        div = (500-10)/15
        return int(ceil(count/div))+9
    if count > 7000:
        if count < 8000:
            return nb_stages_cancer - 2
        else: 
            return nb_stages_cancer - 1
    else: 
        div = (7000-500)/22
        return min(nb_stages_cancer - 1, int(ceil(count/div))+24)

def hcell_state(count):
    return min(nb_stages_healthy - 1, max(0, int(ceil((count-(2875+375))/375))))

def convert(obs):
    discrete_state = (ccell_state(obs[1]), hcell_state(obs[0]))
    return discrete_state

"""
q_table = np.load("C:/Users/Ineed/OneDrive/Bureau/cellular_model/Agents/17/q_table_17.npy", allow_pickle=True)

best_actions = np.argmax(q_table, axis=2)

# Visualize the best actions
plt.figure(figsize=(10, 10))
plt.imshow(best_actions, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.title("Best Action for Each State")
plt.xlabel("State Dimension 1")
plt.ylabel("State Dimension 2")
plt.show()
"""

"""
# Number of random samples
num_samples = 10

# Randomly select simulations
selected_indices = np.random.choice(len(all_healthy_arr), num_samples)

# Create a figure
fig, axs = plt.subplots(num_samples, figsize=[32,16*num_samples])

# Plot the selected trajectories along with discrete states
for i, idx in enumerate(selected_indices):
    healthy_cells = all_healthy_arr[idx]
    cancer_cells = all_cancer_arr[idx]
    actions = [np.argmax(q_table[convert((healthy_cells[j], cancer_cells[j]))]) for j in range(len(healthy_cells))]

    healthy_cells_percentage = 100*healthy_cells / np.max(healthy_cells)
    cancer_cells_percentage = 100*cancer_cells / np.max(cancer_cells)
    
    axs[i].plot(all_times[idx], healthy_cells_percentage, label='Healthy Cells', color='green')
    axs[i].plot(all_times[idx], cancer_cells_percentage, label='Cancer Cells', color='red')

    # Annotate with discrete states
    for j, action in enumerate(actions):
        axs[i].annotate(f"{action+1} Gy", (all_times[idx][j], healthy_cells_percentage[j]), fontsize=22, verticalalignment='bottom', horizontalalignment='right')
        axs[i].annotate(f"{action+1} Gy", (all_times[idx][j], cancer_cells_percentage[j]), fontsize=22, verticalalignment='top', horizontalalignment='left')

    axs[i].set_xlabel('Time (hours)', fontsize=20)
    axs[i].set_ylabel('Cell Count', fontsize=20)
    axs[i].legend(fontsize=20)
    axs[i].set_title(f'Simulation {idx}', fontsize=25)
    axs[i].grid(alpha=0.5)
    
    axs[i].set_xticks(np.arange(0, len(all_times[idx]) * 24, 24))

plt.tight_layout()
plt.savefig("simulation_examples.svg")
plt.show()
"""