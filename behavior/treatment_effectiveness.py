# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 10:26:46 2023

@author: Ineed
"""

import numpy as np
import matplotlib.pyplot as plt
import glob

# List of uploaded files
path = "C:/Users/Ineed/OneDrive/Bureau/cellular_model/behavior/Treatments/"
file_names = glob.glob(path + '*parotid.npz')

# Load data from each file and store in a dictionary
data_dict = {}

for file_name in file_names:
    with np.load(file_name) as data:
        for key in data.keys():
            data_dict[f"{file_name.split('/')[-1].replace('.npz', '')}_{key}"] = data[key]

NTCP = {'Treatments\\mean_parotid_treatment' : 31.29,
        'Treatments\\min_parotid_treatment' : 39.36,
        'Treatments\\test_parotid_treatment' : 29.11,
        'Treatments\\test2_parotid_treatment' : 42.15}

fig, ax = plt.subplots(1, 1, figsize=(24, 18))

# Plot each dataset
for key in data_dict.keys():
    label, data = NTCP[key], data_dict[key]
    ax.plot(np.arange(0,len(data)*24,24), data, label=f"NTCP = {label}")

max_length = max([len(data) for data in data_dict.values()])
ax.set_xticks(ticks=np.arange(0,max_length*24,24), labels=np.arange(0,max_length*24,24), fontsize=25)
ax.set_xlabel('Hours', fontsize=42)
ax.set_ylabel('Dose (Gy)', fontsize=42)
ax.set_title('Radiotherapy treatments comparison', fontsize=50)
plt.legend(fontsize=32, loc=1)
ax.tick_params(axis='both', labelsize=25)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.show()
