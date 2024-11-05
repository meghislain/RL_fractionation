# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 13:08:14 2023

@author: Florian Martin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy import interpolate
import glob

def load_DVH(file_name):
    data = np.load(file_name)
    volumes = data['volumes']
    dose_bins = data['dose_bins']
    return volumes, dose_bins


def clinical_DVH(dose_bins, file_name):
    # Read the CSV file, specifying ',' as the decimal point
    df = pd.read_csv(file_name, decimal=',', sep = '.', names = ["doses", "volumes"])

    # Get the doses and volumes
    clinical_doses = df.iloc[:, 0].values
    clinical_volumes = df.iloc[:, 1].values
    return clinical_volumes, clinical_doses


def compare_DVHs(simulated_volumes, clinical_volumes, simulated_dose_bins):
    assert simulated_volumes.shape == clinical_volumes.shape, "Simulated and clinical volumes must have the same shape."

    diff = simulated_volumes - interpolated_clinical_volumes
    underestimation_penalty = 2.0
    diff = np.where(diff < 0, diff * underestimation_penalty, diff)
    weights = simulated_dose_bins / np.max(simulated_dose_bins)
    
    rmse = np.sqrt(np.mean(weights * diff**2))
    
    return rmse

def comparative_plot(simulated_volumes, interpolated_clinical_volumes, real_clinical_volumes, simulated_dose_bins, clinical_dose_bins, real_dose_bins):
    rmse = compare_DVHs(simulated_volumes, interpolated_clinical_volumes, simulated_dose_bins)

    # Plot the simulated and clinical DVH
    plt.figure(figsize=(20,12))
    plt.plot(simulated_dose_bins, simulated_volumes, label='Simulated DVH', color='blue', linewidth = 4)
    plt.plot(clinical_dose_bins, interpolated_clinical_volumes, label='Interpolated Clinical DVH', linestyle='-', color='orange', linewidth = 4)
    plt.scatter(real_dose_bins, real_clinical_volumes, label='Clinical DVH', color='red', s=75)

    plt.xlabel('Dose (Gy)', fontsize = 40)
    plt.ylabel('Volume Fraction', fontsize = 40)
    plt.title(f'Comparative Dose-Volume Histogram\n RMSE: {rmse:.2f}', fontsize = 40)
    plt.legend(fontsize = 32)
    plt.xticks(fontsize=36)
    plt.yticks(fontsize=36)
    plt.grid(True)
    plt.savefig('comp_dvh.svg')
    plt.show()

    
def interpolate_clinical_dvh(simulated_dose_bins, clinical_volumes, clinical_dose_bins):
    interp_func = interpolate.interp1d(clinical_dose_bins, clinical_volumes, fill_value="extrapolate", kind='linear')
    interpolated_clinical_volumes = interp_func(simulated_dose_bins)
    interpolated_clinical_volumes = np.where(np.isinf(interpolated_clinical_volumes), 0, interpolated_clinical_volumes)
    first_non_zero_index = np.nonzero(interpolated_clinical_volumes)[0][0]
    interpolated_clinical_volumes[:first_non_zero_index] = 1.0
    return interpolated_clinical_volumes




file_list = glob.glob('dose_volume_data/*.npz')

all_volumes = []
all_dose_bins = []

for file in file_list:
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
average_volumes = np.mean(all_volumes, axis=0)

simulated_volumes, simulated_dose_bins = average_volumes[:], master_dose_bins[:]

#simulated_volumes, simulated_dose_bins = load_DVH("dose_volume_data.npz")
clinical_volumes, clinical_dose_bins = clinical_DVH(simulated_dose_bins, "dvh_tucker.csv")
interpolated_clinical_volumes = interpolate_clinical_dvh(simulated_dose_bins, clinical_volumes, clinical_dose_bins)


comparative_plot(simulated_volumes, interpolated_clinical_volumes, clinical_volumes, simulated_dose_bins, simulated_dose_bins, clinical_dose_bins)
