# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 19:36:26 2023

@author: Florian Martin

"""

import argparse
import datetime
import pickle
import os
print(datetime.datetime.now())
import numpy as np

dir_path = '/home/meghislain/ARIES-RL/tumor_growth_model/Results'

parser = argparse.ArgumentParser(description='2D Env. with Tabular Agent')

#parser.add_argument("mode", type = str, choices = ["train", "test"])


parser.add_argument("--epochs", type = int, help="Number of epochs, train steps, test steps", nargs = 3, default = [20, 2500, 100])
parser.add_argument("--alpha", type = float, default = 26.5*1e-2)
parser.add_argument("--beta", type = float, default = 54.0*1e-3)


args = parser.parse_args()
print(args)

from conventional_treatment import Agent 
agent = Agent(episodes=100, path=dir_path, agent_name="Rectum_retrained_14", agent_type="SARSAgent")

# from conventional_treatments import Rectal

# cancer = Rectal(5)

# """
# alpha_tumor = args.alpha
# beta_tumor  = args.beta

# cancer = Rectal(alpha_tumor, beta_tumor)

# results = cancer.baseline(episodes=args.epochs[2])

# path = '/linux/martinflor/Radiotherapy/cellular_model/Results/'

# with open(path + f"rectum_{alpha_tumor}_{beta_tumor}.txt", "w") as f:
#     f.write('Performances of the agent on the new environment' + '\n')
#     f.write("TCP :" + str(results["TCP"]) + "\n") 
#     f.write("NTCP (rectal bleeding): " + str(np.mean(results["NTCP"])) + " std dev: " + str(np.std(results["NTCP"])) + "\n") 
#     f.write("Average num of fractions: " + str(np.mean(results["fractions"])) + " std dev: " + str(np.std(results["fractions"])) + "\n") 
#     f.write("Average duration: " + str(np.mean(results["duration"])) + " std dev: " + str(np.std(results["duration"])) + "\n") 
#     f.write("Average survival: " + str(np.mean(results["survival"])) + " std dev: " +  str(np.std(results["survival"])) + "\n") 
#     f.write("Average radiation dose: " + str(np.mean(results["doses"])) + " std dev: " + str(np.std(results["doses"])) + "\n")

# #cancer.run(n_epochs=args.epochs[0], train_steps=args.epochs[1], test_steps=args.epochs[2], final_epsilon=0.05, final_alpha=0.5)
# """