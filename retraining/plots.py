# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 18:25:44 2023

@author: Florian Martin

"""


import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import os
import pickle
import numpy as np
import pandas as pd

dir_path = os.path.dirname(os.path.realpath(__file__))


class AgentResults:
    def __init__(self, result_name):
        
        self.results = []
        self.fractions = []
        self.doses = []
        self.duration = []
        self.TCPs = []
        self.survivals = []
        self.rewards = {}
        self.dicts = []
        self.epsilons = []
        self.alphas = []
        self.counts = []
        self.start_hour = []
        self.start_nb = []
        self.names = []
        
        self.list_dir = list_dir = [(f.name, f.path) for f in os.scandir(dir_path) if (f.is_dir()) and (result_name in f.name) and ('pycache' not in f.name)]
    
        for i in range(len(self.list_dir)):
            self.names.append(self.list_dir[i][0])
            with open(self.list_dir[i][1] + f'\\results_{self.list_dir[i][0]}' + '.pickle', 'rb') as file:
                tmp_dict = pickle.load(file)
                self.dicts.append(tmp_dict)
                self.fractions.append(str(np.mean(tmp_dict["fractions"])) + f' \u00B1 {np.std(tmp_dict["fractions"]):.2f}')
                self.doses.append(str(np.mean(tmp_dict["doses"])) + f' \u00B1 {np.std(tmp_dict["doses"]):.2f}')
                self.duration.append(str(np.mean([t for t in tmp_dict["duration"]])) + f' \u00B1 {np.std([t for t in tmp_dict["duration"]]):.2f}')
                self.TCPs.append(tmp_dict["TCP"])
                self.survivals.append(np.mean(tmp_dict["survival"]))
                self.results.append(tmp_dict)
                self.epsilons.append(tmp_dict["epsilon"])
                if "rewards" in tmp_dict.keys():
                    self.rewards[i] = tmp_dict["rewards"]
                    
                self.q_table = np.load(self.list_dir[i][1] + f'\\q_table_{self.list_dir[i][0]}' + '.npy', allow_pickle=False)
                count = 0
                for x, x_vals in enumerate(self.q_table):
                    for y, y_vals in enumerate(x_vals):
                        if all(x==y_vals[0] for x in y_vals):
                            count += 1
                            
                self.counts.append(count)
                    
                print(i, f'Mean : {np.mean(tmp_dict["survival"])}, Std : {np.std(tmp_dict["survival"])}')
                file.close()
            
            with open(self.list_dir[i][1] + f'\\SARSAgent{self.list_dir[i][0]}' + '.pickle', 'rb') as file2:
                tmp = pickle.load(file2)
                self.start_hour.append(np.argmax(tmp.env.cancer_arr[:355]))
                self.start_nb.append(np.max(tmp.env.cancer_arr[:355]))
    
        self.dict_ = {"fractions" : self.fractions, 
                      "doses" : self.doses, 
                      "duration" : self.duration, 
                      "TCP" : self.TCPs, 
                      "survival" : self.survivals,
                      "Start Hour" : self.start_hour,
                      "Nb Start CC" : self.start_nb,
                      "name" : self.names
                      }
    
    def get_results(self):
        return self.dict_
    
    def print_results(self):
        df = pd.DataFrame.from_dict(self.dict_)
        print(df)
        return df
        
    def print_mean_std(self):
        df = pd.DataFrame.from_dict(self.dict_)
        print(df.mean())
        print("\n")
        print(df.std())
        
    def print_mean_std_TCP(self):
        df = pd.DataFrame.from_dict(self.dict_)
        df = df[df["TCP"] == 100.0]
        df.reset_index(inplace=True)
        print(df.mean())
        print("\n")
        print(df.std())
        
    def print_best_results(self):
        df = pd.DataFrame.from_dict(self.dict_)
        df = df[df["TCP"] == 100.0]
        df.reset_index(inplace=True)
    
        print(df)
        print(f"Smallest fractions : {np.min(df.fractions)} at index {np.argmin(df.fractions)}")
        print(f"Smallest dose : {np.min(df.doses)} at index {np.argmin(df.doses)}")
        print(f"Smallest duration : {np.min(df.duration)} at index {np.argmin(df.duration)}")
        return df


name = 'radio07'
Sarsa = AgentResults(name)
df = Sarsa.print_results()


print(df)
print(f"Smallest fractions : {np.min(df.fractions)} at index {np.argmin(df.fractions)}")
print(f"Smallest dose : {np.min(df.doses)} at index {np.argmin(df.doses)}")
print(f"Smallest duration : {np.min(df.duration)} at index {np.argmin(df.duration)}")
print(f"Biggest Survival : {np.max(df.survival)} at index {np.argmax(df.survival)}")

with open(f'summary_{name}.txt', 'w') as file:
    #df_ = df.sort_values(by=['TCP'])
    df_ = df.drop(columns=['name', 'Nb Start CC', 'Start Hour'])
    df_ = df_.reset_index(drop=True)
    file.write(df_.to_latex(escape=False, index=False))
