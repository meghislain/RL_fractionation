# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 16:39:05 2023

@author: Florian Martin

"""

import matplotlib.pyplot as plt
import matplotlib
import matplotlib.animation as animation 
import seaborn as sns
import imageio
import random
import numpy as np
import os
import pickle
from environment import GridEnv
from grid import Grid
from math import exp, log, ceil, floor
from cell import HealthyCell, CancerCell, OARCell


nb_stages_cancer = 50
nb_stages_healthy = 5
nb_actions        = 4

dir_path = '/linux/martinflor/RL-Radiotherapy/RE-TRAIN/Results/'
     
class Agent:
    
    def __init__(self, env, gamma, alpha, epsilon, q_table,
                 anim_simu=False, anim_q_table=False, anim_results=False):
        
        self.env = env
        self.nb_stages_healthy = nb_stages_healthy
        self.nb_stages_cancer  = nb_stages_cancer
        self.nb_actions        = nb_actions
        
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        
        self.anim_simu = anim_simu
        
        self.q_table = q_table
        
    def choose_action(self, state):
        if np.random.random() < self.epsilon :
            return np.random.randint(self.nb_actions, dtype=int)
        else :
            actions = np.argwhere(self.q_table[state]==np.max(self.q_table[state])).flatten()
            return np.random.choice(actions)
    
    def train(self, steps):
        
        self.env.reset()
        while steps > 0:
            print(steps)
            while not self.env.inTerminalState() and steps > 0:
                state = self.env.convert(self.env.observe())
                action = self.choose_action(state)
                reward = self.env.act(action)
                next_state = self.env.convert(self.env.observe())
                
                self.update(state, next_state, action, reward)
                steps -= 1
            if steps > 0:
                self.env.reset()
    
    def test(self, episodes):
        lengths_arr = []
        fracs_arr = []
        doses_arr = []
        survivals_arr = []
        
        doses_per_hour = {}
        fracs_per_hour = {}
        rewards = []
        self.states = np.zeros((self.nb_stages_cancer, self.nb_stages_healthy))
        
        sum_w = 0
        for ep in range(episodes):
            self.env.reset()
            sum_r = 0
            fracs = 0
            doses = 0
            time = 0
            doses_per_hour[ep] = {}
            init_hcell = HealthyCell.cell_count
            while not self.env.inTerminalState():
                state = self.env.convert(self.env.observe())
                self.states[state] += 1
                action = np.argmax(self.q_table[state])
                reward = self.env.act(action)

                print(action + 1, "grays, reward =", reward)
                fracs += 1
                doses += action + 1
                doses_per_hour[ep][time] = action + 1
                time += 24
                sum_r += reward
                next_state = self.env.convert(self.env.observe())
                
                
            if self.env.end_type == 'W':
                sum_w += 1
            
            fracs_arr.append(fracs)
            doses_arr.append(doses)
            lengths_arr.append(time)
            survival = HealthyCell.cell_count / init_hcell
            survivals_arr.append(survival)
            rewards.append(sum_r)

            
        self.epochs_arr = np.arange(episodes)
        self.fracs_arr = np.array(fracs_arr)
        self.doses_arr = np.array(doses_arr)
        self.lengths_arr = np.array(lengths_arr)
        self.survivals_arr = np.array(survivals_arr)
        self.rewards = np.array(rewards)
        self.tcp = 100.0 *sum_w/episodes

        print("TCP: " , self.tcp)
        print("Average num of fractions: ", np.mean(self.fracs_arr), " std dev: ", np.std(self.fracs_arr))
        print("Average radiation dose: ", np.mean(self.doses_arr), " std dev: ", np.std(self.doses_arr))
        print("Average duration: ", np.mean(self.lengths_arr), " std dev: ", np.std(self.lengths_arr))
        print("Average survival: ", np.mean(self.survivals_arr), " std dev: ", np.std(self.survivals_arr))
        
        results = {"TCP" : self.tcp,
                    "fractions" : self.fracs_arr,
                   "doses" : self.doses_arr,
                   "duration" : self.lengths_arr,
                   "survival" : self.survivals_arr,
                   "doses_per_hour" : doses_per_hour,
                   "rewards" : self.rewards
                   }
        
        return results
        
    def save_table(self, name):
    
        fig, axs = plt.subplots(1, 2, constrained_layout=True, figsize = (10,8))
        im = axs[0].imshow(self.states)
        axs[0].set_xticks([0,1,2,3,4])
        axs[0].set_yticks([0,10,20,30,40,50])
        fig.colorbar(im, ax=axs[0])
        
        bbox = dict(boxstyle ="round", fc ="0.9")

  
        cellular_model = axs[1]
        cellular_model.set_axis_off()
        
        
        cellular_model.annotate(f'# Average Healthy Glucose Absorption = {self.env.average_healthy_glucose_absorption:.2f}',
            xy =(0., 0.7),
            fontsize = 18,
            bbox = bbox)
        
        cellular_model.annotate(f'# Average Cancer Glucose Absorption = {self.env.average_cancer_glucose_absorption:.2f}',
            xy =(0., 0.6),
            fontsize = 18,
            bbox = bbox)
        
        cellular_model.annotate(f'# Average Healthy Oxygen Consumption = {self.env.average_healthy_oxygen_consumption:.2f}',
            xy =(0., 0.5),
            fontsize = 18,
            bbox = bbox)
        
        cellular_model.annotate(f'# Average Cancer Oxygen Consumption = {self.env.average_cancer_oxygen_consumption:.2f}',
            xy =(0., 0.4),
            fontsize = 18,
            bbox = bbox)
        
        cellular_model.annotate(f'# Quiescent Glucose Level = {self.env.quiescent_glucose_level:.2f}',
            xy =(0., 0.3),
            fontsize = 18,
            bbox = bbox)
        
        cellular_model.annotate(f'# Quiescent Oxygen Level = {self.env.quiescent_oxygen_level:.2f}',
            xy =(0., 0.2),
            fontsize = 18,
            bbox = bbox)
        
        cellular_model.annotate(f'# Critical Glucose Level = {self.env.critical_glucose_level:.2f}',
            xy =(0., 0.1),
            fontsize = 18,
            bbox = bbox)
        
        cellular_model.annotate(f'# Critical Oxygen Level = {self.env.critical_oxygen_level:.2f}',
            xy =(0., 0.),
            fontsize = 18,
            bbox = bbox)
            
        plt.savefig(name + ".png")
    
    def run(self, n_epochs, train_steps, test_steps, final_epsilon, final_alpha, name):
        self.test_steps = test_steps
        
        epsilon_change = (self.epsilon - final_epsilon) / (n_epochs - 1)
        alpha_change = (self.alpha - final_alpha)       / (n_epochs - 1)
        
        for i in range(n_epochs):
            print("Epoch ", i + 1)
            self.train(steps=train_steps)
            self.results = self.test(episodes=test_steps)
            self.save(name + '_' + str(i), self.results)
            
            self.alpha   -= alpha_change
            self.epsilon -= epsilon_change
   
class RandomAgent(Agent):
    def __init__(self, env, gamma, alpha, epsilon,
                 anim_simu=False, anim_q_table=False, anim_results=False):
        super().__init__(env, gamma, alpha, epsilon=1.0, anim_simu=anim_simu, anim_q_table=anim_q_table, anim_results=anim_results)
        
        
    def save(self, name, results):
        filename = dir_path + "/Random/" 
        try:
          os.mkdir(filename+name)
        except:
          print("Overwritting")
        
        results["alpha"] = self.alpha
        results["epsilon"] = self.epsilon
        
        with open(filename+name+"/results_" + name + ".pickle", 'wb') as file:
            pickle.dump(results, file)
            
        with open(filename + name + "/Random" + name + ".pickle", 'wb') as file_agent:
            pickle.dump(self, file_agent)
            
        self.save_table(filename+name+"_table")
            
    def run(self, n_epochs, train_steps, test_steps, final_epsilon, final_alpha):
        self.test_steps = test_steps
        
        epsilon_change = (self.epsilon - final_epsilon) / (n_epochs - 1)
        alpha_change = (self.alpha - final_alpha)       / (n_epochs - 1)
        
        for i in range(n_epochs):
            print("Epoch ", i + 1)
            self.results = self.test(episodes=test_steps)
            self.save(str(i), self.results)
            
            self.alpha   -= alpha_change
            self.epsilon -= epsilon_change

class QAgent(Agent):
    
    def __init__(self, env, gamma, alpha, epsilon,
                 anim_simu=False, anim_q_table=False, anim_results=False):
        super().__init__(env, gamma, alpha, epsilon, anim_simu, anim_q_table, anim_results)

        
    def update(self, state, next_state, action, reward):
        
        new_q = (1-self.alpha)*self.q_table[state + (action,)] + self.alpha * (reward + self.gamma * np.max(self.q_table[next_state]))
        self.q_table[state + (action, )] = new_q
            
    def save(self, name, results):
        filename = dir_path + "/QAgent/" 
        try:
          os.mkdir(filename+name)
        except:
          print("Overwritting")
        np.save(filename+name+"/" + "q_table_" + name, self.q_table, allow_pickle=False)
        
        results["alpha"] = self.alpha
        results["epsilon"] = self.epsilon
        
        with open(filename+name+"/results_" + name + ".pickle", 'wb') as file:
            pickle.dump(results, file)
            
        with open(filename + name + "/QAgent" + name + ".pickle", 'wb') as file_agent:
            pickle.dump(self, file_agent)
            
        #table_name=filename + name + "_table"
        #self.save_table(name=table_name)
                
    def load(self, name):
        filename = dir_path + "/QAgent/" 
        self.q_table = np.load(filename+name+"/" + "q_table_" + name + '.npy', allow_pickle=False)
            
class SARSAgent(Agent):
    
    def __init__(self, env, gamma, alpha, epsilon,
                 anim_simu=False, anim_q_table=False, anim_results=False):
        super().__init__(env, gamma, alpha, epsilon, anim_simu, anim_q_table, anim_results)
        
    def update(self, state, next_state, action, next_action, reward):
        
        self.q_table[state + (action, )] = (1-self.alpha)*self.q_table[state + (action,)] + self.alpha * (reward + self.gamma * self.q_table[next_state + (next_action,)])
                
    def train(self, steps):
        
        self.env.reset()
        while steps > 0:
            print(steps)
            state = self.env.convert(self.env.observe())
            action = self.choose_action(state)
            
            while not self.env.inTerminalState() and steps > 0:
                
                reward = self.env.act(action)
                next_state = self.env.convert(self.env.observe())
                next_action = self.choose_action(next_state)
                
                self.update(state, next_state, action, next_action, reward)
                state = next_state
                action = next_action
                steps -= 1
            if steps > 0:
                self.env.reset()
            
    def save(self, name, results):
        filename = dir_path + "/SARSAgent/" 
        try:
          os.mkdir(filename+name)
        except:
          print("Overwritting")
        np.save(filename+name+"/" + "q_table_" + name, self.q_table, allow_pickle=False)
        
        results["alpha"] = self.alpha
        results["epsilon"] = self.epsilon
        
        with open(filename+name+"/results_" + name + ".pickle", 'wb') as file:
            pickle.dump(results, file)
            
        with open(filename + name + "/SARSAgent" + name + ".pickle", 'wb') as file_agent:
            pickle.dump(self, file_agent)
            
        #self.save_table(filename + name + "_table")
            
    def load(self, name):
        filename = dir_path + "/SARSAgent/" 
        self.q_table = np.load(filename+name+"/" + "q_table_" + name + '.npy', allow_pickle=False)
            
            
class ExpSARSAgent(Agent):
    
    def __init__(self, env, gamma, alpha, epsilon,
                 anim_simu=False, anim_q_table=False, anim_results=False):
        super().__init__(env, gamma, alpha, epsilon, anim_simu, anim_q_table, anim_results)
        
    def update(self, state, next_state, action, reward):

        q_max = np.max(self.q_table[next_state])
        nb_greedy_actions = 0
        for i in range(self.nb_actions):
            if self.q_table[next_state][i] == q_max:
                nb_greedy_actions += 1
     
        non_greedy_action_prob = self.epsilon/self.nb_actions
        greedy_action_prob = ((1 - self.epsilon)/nb_greedy_actions) + non_greedy_action_prob
 
        expected_q = 0
        for i in range(self.nb_actions):
            if self.q_table[next_state][i] == q_max:
                expected_q += self.q_table[next_state][i] * greedy_action_prob
            else:
                expected_q += self.q_table[next_state][i] * non_greedy_action_prob
     
        self.q_table[state + (action,)] = (1-self.alpha)*self.q_table[state + (action,)] + self.alpha * (reward + self.gamma * expected_q)
            
    def save(self, name, results):
        filename = dir_path + "/ExpSARSAgent/" 
        try:
          os.mkdir(filename+name)
        except:
          print("Overwritting")
        np.save(filename+name+"/" + "q_table_" + name, self.q_table, allow_pickle=False)
        
        results["alpha"] = self.alpha
        results["epsilon"] = self.epsilon
        
        with open(filename+name+"/results_" + name + ".pickle", 'wb') as file:
            pickle.dump(results, file)
            
        with open(filename + name + "/ExpSARSAgent" + name + ".pickle", 'wb') as file_agent:
            pickle.dump(self, file_agent)
            
        #self.save_table(filename + name + "_table")
            
    def load(self, name):
        filename = dir_path + "/ExpSARSAgent/" 
        self.q_table = np.load(filename+name+"/" + "q_table_" + name + '.npy', allow_pickle=False)
      
def patch_type_color(patch):
    if len(patch) == 0:
        return 0, 0, 0
    else:
        return patch[0].cell_color()
        
