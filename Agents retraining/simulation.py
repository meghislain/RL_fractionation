# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 16:39:05 2023

@author: Florian Martin

"""

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

dir_path = os.path.dirname(os.path.realpath(__file__))
     
class Agent:
    
    def __init__(self, env, gamma, alpha, epsilon, q_table=None):
        
        self.env = env
        self.nb_stages_healthy = nb_stages_healthy
        self.nb_stages_cancer  = nb_stages_cancer
        self.nb_actions        = nb_actions
        
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon

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

            
class SARSAgent(Agent):
    
    def __init__(self, env, gamma, alpha, epsilon):
        super().__init__(env, gamma, alpha, epsilon)
        
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
        filename = dir_path + "/retrained_rectum_16/"
        np.save(filename + name + ".npy", self.q_table, allow_pickle=False)
        
        results["alpha"] = self.alpha
        results["epsilon"] = self.epsilon
        
        with open(filename + name + ".pickle", 'wb') as file:
            pickle.dump(results, file)
            
        with open(filename + name + "_SARSAgent.pickle", 'wb') as file_agent:
            pickle.dump(self, file_agent)
            
    def load(self, path):
        print("Loading q-table")
        self.q_table = np.load(path, allow_pickle=False)

        
