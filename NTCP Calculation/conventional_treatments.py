import os
import random
import pickle
import numpy as np
from environment import GridEnv
from grid import Grid
from math import exp, log, ceil, floor
from cell import HealthyCell, CancerCell, OARCell
from NTCP_model import calculate_effective_dose, Lyman_Kutcher_Burman

dir_path = os.path.dirname(os.path.realpath(__file__))

class Conventional:
    """
    Class to simulate conventional treatment for a given cancer site.
    Modify the parameters at the bottom of this code to reflect the properties of the cancer site.
    """
    def __init__(self, episodes):
        self.episodes = episodes
        self.alpha_norm_tissue = alpha_norm_tissue
        self.beta_norm_tissue = beta_norm_tissue
        self.alpha_tumor = alpha_tumor
        self.beta_tumor  = beta_tumor
        self.n = n
        self.m = m
        self.TD50 = TD50
        self.average_cancer_glucose_absorption = .54
        self.average_cancer_oxygen_consumption = 20
        
        self.dose_bin_width = 0.5
        
        self.env = GridEnv(reward = 'dose', alpha_norm_tissue=self.alpha_norm_tissue, beta_norm_tissue=self.beta_norm_tissue,
                           alpha_tumor=self.alpha_tumor, beta_tumor=self.beta_tumor, 
                           average_cancer_oxygen_consumption=self.average_cancer_oxygen_consumption, 
                           average_cancer_glucose_absorption=self.average_cancer_glucose_absorption,
                           dose_bin_width=self.dose_bin_width)
        
        results = self.run()
        
        
        with open("Results/rectum_baseline.txt", "w") as f:
            f.write('Performances of the agent on the new environment' + '\n')
            f.write("TCP :" + str(results["TCP"]) + "\n") 
            f.write("Average NTCP (rectal bleeding): " + str(np.mean(results["fractions"])) + " std dev: " + str(np.std(results["fractions"])) + "\n") 
            f.write("Average num of fractions: " + str(np.mean(results["fractions"])) + " std dev: " + str(np.std(results["fractions"])) + "\n") 
            f.write("Average duration: " + str(np.mean(results["duration"])) + " std dev: " + str(np.std(results["duration"])) + "\n") 
            f.write("Average survival: " + str(np.mean(results["survival"])) + " std dev: " +  str(np.std(results["survival"])) + "\n") 
            f.write("Average radiation dose: " + str(np.mean(results["doses"])) + " std dev: " + str(np.std(results["doses"])) + "\n") 
        
    def run(self):
        lengths_arr = []
        fracs_arr = []
        doses_arr = []
        survivals_arr = []
        NTCP_results = []
        
        healthy_arr = [0]
        cancer_arr  = [0]
        
        doses_per_hour = {}
        fracs_per_hour = {}
        rewards = []
        
        episodes = self.episodes
        print(episodes)
        
        sum_w = 0
        for ep in range(episodes):
            self.env.reset()
            sum_r = 0
            fracs = 0
            doses = 0
            time = 0
            doses_per_hour[ep] = {}
            init_hcell = self.env.init_hcell_count
            print(init_hcell)
            while not self.env.inTerminalState():
                state = self.env.convert(self.env.observe())
                action = DOSE
                reward = self.env.act(action)

                print(action + 1, "grays, reward =", reward)
                fracs += 1
                doses += action + 1
                doses_per_hour[ep][time] = action + 1
                time += 24
                sum_r += reward
                next_state = self.env.convert(self.env.observe())
                
                healthy_arr.append(HealthyCell.cell_count)
                cancer_arr.append(CancerCell.cell_count)
                
                
            if self.env.end_type == 'W':
                sum_w += 1
            
            
            fracs_arr.append(fracs)
            doses_arr.append(doses)
            lengths_arr.append(time)
            survival = HealthyCell.cell_count / init_hcell
            survivals_arr.append(survival)
            rewards.append(sum_r)
            
            print("DVH calculation")
            volumes, dose_bins = self.env.grid.calculate_DVH_voxel(name=f"episode_{ep}", mask_OAR=self.env.mask_OAR, mask_PTV=self.env.mask_PTV)
            Deff = calculate_effective_dose(volumes, dose_bins, self.n)
            print("Effective dose : ", Deff)
            print("NTCP calculation")
            NTCP_results.append(Lyman_Kutcher_Burman(Deff, self.TD50, self.m))

            
        self.epochs_arr = np.arange(episodes)
        self.fracs_arr = np.array(fracs_arr)
        self.doses_arr = np.array(doses_arr)
        self.lengths_arr = np.array(lengths_arr)
        self.survivals_arr = np.array(survivals_arr)
        self.rewards = np.array(rewards)
        self.tcp = 100.0 *sum_w/episodes
        self.healthy_arr = np.array(healthy_arr)
        self.cancer_arr = np.array(cancer_arr)

        print("TCP: " , self.tcp)
        print("NTCP", 100*np.mean(NTCP_results))
        print("Average num of fractions: ", np.mean(self.fracs_arr), " std dev: ", np.std(self.fracs_arr))
        print("Average radiation dose: ", np.mean(self.doses_arr), " std dev: ", np.std(self.doses_arr))
        print("Average duration: ", np.mean(self.lengths_arr), " std dev: ", np.std(self.lengths_arr))
        print("Average survival: ", np.mean(self.survivals_arr), " std dev: ", np.std(self.survivals_arr))
        
        results = {"TCP" : self.tcp,
                   "NTCP" : NTCP_results,
                   "fractions" : self.fracs_arr,
                   "doses" : self.doses_arr,
                   "duration" : self.lengths_arr,
                   "survival" : self.survivals_arr,
                   "doses_per_hour" : doses_per_hour,
                   "rewards" : self.rewards,
                   "Max healthy" : np.max(self.healthy_arr),
                   "Max cancer" : np.max(self.cancer_arr),
                   "Healthy array" : self.healthy_arr
                   }
        
        return results
    
    
    
class Agent:
    """
    Class to simulate RL treatment for a given cancer site.
    Modify the parameters at the bottom of this code to reflect the properties of the cancer site.
    """
    def __init__(self, episodes, path, agent_name, agent_type):
        self.episodes = episodes
        self.alpha_norm_tissue = alpha_norm_tissue
        self.beta_norm_tissue = beta_norm_tissue
        self.alpha_tumor = alpha_tumor
        self.beta_tumor  = beta_tumor
        self.n = n
        self.m = m
        self.TD50 = TD50
        self.average_cancer_glucose_absorption = .54
        self.average_cancer_oxygen_consumption = 20
        
        self.dose_bin_width = 0.5
        
        self.env = GridEnv(reward = 'dose', alpha_norm_tissue=self.alpha_norm_tissue, beta_norm_tissue=self.beta_norm_tissue,
                           alpha_tumor=self.alpha_tumor, beta_tumor=self.beta_tumor, 
                           average_cancer_oxygen_consumption=self.average_cancer_oxygen_consumption, 
                           average_cancer_glucose_absorption=self.average_cancer_glucose_absorption,
                           dose_bin_width=self.dose_bin_width)
        
        self.path = path
        self.agent_name = agent_name # Rectum_16
        self.agent_type = agent_type # SARSAgent
        self.load_agent()
        
        results = self.run(episodes)
        
        
        
        with open("RL_treatment.txt", "w") as f:
            f.write('Performances of the agent on the new environment' + '\n')
            f.write("TCP :" + str(results["TCP"]) + "\n") 
            f.write("Average NTCP (rectal bleeding): " + str(np.mean(results["fractions"])) + " std dev: " + str(np.std(results["fractions"])) + "\n") 
            f.write("Average num of fractions: " + str(np.mean(results["fractions"])) + " std dev: " + str(np.std(results["fractions"])) + "\n") 
            f.write("Average duration: " + str(np.mean(results["duration"])) + " std dev: " + str(np.std(results["duration"])) + "\n") 
            f.write("Average survival: " + str(np.mean(results["survival"])) + " std dev: " +  str(np.std(results["survival"])) + "\n") 
            f.write("Average radiation dose: " + str(np.mean(results["doses"])) + " std dev: " + str(np.std(results["doses"])) + "\n") 
        
   
    def load_agent(self):
        load_path = os.path.join(self.path, self.agent_name, f'q_table_{self.agent_name}.npy')
        self.q_table = np.load(load_path, allow_pickle=False)  
        
    def choose_action(self, state):
        actions = np.argwhere(self.q_table[state]==np.max(self.q_table[state])).flatten()
        return np.random.choice(actions)
    
    def run(self, episodes):
        
        lengths_arr = []
        fracs_arr = []
        doses_arr = []
        survivals_arr = []
        NTCP_results = []
        
        healthy_arr = [0]
        cancer_arr  = [0]
        
        doses_per_hour = {}
        fracs_per_hour = {}
        rewards = []
        
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
                action = np.argmax(self.q_table[state])
                reward = self.env.act(action)

                print(action + 1, "grays, reward =", reward)
                fracs += 1
                doses += action + 1
                doses_per_hour[ep][time] = action + 1
                time += 24
                sum_r += reward
                next_state = self.env.convert(self.env.observe())
                
                healthy_arr.append(HealthyCell.cell_count)
                cancer_arr.append(CancerCell.cell_count)
                
                
            if self.env.end_type == 'W':
                sum_w += 1
            
            fracs_arr.append(fracs)
            doses_arr.append(doses)
            lengths_arr.append(time)
            survival = HealthyCell.cell_count / init_hcell
            survivals_arr.append(survival)
            rewards.append(sum_r)
            
            print("DVH calculation")
            volumes, dose_bins = self.env.grid.calculate_DVH_voxel(name=f"episode_{ep}", mask_OAR=self.env.mask_OAR, mask_PTV=self.env.mask_PTV)
            #np.savez(f"DVH/datas/dvh_rl_{ep}.npz", volumes=volumes, dose_bins=dose_bins)
            Deff = calculate_effective_dose(volumes, dose_bins, self.n)
            print("Effective dose : ", Deff)
            print("NTCP calculation")
            NTCP_results.append(Lyman_Kutcher_Burman(Deff, self.TD50, self.m))

            
        self.epochs_arr = np.arange(episodes)
        self.fracs_arr = np.array(fracs_arr)
        self.doses_arr = np.array(doses_arr)
        self.lengths_arr = np.array(lengths_arr)
        self.survivals_arr = np.array(survivals_arr)
        self.rewards = np.array(rewards)
        self.tcp = 100.0 *sum_w/episodes
        self.healthy_arr = np.array(healthy_arr)
        self.cancer_arr = np.array(cancer_arr)

        print("TCP: " , self.tcp)
        print("NTCP", 100*np.mean(NTCP_results))
        print("Average num of fractions: ", np.mean(self.fracs_arr), " std dev: ", np.std(self.fracs_arr))
        print("Average radiation dose: ", np.mean(self.doses_arr), " std dev: ", np.std(self.doses_arr))
        print("Average duration: ", np.mean(self.lengths_arr), " std dev: ", np.std(self.lengths_arr))
        print("Average survival: ", np.mean(self.survivals_arr), " std dev: ", np.std(self.survivals_arr))
        
        results = {"TCP" : self.tcp,
                   "NTCP" : NTCP_results,
                   "fractions" : self.fracs_arr,
                   "doses" : self.doses_arr,
                   "duration" : self.lengths_arr,
                   "survival" : self.survivals_arr,
                   "doses_per_hour" : doses_per_hour,
                   "rewards" : self.rewards,
                   "Max healthy" : np.max(self.healthy_arr),
                   "Max cancer" : np.max(self.cancer_arr),
                   "Healthy array" : self.healthy_arr
                   }
        
        return results
    
    
    
alpha_norm_tissue = 1.24*1e-2
beta_norm_tissue = 4.84*1e-3
alpha_tumor = 0.3113547444918789
beta_tumor  = 0.06528191240051033
n = 0.15
m = 0.15
TD50 = 80.1
DOSE = 0.8
    
agent = Agent(episodes=100, 
              path=dir_path, 
              agent_name="Rectum_16", 
              agent_type="SARSAgent")