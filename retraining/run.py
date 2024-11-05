# -*- coding: utf-8 -*-

import os
import argparse
from simulation import SARSAgent
from environment import GridEnv


parser = argparse.ArgumentParser(description='2D Env. with Tabular Agent')
parser.add_argument("--epochs", type = int, help="Number of epochs, train steps, test steps", nargs = 3, default = [15, 1000, 100])
parser.add_argument("--gamma", type = float, help="Discount Factor", default = 0.95)
parser.add_argument("--alpha", type = float, help="Learning Rate", default = 0.8)
parser.add_argument("--epsilon", type = float, help="epsilon greedy policy argument", default = 0.8)
parser.add_argument("--final_alpha", type = float, help="Final value of learning rate", default = 0.5)
parser.add_argument("--final_epsilon", type = float, help="Final value of epsilon", default = 0.05)

# alpha/beta tumor
parser.add_argument("--alpha_tumor", type = float, default = 0.465)
parser.add_argument("--beta_tumor", type = float, default = 0.103)

args = parser.parse_args()
print(args)


env = GridEnv(alpha_norm_tissue=0.124, 
              beta_norm_tissue=0.0484,
              alpha_tumor=args.alpha_tumor, 
              beta_tumor=args.beta_tumor
              )

agent = SARSAgent(env=env, gamma=args.gamma, alpha=args.alpha, epsilon=args.epsilon)

dir_path = os.path.dirname(os.path.realpath(__file__))
path = dir_path + "/Rectum_16/q_table_Rectum_16.npy"
agent.load(path)

agent.run(args.epochs[0], args.epochs[1], args.epochs[2], args.final_epsilon, args.final_alpha, "Rectum_retrained2")


