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
parser.add_argument("--alpha_tumor", type = float, help="Alpha value for tumoral cells")
parser.add_argument("--beta_tumor", type = float, help="Beta value for tumoral cells")
parser.add_argument("--alpha_norm", type = float, help="Alpha value for healthy cells")
parser.add_argument("--beta_norm", type = float, help="Beta value for healthy cells")
parser.add_argument("--load_path", type = str, help="Path for loading trained agent (optional)", default=None)

args = parser.parse_args()
print(args)


env = GridEnv(alpha_norm_tissue=args.alpha_norm, 
              beta_norm_tissue=args.beta_norm,
              alpha_tumor=args.alpha_tumor, 
              beta_tumor=args.beta_tumor
              )

agent = SARSAgent(env=env, 
                  gamma=args.gamma, 
                  alpha=args.alpha, 
                  epsilon=args.epsilon
                  )

dir_path = os.path.dirname(os.path.realpath(__file__))

if args.load_path is not None:
    path = dir_path + "/Rectum_16/q_table_Rectum_16.npy"
    agent.load(path)

agent.run(n_epochs=args.epochs[0], 
          train_steps=args.epochs[1], 
          test_steps=args.epochs[2], 
          final_epsilon=args.final_epsilon, 
          final_alpha=args.final_alpha, 
          name="agent_retrained"
          )


