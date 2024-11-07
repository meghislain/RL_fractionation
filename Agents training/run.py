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
parser.add_argument("--alpha_tumor", type = float)
parser.add_argument("--beta_tumor", type = float)

#alpha/beta normal/healthy tissues
parser.add_argument("--alpha_norm", type = float)
parser.add_argument("--beta_norm", type = float)

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

agent.run(n_epochs=args.epochs[0], 
          train_steps=args.epochs[1], 
          test_steps=args.epochs[2], 
          final_epsilon=args.final_epsilon, 
          final_alpha=args.final_alpha, 
          name="agent_retrained"
          )


