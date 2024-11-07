# Reinforcement Learning for Radiotherapy treatments optimization

This repository contains two main projects:

1. **Main Simulation Code**: The primary codebase for simulations, executed using `run.py`.
2. **NTCP Calculation**: A specialized module for calculating the Normal Tissue Complication Probability (NTCP), found in the `NTCP Calculation` folder.

## Repository Structure

```plaintext
.
├── NTCP Calculation/
│   └── conventional_treatments.py   # Code for NTCP calculation
├── Rectum_16/                       # Trained agents on rectum cancer site with default parameters
├── cell.py                          # Module for cell objects
├── environment.py                   # Module for environment setup
├── grid.py                          # Module for grid dynamic
├── run.py                           # Main entry point for the simulation
├── simulation.py                    # Main simulation logic for RL agents training/retraining
└── README.md                        # Documentation

