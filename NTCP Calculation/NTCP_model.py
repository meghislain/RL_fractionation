# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 09:23:50 2023

@author: Florian Martin
"""

import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd

def calculate_effective_dose(volumes, doses, n):
    
    # Calculating the gEUD
    gEUD = sum(v * (d**(1/n)) for v, d in zip(volumes, doses))
    gEUD = gEUD**n
    
    return gEUD


def Lyman_Kutcher_Burman(gEUD, TD50, m, dx=0.01):
    
    #Lyman-Kutcher-Burman Model
    # uses gEUD in LKB model
    
    t=(gEUD-TD50)/(m*TD50)
    print(f"Integral from -999 to {t}")
    num_range=np.arange(-999,t,dx)
    sum_ntcp=0.
    for dummy in range(len(num_range)):
        sum_ntcp+=np.exp(-1*num_range[dummy]**2/2)*dx
        
    return 1./np.sqrt(2*np.pi)*sum_ntcp


