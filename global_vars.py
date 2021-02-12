#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 17:00:57 2021

@author: mac
"""

import numpy as np
from scipy import special as sp
import math

size_area=200.0 # size of area sensors are distributed 

pu_active_prob = 1.0

pl_const = 34.5 # use pathloss constant   (wslee paper)
pl_alpha = 38.0 # use pathloss constant   (wslee paper)

d_ref = 50.0 # reference distance 
sh_sigma = 7.9 # shadow fading constant   (wslee paper)

p_t_dB = 23.0 # tx power - 23dBm
p_t = 10*(p_t_dB/10)

sigma_v = 1 # noice variance


batch_s = 64
num_sens = 10 
samples_factor = 100
num_samples = batch_s*samples_factor

sen_loc = size_area*(np.random.rand(num_sens, 2)-0.5)
pri_loc = size_area*(np.random.rand(1, 2)-0.5) #placing sensing entities and primary user randomly

pf = 0.01
pd = np.arange(0, 1, 0.05)# probability of detection
val = 1-2*pf
thresh = ((math.sqrt(2)*sp.erfinv(val))/ math.sqrt(num_samples))+1

rounds = 10

n=50

local_pds = {k: [] for k in range(num_sens)}
cooperative_pds = list()

pi_0 = 0.3
pi_1 = 0.7
fs = 100*1000
tr = 0.2*pow(10,-3)
T_cte = (n/fs)+num_sens*tr
