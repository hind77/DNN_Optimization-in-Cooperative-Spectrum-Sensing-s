#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 17:00:57 2021

@author: mac
"""

import numpy as np

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

pf = np.arange(0, 1, 0.05)# probability of false alarm 
pd = np.arange(0, 1, 0.05)# probability of detection
thresh = [None] * len(pf) # the threshold

rounds = 4

n=500

local_pds = {k: [] for k in range(num_sens)}
cooperative_pds = list()