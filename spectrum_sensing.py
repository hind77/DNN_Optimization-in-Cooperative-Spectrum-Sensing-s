#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 17:35:56 2021

@author: mac
"""

from global_vars import *
import math 
from scipy import special as sp
from mathematical_model import MathematicalModel

class SpectrumSensing:
    
    def define_users_snrs(self, snrs)-> dict:
        '''
        Function to return a dictionary that representes each user with its signal
        
        '''    
        signals = dict()
        for i in range(0,num_sens):
            signals[i]= snrs[i]
            
        return signals




    def get_local_decisions(self, static,thresh,decisions)-> dict:
        
      '''
        Function to return a dictionary that representes each user with its decisions
        
      '''       
      for k,v in static.items():
        if v > thresh:
          decisions[k].append(1)
        else:
          decisions[k].append(0)
      #print("loop decisioon",decisions)
      return decisions

    def get_cooperative_decision(self, static,thresh,weights)-> np.ndarray:
      '''
        Function to return a cooperative decision
        
      '''  
      cooperative_static = list()   
      for k,v in static.items():
          
          cooperative_static.append(weights[k]*v)
          
      return int(sum(cooperative_static)> thresh)
  
      def get_cooperatieve_pd(self, cooperative_decisions)-> np.ndarray:
        
        return sum(cooperative_decisions)/rounds
     
      

      def local_pd(self, decisions: dict ,pds: np.ndarray)-> np.ndarray:
        
              
          '''
            Function to return a dictionary that representes each user with its local probability of detection
            
          '''  
          pd = dict()
          for k,v in decisions.items():
            pd[k] = sum(v)/rounds
            pds[k].append(sum(v)/rounds)
          
          return pd
      
        
    def Compute_PdVSPf(self,model, snrs_test,
                       signal_power) -> np.ndarray :
        
      '''
        Function to return the cooperative probability of detection 
        
      '''  
      mathematical_model = MathematicalModel()
      for m in range(0,len(pf)):
        local_decisions = {k: [] for k in range(num_sens)}
        cooperative_decisions = list()
        
        for k in range(0,rounds):#Number of Monte Carlo Simulations   
            snrs = define_users_snrs(snrs_test[k])
            val = 1-2*pf[m]
            thresh[m] = ((math.sqrt(2)*sp.erfinv(val))/ math.sqrt(num_samples))+1
            weights = model.predict(snrs_test)
            weights = weights[k]
            old_maths_weights = mathematical_model.old_paper_mathematical_weights(snrs_test[k]) 
            print("mathematical weights", old_maths_weights)
            DNN_dm_square = mathematical_model.compute_deflection_coef(weights,
                                                    snrs_test[k])
            print("DNN deflection coef",DNN_dm_square)
            mathematical_dm_square = mathematical_model.compute_deflection_coef(old_maths_weights, 
                                                             snrs_test[k])
            print("mathematical deflection coef",mathematical_dm_square)        
            local_decisions = self.get_local_decisions(snrs,thresh[m],
                                                  local_decisions)
            cooperative_decisions.append(self.get_cooperative_decision(snrs,
                                                                  thresh[m],
                                                                  weights))
      
        local_pd(local_decisions,local_pds)
        cooperative_pds.append(get_cooperatieve_pd(cooperative_decisions))
        
      return cooperative_pds
