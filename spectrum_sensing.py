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
import scipy.spatial.distance
import matplotlib.pyplot as plt
from texttable import Texttable

class SpectrumSensing:
    
    def define_users_snrs(self, snrs)-> dict:
        
        '''
        Function to return a dictionary that representes each user with its signal
        
          Parameters: 
              secondaru users snrs
          Outputs:
              users snrs dictionary 
        
        '''    
        signals = dict()
        for i in range(0,num_sens):
            signals[i]= snrs[i]            
        return signals


    def get_local_decisions(self, static,thresh,decisions)-> dict:
        
      '''
        Function to return a dictionary that representes each user with its decisions
        
          Prameters:
              static: the statistic test here we consider the snrs
              thresh: threshold associated to SUs
              decisions: the binary decisions provided by each SU
         Output:
             local decision for each secondary user
              
        
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
        
         Prameters:
              static: the statistic test here we consider the snrs
              thresh: threshold associated to SUs
              weights: SUs'weights
         Output:
             the cooperative decision
        
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
    
        
    def compute_euclidian_distance(self, numerical_weights, dnn_weights):
        
        '''
            return the euclidian distance between two arrays 
            
          '''  
        distance = []
        for i in range(0,len(numerical_weights)):
             dnn_weights = dnn_weights + (i / 10000.)
             distance.append(scipy.spatial.distance.euclidean(numerical_weights.flatten(), dnn_weights.flatten()))
        
        plt.figure(4)
        ax = plt.axes()    
        ax.plot(distance, linestyle= 'solid', color='orange', label="euclidian distance")
        ax.margins(x=0,y=0)
        plt.xlabel('')
        plt.ylabel('distance')
        plt.title('euclidian distance: (dnn vs numerical) weights')
        plt.legend()
        plt.show()
        
        
    def benshmarking_table(self,dnn_coef, math_coef, numerical_coef):
        table = Texttable()
        table.add_rows([['Deflection_coef', 'Value'], 
                    ['dnn_coef', dnn_coef], ['numerical_coef', numerical_coef], 
                    ['mathematical_coef',math_coef]])
        print(table.draw())
     
        
    def generate_weights(self,model, snrs_test,
                       signal_power) -> np.ndarray :
        
      '''
        Function to returns different weights (dnn/numerical/mathematical)
        
      '''      
        
      for k in range(0,rounds):#Number of Monte Carlo Simulations 
            print("\n")
            print("******** Results for Round {} ********".format(k))
    
            weights = model.predict(snrs_test)
            dnn_weights = weights[k]
            numerical_weights = MathematicalModel.compute_weights_using_deflection_coef(snrs_test[k])
            maths_weights_2007 = MathematicalModel.old_paper_mathematical_weights(snrs_test[k])
            maths_weights_2016 = MathematicalModel.new_paper_mathematical_weights(snrs_test[k])
            
            # print("dnn weights", dnn_weights)        
            # print("numerical weights", numerical_weights)        
            # print("mathematical weights", maths_weights_2007)
            DNN_dm_square = MathematicalModel.compute_deflection_coef(dnn_weights,
                                                    snrs_test[k])
            mathematical_dm_square_2007 = MathematicalModel.compute_deflection_coef(maths_weights_2007, 
                                                             snrs_test[k])
            mathematical_dm_square_2016 = MathematicalModel.compute_deflection_coef(maths_weights_2016, 
                                                             snrs_test[k])            
            numerical_dm_square = MathematicalModel.compute_deflection_coef(numerical_weights,
                                                                            snrs_test[k])
            
            #self.benshmarking_table(DNN_dm_square, mathematical_dm_square, numerical_dm_square)
            print("DNN deflection coef",DNN_dm_square)
            print("numerical deflection coef",numerical_dm_square)
            print("mathematical deflection coef",mathematical_dm_square_2007)
            print("mathematical deflection coef",mathematical_dm_square_2016)
     
            
    def generate_thresholds(self, snrs_test, model_d) -> np.ndarray :
        thresholds_d = model_d.predict(snrs_test)
        dnn_thresholds_d = thresholds_d[0]

        numerical_thresholds = MathematicalModel.compute_numerical_thresholds(snrs_test[0])
        print("DNN thresholds for dropeout model ",dnn_thresholds_d)
        print("numerical thresholds",numerical_thresholds)        