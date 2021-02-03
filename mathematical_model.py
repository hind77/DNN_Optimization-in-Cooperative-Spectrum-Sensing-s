#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: hind
"""
import numpy as np
from global_vars import *
import math 

class MathematicalModel:
    
    def old_paper_mathematical_weights(self,snrs)-> np.ndarray:
        
        '''
        Function to compute the weights from the mathematical 
        formula in 2007 paper
        
        Parameters:
            
            snrs: the row of  the sensing units snrs 
            
        Output:
            mathematical weights
        '''        
        
        snrs = snrs.transpose()       
        identity = np.identity(num_sens)
        diagonal = np.diag(snrs)
        D = np.sqrt((num_samples*identity)+diagonal)
        D_inv = np.linalg.inv(D)
        mat = D_inv*snrs*snrs.transpose()*D_inv
        v, vects = np.linalg.eig(mat)
        maxcol = list(v).index(max(v))
        q_zero = vects[:,maxcol]        
        norm = np.linalg.norm(np.dot( D_inv,q_zero), ord=2)        
        w1 = np.dot( D_inv,q_zero)/norm
        w1 = self.real(w1)
        w1 = np.array(w1)     
        return w1
    
    def new_paper_mathematical_weights(self, snrs: np.ndarray, signal_power: np.ndarray)-> np.ndarray:
    
        '''
        Function to compute the weights from the mathematical formula in 2016 paper
        
        Parameters:
            
            snrs:  the row of  the sensing units snrs 
            signak_power: the power of the signal received by the sensing units
            
        Output:
            mathematical weights
        '''      
        
        C_inv = np.linalg.pinv(signal_power)
        return np.dot(C_inv,snrs)
    
    def compute_deflection_coef(self, weights: np.ndarray, snrs: np.ndarray)-> float:
        '''
        Function to compute the deflection coef from the mathematical 
        formula in 2007 paper
        
        Parameters:
            
            snrs: the row of  the sensing units snrs 
            weights: weights array 
            
        Output:
            mathematical weights
        '''        
                
        
        snrs = snrs.transpose()
        snrs = snrs.reshape(10,1)
        weights = weights.reshape(10,1)
        t = np.dot(snrs.transpose(),weights)**2
        t1 = np.dot(4*weights.transpose(),
                    (num_samples*np.identity(num_sens)+np.diag(snrs)))
        b = np.dot(t1,weights)       
        dm_square = t/b
        
        return dm_square
    
    def real(self,w1):
        return w1.real

    def convert(self,lst): 
        return [[el] for el in lst]