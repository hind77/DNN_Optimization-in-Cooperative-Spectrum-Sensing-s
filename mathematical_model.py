#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: hind
"""
import numpy as np
from global_vars import *
import math 
from scipy.optimize import minimize, rosen, rosen_der
from numpy import linalg as LA
import itertools
from numpy.linalg import multi_dot
from scipy import special
import random

class MathematicalModel:
    
    @classmethod
    def old_paper_mathematical_weights(cls,snrs)-> np.ndarray:
        
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
        w1 = cls.real(w1)
        w1 = np.array(w1)     
        return w1
    
    
    @classmethod
    def new_paper_mathematical_weights(cls,snrs: np.ndarray)-> np.ndarray:
    
        '''
        Function to compute the weights from the mathematical formula in 2016 paper
        
        Parameters:
            
            snrs:  the row of  the sensing units snrs 
            signak_power: the power of the signal received by the sensing units
            
        Output:
            mathematical weights
        '''      
        C = np.diag(1+2*snrs)
        C_inv = np.linalg.pinv(C)
        norm = np.linalg.norm(np.dot(C_inv,snrs), ord=2)  
        w_opt = np.dot( C_inv, snrs)/norm
        w_opt = cls.real(w_opt)
        w_opt = np.array(w_opt)   
        return w_opt
    
    
    @staticmethod
    def compute_deflection_coef(weights: np.ndarray, snrs: np.ndarray)-> float:
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
                    (n*np.identity(num_sens)+np.diag(snrs)))
        b = np.dot(t1,weights)       
        dm_square = t/b
        
        return dm_square
    
    
    @staticmethod
    def compute_weights_using_deflection_coef(snrs) -> np.ndarray:
        '''
        This function is the numerical computation of the weights using the minimization function
        
         Parameters:
             snrs: secondary users snrs
         output:
             the optimal weights
        
        '''
        #fun = lambda x: ((x[i]*nu[i] for i in range(0,len(nu)))**2)/(4*((n+nu[i])*x[i]**2 for i in range(0,len(nu))))
        n = 50
        nu = snrs
        fun = lambda x: - ((x[0]*nu[0] + x[1]*nu[1]+x[2]*nu[2]+x[3]*nu[3]+x[4]*nu[4]+x[5]*nu[5]+x[6]*nu[6]+x[7]*nu[7]+x[8]*nu[8]+x[9]*nu[9])**2)/(4*((n+nu[0])*x[0]**2+(n+nu[1])*x[1]**2+(n+nu[2])*x[2]**2+(n+nu[3])*x[3]**2+(n+nu[4])*x[4]**2+(n+nu[5])*x[5]**2+(n+nu[6])*x[6]**2+(n+nu[7])*x[7]**2+(n+nu[8])*x[8]**2+(n+nu[9])*x[9]**2))       
        #fun = lambda x: (sum_s(nu, x))/mul(nu, x)        
        constraint = ({'type': 'eq', 'fun': lambda x:  LA.norm(x,1)-1})
        bnds = ((0, 1),) * len(snrs)
        random_guess = np.random.dirichlet(np.ones(10)*1000.,size=1)
        random_guess = tuple(list(random_guess[0]))
        res = minimize(fun, random_guess, method='SLSQP', bounds=bnds, constraints=constraint)
        return(res.x)
    
    
    @staticmethod
    def Q(x):
        """
        the Q-function
        """
        
        return 0.5-0.5*special.erf(x/ math.sqrt(2))
    
    
    @classmethod
    def func(cls,x, *args):
    
        """
        The optimization function from the 2016 paper
        
        """
        snrs=np.array(args[0])

        epsilon = 0.05 # to avoid the division by 0 
        h = np.ones(num_sens)# I set h for ones to simplify the computation
        h = h.reshape(10,1)# I reshape the snrs for matricial multiplication 
        bold_one = np.identity(num_sens)
        C = np.diag(1+2*snrs) # the C value from the formula diag{[1 +2snrs]}
        C_inv = np.linalg.inv(C) # C_inv
        snrs_s = snrs.reshape(10,1) 
        exp1 = multi_dot([snrs_s.transpose(), C_inv.transpose(),h])
        exp2 = np.linalg.norm(np.dot( C_inv,snrs_s), ord=1)
        identity_vector = np.ones(num_sens)
        identity_vector = identity_vector.reshape(10,1)
        add = identity_vector+h       
        exp3 = num_sens* multi_dot([snrs_s.transpose(), C_inv.transpose(),add])- (x*np.linalg.norm(np.dot( C_inv,snrs_s), ord=1))
        exp4 = np.sqrt(multi_dot([snrs_s.transpose(), C_inv.transpose(), snrs_s])+epsilon)
        t1 = np.divide(exp1, exp2)
        t2 = np.divide(exp3, exp4)
        p_0 = pi_0*cls.Q(x-(exp1/exp2))
        p_1 = pi_1*cls.Q(exp3/exp4)
        f = p_0 + p_1
        return f[0][0]
    
    
    @classmethod
    def compute_numerical_thresholds(cls,snrs):
        
        '''
        This function compute the numerical optimal thresholds using minimization
        
          Parameters:
              snrs of the secondary users
          Outputs:
              numerical optimal thresholds
        '''
        bnds = ((0, 1),)  
        random_guess = random.choice(snrs)
        constraint = ({'type': 'ineq', 'fun': lambda x:  x>0})
        res = minimize(cls.func, random_guess,args=(snrs), method='SLSQP', bounds=bnds, constraints=constraint)
        return res.x 
     
    
    @staticmethod
    def real(w1):
        return w1.real
    

    
 