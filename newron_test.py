#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 13:43:34 2021

@author: mac
"""
from channel_model import *
from global_vars import *
from scipy import optimize
from scipy import special
import random
from scipy.optimize import minimize, rosen, rosen_der
from numpy.linalg import multi_dot
from decimal import Decimal


#Initialization
snrs_train = np.array(ChannelModel.ch_gen(num_samples)['snrs'])
snrs = snrs_train[0]
print(snrs)




def Q(x):
    """
    the Q-function
    """
    
    return 0.5-0.5*special.erf(x/ math.sqrt(2))



def func(x, *args):
    
    """
    The optimization function 
    """
   
    epsilon = 0.05 # to avoid the division by 0 
    h = np.ones(num_sens)# I set h for ones to simplify the computation
    h = h.reshape(10,1)
    print("h shape", h.shape)
    bold_one = np.identity(num_sens)
    C = np.diag(1+2*snrs) # the C value from the formula diag{[1 +2snrs]}
    print("this is C shape", C.shape)
    C_inv = np.linalg.inv(C) # C_inv
    print("this is C_inv shape", C_inv.shape)
    snrs_s = snrs.reshape(10,1) # I reshape the snrs for matricial multiplication 
    exp1 = multi_dot([snrs_s.transpose(), C_inv.transpose(),h])
    print("exp1 shape",exp1.shape)
    exp2 = np.linalg.norm(np.dot( C_inv,snrs_s), ord=1)
    print("exp2 shape",exp2.shape)
    identity_vector = np.ones(num_sens)
    identity_vector = identity_vector.reshape(10,1)
    add = identity_vector+h
    print("add shape",add.shape)
   
    exp3 = num_sens* multi_dot([snrs_s.transpose(), C_inv.transpose(),add])- (x*np.linalg.norm(np.dot( C_inv,snrs_s), ord=1))
    print("exp3 shape",exp3.shape)
    exp4 = np.sqrt(multi_dot([snrs_s.transpose(), C_inv.transpose(), snrs_s])+epsilon)
    t1 = np.divide(exp1, exp2)
    print("t1 shape", t1.shape)
    t2 = np.divide(exp3, exp4)
    print("t2 shape", t2.shape)
    #exp4 = np.sqrt(snrs_s.transpose()*C_inv.transpose()*snrs_s)+epsilon
    print("exp4 shape",exp4.shape)
    p_0 = pi_0*Q(x-(exp1/exp2))
    p_1 = pi_1*Q(exp3/exp4)
    f = p_0 + p_1
    print(x.shape)
    print("this is f", f[0][0])

    
    
    return f[0][0]

if __name__ == '__main__':

    bnds = ((0, 1),) 
    random_guess = random.choice(snrs)
    #print("this is the random guess", random_guess)
    #random_guess = tuple(list(snrs)) 
    print("this is the random guess 2", random_guess)
    constraint = ({'type': 'ineq', 'fun': lambda x:  x>0})
    res = minimize(func, random_guess,args=(snrs), method='SLSQP', bounds=bnds, constraints=constraint)
    print(res.x)
    print(snrs)
    