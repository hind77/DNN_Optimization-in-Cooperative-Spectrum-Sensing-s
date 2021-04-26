#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 13:01:41 2020

@author: mac
"""

import numpy as np
from sklearn.model_selection import train_test_split
import pickle
from global_vars import *
from channel_model import ChannelModel
from dnn_model_P1 import DNNModelP1
from dnn_model_P2 import DNNModelP2
from sklearn.preprocessing import normalize
from spectrum_sensing import SpectrumSensing
import time
import psutil
from decimal import *
from memory_profiler import profile
import os
from sklearn import preprocessing


def main():
    
    spectrum_sensing = SpectrumSensing()
    #with open('trainData.data', 'rb') as filehandle:
        #snrs_train = np.asarray(pickle.load(filehandle))
        #snrs_train = snrs_train.reshape(num_samples,10)
    #to stock data
    # snrs_train = np.array(ChannelModel.ch_gen(num_samples)['snrs'])
    
    # with open('trainData.data', 'wb') as filehandle: 
    #     pickle.dump(snrs_train, filehandle)
    
    output =ChannelModel.ch_gen(num_samples)
    #print("this is the output", output)
    snrs_train = np.array(output['snrs'])


    signal_gain= np.array(output['gain'])

    snrs_test = np.array(output['snrs'])


    #print("snrs train", snrs_train)
    X_train_P1, X_val_P1, y_train_P1, y_val_P1 = train_test_split(snrs_train, snrs_train, test_size=0.30)
    
    #the model for the first problem(p1) that provided weights to maximize the deflection coef
    problem1_id = 'P1'
    

    model_R2_P1 = DNNModelP1.choose_model_P1(DNNModelP1.get_model, "dropout_Rl2")
    

    history_R2_P1 = DNNModelP1.train_model(model_R2_P1, X_train_P1, X_val_P1, y_train_P1, y_val_P1, DNNModelP1.loss_fn_P1) 


    #DNNModelP1.eval_metric(model_R2_P1, history_R2_P1, "loss", problem1_id)

    
    #the model for the first problem(p2) that provided thresholds to minimize the probability of error
    problem2_id = 'P2'
    
    #train data scaling
    # scaler_train = preprocessing.StandardScaler().fit(snrs_train)
    # snrs_train_P2 = scaler_train.transform(snrs_train)
    # snrs_train_P2 = np.absolute(snrs_train)
    
    #test data scaling 
    # scaler_test = preprocessing.StandardScaler().fit(snrs_test)
    # snrs_test_P2 = scaler_test.transform(snrs_test)
    # snrs_test_P2 = np.absolute(snrs_test)
    
    # X_train_P2, X_val_P2, y_train_P2, y_val_P2 = train_test_split(snrs_train_P2, snrs_train_P2, test_size=0.30)

    # model_dropout_P2 = DNNModelP2.choose_model_P2(DNNModelP2.get_model, "dropout")


    # history_dropout_P2 = DNNModelP2.train_model(model_dropout_P2, X_train_P2, X_val_P2, y_train_P2, y_val_P2, DNNModelP2.loss_fn_P2_with_gain(gain=signal_gain))

    
    # DNNModelP2.eval_metric(model_dropout_P2, history_dropout_P2, "loss", problem2_id)
    start_time = time.time()
    
  
    #spectrum_sensing.generate_dnn_weights(model_R2_P1, snrs_test)
    spectrum_sensing.generate_numerical_weights(snrs_test)
    #spectrum_sensing.generate_thresholds(snrs_test, model_dropout_P2)
    print("--- %s seconds ---" % (time.time() - start_time)) 
    


def execute_main():    
    main()
   
    
if __name__ == '__main__': 
    
    
    
    execute_main()
    #print("--- %s seconds ---" % (time.time() - start_time))    
    #print ("The value of CPU usage (using %) is : ",end="")
    #print ('%.10f'%psutil.cpu_percent())
    # print ("The value of RAM usage (using %) is : ",end="")
    # print ('%.5f'%psutil.virtual_memory())
    
    
    
