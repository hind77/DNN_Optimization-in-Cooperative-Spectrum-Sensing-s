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
from dnn_model import DNNModel
from spectrum_sensing import SpectrumSensing

def main():
    
    with open('trainData.data', 'rb') as filehandle:
        snrs_train = np.asarray(pickle.load(filehandle))
        snrs_train = snrs_train.reshape(num_samples,10)
    
   
    signal_power = np.array(ChannelModel.ch_gen(num_samples)['power'])
    snrs_test = np.array(ChannelModel.ch_gen(num_samples)['snrs'])
    X_train, X_val, y_train, y_val = train_test_split(snrs_train, snrs_train, test_size=0.30)
    

    model_dropout = DNNModel.choose_model(DNNModel.get_model, "dropout")
    model_R1 = DNNModel.choose_model(DNNModel.get_model, "dropout_Rl1")
    model_R2 = DNNModel.choose_model(DNNModel.get_model, "dropout_Rl2")

    history_dropout = DNNModel.train_model(model_dropout, X_train, X_val, y_train, y_val)
    history_R1 = DNNModel.train_model(model_R1, X_train, X_val, y_train, y_val)
    history_R2 = DNNModel.train_model(model_R2, X_train, X_val, y_train, y_val)

    DNNModel.eval_metric(model_dropout, history_dropout, "loss")
    DNNModel.eval_metric(model_R1, history_R1, "loss")
    DNNModel.eval_metric(model_R2, history_R2, "loss")

    DNNModel.compare_models(model_dropout, model_R1, model_R2,  history_dropout, history_R1, history_R2, "val_loss")
    
    spectrum_sensing = SpectrumSensing()
    
    spectrum_sensing.generate_weights(model_R2, snrs_test, signal_power)


   
    
if __name__ == '__main__': 
    main()
