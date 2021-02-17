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

from spectrum_sensing import SpectrumSensing

def main():
    
    with open('trainData.data', 'rb') as filehandle:
        snrs_train = np.asarray(pickle.load(filehandle))
        snrs_train = snrs_train.reshape(num_samples,10)
    
    
    signal_power = np.array(ChannelModel.ch_gen(num_samples)['power'])
    snrs_test = np.array(ChannelModel.ch_gen(num_samples)['snrs'])
    X_train, X_val, y_train, y_val = train_test_split(snrs_train, snrs_train, test_size=0.30)
    
    # the model for the first problem(p1) that provided weights to maximize the deflection coef
    # model_dropout_P1 = DNNModelP1.choose_model_P1(DNNModelP1.get_model, "dropout")
    # model_R1_P1 = DNNModelP1.choose_model_P1(DNNModelP1.get_model, "dropout_Rl1")
    # model_R2_P1 = DNNModelP1.choose_model_P1(DNNModelP1.get_model, "dropout_Rl2")

    # history_dropout_P1 = DNNModelP1.train_model(model_dropout_P1, X_train, X_val, y_train, y_val, DNNModelP1.loss_fn_P1)
    # history_R1_P1 = DNNModelP1.train_model(model_R1_P1, X_train, X_val, y_train, y_val, DNNModelP1.loss_fn_P1)
    # history_R2_P1 = DNNModelP1.train_model(model_R2_P1, X_train, X_val, y_train, y_val, DNNModelP1.loss_fn_P1) 

    # DNNModelP1.eval_metric(model_dropout_P1, history_dropout_P1, "loss")
    # DNNModelP1.eval_metric(model_R1_P1, history_R1_P1, "loss")
    # DNNModelP1.eval_metric(model_R2_P1, history_R2_P1, "loss")

    # DNNModelP1.compare_models(model_dropout_P1, model_R1_P1, model_R2_P1,  history_dropout_P1, history_R1_P1, history_R2_P1, "val_loss")
    
    # the model for the second problem(p2) to get the optimal threshold to minimize the error function
    
    model_dropout_P2 = DNNModelP2.choose_model_P2(DNNModelP2.get_model, "dropout")
    model_R1_P2 = DNNModelP2.choose_model_P2(DNNModelP2.get_model, "dropout_Rl1")
    model_R2_P2 = DNNModelP2.choose_model_P2(DNNModelP2.get_model, "dropout_Rl2")

    history_dropout_P2 = DNNModelP2.train_model(model_dropout_P2, X_train, X_val, y_train, y_val, DNNModelP2.loss_fn_P2)
    history_R1_P2 = DNNModelP2.train_model(model_R1_P2, X_train, X_val, y_train, y_val, DNNModelP2.loss_fn_P2)
    history_R2_P2 = DNNModelP2.train_model(model_R2_P2, X_train, X_val, y_train, y_val, DNNModelP2.loss_fn_P2)

    DNNModelP2.eval_metric(model_dropout_P2, history_dropout_P2, "loss")
    DNNModelP2.eval_metric(model_R1_P2, history_R1_P2, "loss")
    DNNModelP2.eval_metric(model_R2_P2, history_R2_P2, "loss")

    DNNModelP2.compare_models(model_dropout_P2, model_R1_P2, model_R2_P2,  history_dropout_P2, history_R1_P2, history_R2_P2, "val_loss")
        
    # spectrum_sensing = SpectrumSensing()
    
    # spectrum_sensing.generate_weights(model_R2, snrs_test, signal_power)


   
    
if __name__ == '__main__': 
    main()
