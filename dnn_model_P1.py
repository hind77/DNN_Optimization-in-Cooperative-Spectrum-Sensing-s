#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 12:14:20 2021

@author: mac
"""

from global_vars import *
import tensorflow as tf
from dnn_model import DNNModel
from keras.layers import Dense
from tensorflow.keras import backend as K
from tensorflow.keras.layers import BatchNormalization
from keras.layers import LeakyReLU
from keras import layers
from keras import models
from keras import regularizers

class DNNModelP1(DNNModel):
    
    @staticmethod
    def loss_fn_P1(snrs: np.ndarray, predicted) -> float:
      """ 
        This is a customized loss function
        
        Parameters:
            SNRs : the signal noise ratio
            predicted: the output weights of the model
    
        Output:
            loss value   
          
      """      
    
      dim=batch_s*num_sens      
      t = tf.reduce_sum(tf.math.multiply(snrs,predicted))**2
      snrs = tf.reshape(snrs , [dim])
      predicted=tf.reshape(predicted,[dim])
      identity = tf.eye(dim,dtype=tf.float32)
      diagonal = tf.linalg.tensor_diag(snrs)
      b_1 = 4*tf.keras.backend.transpose(predicted)
      b_2 = n*identity+diagonal
      b_3 = predicted
      b=tf.linalg.matvec(b_2,b_3)
      b=tf.reduce_sum(tf.math.multiply(b_1,b))
      loss= -t/b
      return loss / batch_s
  
    

    @staticmethod
    def choose_model_P1(init_model, choice: str):
        """ 
        This function defines the model architecture and returns the model
      
        """   
        model, initializer = init_model()
        
        if choice == "dropout":            
                model.add(Dense(num_sens, kernel_initializer=initializer))
                model.add(LeakyReLU(alpha=0.05))
                model.add(layers.Dropout(0.2))
                model.add(BatchNormalization())
                model.add(Dense(64))
                model.add(LeakyReLU(alpha=0.05))
                model.add(layers.Dropout(0.2))
                model.add(BatchNormalization())
                model.add(Dense(32))
                model.add(LeakyReLU(alpha=0.05))
                model.add(layers.Dropout(0.2))
                model.add(BatchNormalization())
                model.add(Dense(16))
                model.add(LeakyReLU(alpha=0.05))
                model.add(layers.Dropout(0.2))
                model.add(BatchNormalization())
                model.add(Dense(num_sens, activation='softmax',name='last_layer'))
                model._name = 'dropout'
                
        if choice == "dropout_Rl1":           
                model.add(Dense(num_sens, kernel_initializer=initializer))
                model.add(LeakyReLU(alpha=0.001))
                model.add(layers.Dropout(0.2))
                model.add(BatchNormalization())
                model.add(Dense(64, kernel_regularizer=regularizers.l1(0.0005)))
                model.add(LeakyReLU(alpha=0.001))
                model.add(layers.Dropout(0.2))
                model.add(BatchNormalization())
                model.add(Dense(32, kernel_regularizer=regularizers.l1(0.0005)))
                model.add(LeakyReLU(alpha=0.001))
                model.add(layers.Dropout(0.2))
                model.add(BatchNormalization())
                model.add(Dense(16, kernel_regularizer=regularizers.l1(0.0005)))
                model.add(LeakyReLU(alpha=0.001))
                model.add(layers.Dropout(0.2))
                model.add(BatchNormalization())
                model.add(Dense(num_sens, activation='softmax',name='last_layer'))
                model._name = 'Dropout_Model_Rl1'
     
        if choice == "dropout_Rl2":           
                model.add(Dense(num_sens, kernel_initializer=initializer))
                model.add(LeakyReLU(alpha=0.001))
                model.add(layers.Dropout(0.2))
                model.add(BatchNormalization())
                model.add(Dense(64,kernel_regularizer=regularizers.l2(0.0005)))
                model.add(LeakyReLU(alpha=0.001))
                model.add(layers.Dropout(0.2))
                model.add(BatchNormalization())
                model.add(Dense(32,kernel_regularizer=regularizers.l2(0.0005)))
                model.add(LeakyReLU(alpha=0.001))
                model.add(layers.Dropout(0.2))
                model.add(BatchNormalization())
                model.add(Dense(16,kernel_regularizer=regularizers.l2(0.0005)))
                model.add(LeakyReLU(alpha=0.001))
                model.add(layers.Dropout(0.2))
                model.add(BatchNormalization())
                model.add(Dense(num_sens, activation='softmax',name='last_layer'))
                model._name = 'Dropout_Model_Rl2'            
                
        if choice == "linear":           
                model.add(Dense(num_sens, kernel_initializer=initializer))
                model.add(LeakyReLU(alpha=0.01))
                model.add(layers.Dropout(0.2))
                model.add(BatchNormalization())
                model.add(Dense(64))
                model.add(LeakyReLU(alpha=0.01))
                model.add(layers.Dropout(0.2))
                model.add(BatchNormalization())
                model.add(Dense(32))
                model.add(LeakyReLU(alpha=0.01))
                model.add(layers.Dropout(0.2))
                model.add(BatchNormalization())
                model.add(Dense(16))
                model.add(LeakyReLU(alpha=0.01))
                model.add(layers.Dropout(0.2))
                model.add(BatchNormalization())
                model.add(Dense(num_sens, activation='linear',
                                name='last_layer'))
                model._name = 'Linear_Model'              
                
        return model