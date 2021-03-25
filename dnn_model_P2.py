#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 12:17:39 2021

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


class DNNModelP2(DNNModel):
    
    @classmethod
    def loss_fn_P2(cls,snrs: np.ndarray, predicted) -> float:
      """ 
        This is a customized loss function
        
        Parameters:
            SNRs : the signal noise ratio
            predicted: the output threshold of the model
    
        Output:
            loss value            
      """  
      # proto_tensor = tf.make_tensor_proto(predicted)
      # predicted = tf.make_ndarray(proto_tensor)
      # print(predicted)
      print("this is the snrs shape:", snrs.shape)
      print("this is the predicted shape:", predicted.shape)
      dim=batch_s * num_sens
     
      predicted = predicted * tf.ones((1,num_sens))
      predicted=tf.reshape(predicted,[dim])
      predicted = tf.math.abs(predicted)
      print("this is the predicted after reshape", predicted.shape)
      #snrs_tf = tf.slice(snrs,[0,0,0],dim)
      snrs_tf = tf.reshape(snrs , [dim])
      snrs_tf = tf.math.abs(snrs_tf)
      print("this is snrs after reshape", snrs_tf.shape)
      tf.debugging.assert_non_negative(snrs_tf, message="there is a negative value in the tensor snrs_tf")
      C = tf.linalg.tensor_diag(1+2*snrs_tf)
      C =tf.math.abs(C)
      C_inv = tf.linalg.inv(C)
      C_inv =tf.math.abs(C_inv)
      zero_tensor = tf.zeros((dim,dim))
      epsilon = 0.05# to avoid the division by 0 
      epsilon_tensor = tf.fill((dim, dim), epsilon)
      tf.debugging.assert_none_equal(epsilon_tensor, zero_tensor, message="there is a null value here in epsilon tensor")
      exp1 = tf.math.multiply(predicted,tf.norm(tf.math.multiply(C_inv,snrs_tf), ord=1, axis=None))-tf.math.multiply(tf.math.multiply(fs*(T_cte - num_sens*tr),tf.keras.backend.transpose(snrs_tf)),tf.keras.backend.transpose(C_inv)) 
      exp2 = tf.linalg.norm(C*snrs_tf, ord=1)* math.sqrt(2*fs*(T_cte - num_sens*tr))
      exp3 = fs*(T_cte - num_sens*tr)*tf.keras.backend.transpose(snrs_tf)*tf.keras.backend.transpose(C_inv)- predicted*tf.linalg.norm(C_inv*snrs_tf, ord=1)
      ext = tf.math.sqrt((tf.keras.backend.transpose(snrs_tf)+epsilon_tensor)*(tf.keras.backend.transpose(C_inv)+epsilon_tensor)*snrs_tf) 
      exp4 = math.sqrt(2*fs*(T_cte - num_sens*tr))* ext  
      test = tf.keras.backend.transpose(C_inv)+epsilon_tensor
      tf.debugging.assert_none_equal(test, zero_tensor, message="there is a null value here in test tensor")
      tf.debugging.assert_non_negative(tf.keras.backend.transpose(snrs_tf), message="there is a negative value in the tf.keras.backend.transpose(snrs_tf)")
      tf.debugging.assert_non_negative(tf.keras.backend.transpose(C_inv), message="there is a negative value in the tf.keras.backend.transpose(C_inv)")
      tf.debugging.assert_non_negative(snrs_tf, message="there is a negative value in the snrs_tf")
      tf.debugging.assert_non_negative(ext, message="there is a negative value in the ext")
      tf.debugging.assert_non_negative(exp4, message="there is a negative value in the tensor")      
      tf.debugging.assert_none_equal(snrs_tf, zero_tensor, message="there is a null value here in snrs_tf")
      #tf.debugging.assert_none_equal(tf.keras.backend.transpose(C_inv), zero_tensor, message="there is a null value here in transpose of C_inv")
      tf.debugging.assert_none_equal(exp4, zero_tensor, message="there is a null value here in exp4")
      p_0 = tf.math.multiply(pi_0,cls.Q(exp1/exp2))
      p_1 = tf.math.multiply(pi_1,cls.Q(exp3/exp4)) 
      loss = p_0 + p_1
     
      return loss / batch_s
  
    @staticmethod  
    def Q(x):
        """
        the Q-function
        """
        value = tf.cast(x, tf.float32)
        return 0.5-0.5*tf.math.erf(tf.cast(value/ math.sqrt(2),tf.float32))

    
    @staticmethod
    def choose_model_P2(init_model, choice: str):
        """ 
        This function defines the model architecture and returns the model
      
        """   
        model, initializer = init_model()
        
        if choice == "dropout":            
                model.add(Dense(num_sens, kernel_initializer=initializer))
                model.add(LeakyReLU(alpha=0.005))
                model.add(layers.Dropout(0.3))
                #model.add(BatchNormalization())
                model.add(Dense(64))
                model.add(LeakyReLU(alpha=0.005))
                model.add(layers.Dropout(0.3))
                #model.add(BatchNormalization())
                model.add(Dense(32))
                model.add(LeakyReLU(alpha=0.005))
                model.add(layers.Dropout(0.3))
                #model.add(BatchNormalization())
                model.add(Dense(16))
                model.add(LeakyReLU(alpha=0.005))
                
                model.add(layers.Dropout(0.3))
                #model.add(BatchNormalization())
                model.add(Dense(1, name='last_layer'))
                model.add(LeakyReLU(alpha=0.005))
                
                model._name = 'dropout'
                
        if choice == "dropout_Rl1":           
                model.add(Dense(num_sens, activation='relu', kernel_initializer=initializer))
                model.add(LeakyReLU(alpha=0.01))
                #model.add(layers.Dropout(0.2))
                #model.add(BatchNormalization())
                model.add(Dense(64))
                model.add(LeakyReLU(alpha=0.01))
                #model.add(layers.Dropout(0.2))
                #model.add(BatchNormalization())
                model.add(Dense(32))
                model.add(LeakyReLU(alpha=0.01))
                #model.add(layers.Dropout(0.2))
                #model.add(BatchNormalization())
                model.add(Dense(16))
                model.add(LeakyReLU(alpha=0.01))
                
                #model.add(layers.Dropout(0.2))
                #model.add(BatchNormalization())
                model.add(Dense(1,name='last_layer'))
                model.add(LeakyReLU(alpha=0.01))
                model._name = 'dropout'
                model._name = 'Standard'
     
        if choice == "dropout_Rl2":           
                model.add(Dense(num_sens,activation='relu', kernel_initializer=initializer))
                #model.add(LeakyReLU(alpha=0.005))
                model.add(layers.Dropout(0.1))
                #model.add(BatchNormalization())
                model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.0005)))
                #model.add(LeakyReLU(alpha=0.005))
                model.add(layers.Dropout(0.1))
                #model.add(BatchNormalization())
                model.add(Dense(32,activation='relu', kernel_regularizer=regularizers.l2(0.0005)))
                #model.add(LeakyReLU(alpha=0.005))
                model.add(layers.Dropout(0.1))
                #model.add(BatchNormalization())
                model.add(Dense(16,activation='relu',kernel_regularizer=regularizers.l2(0.0005)))
                #model.add(LeakyReLU(alpha=0.005))
                model.add(layers.Dropout(0.1))
                #model.add(BatchNormalization())
                model.add(Dense(1, activation='relu',name='last_layer'))
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