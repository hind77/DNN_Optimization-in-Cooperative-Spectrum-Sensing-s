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
      
     
      dim=batch_s*num_sens
      #snrs_np = np.reshape(snrs, [dim])
      predicted=tf.reshape(predicted,[dim])
      
      snrs_tf = tf.reshape(snrs , [dim])
      
      print("predicted shape", predicted.shape)
      print("snrs shape",snrs_tf.shape)
      C = tf.linalg.tensor_diag(1+2*snrs_tf)
      C_inv = tf.linalg.inv(C)
      exp1 = tf.math.multiply(predicted,tf.norm(tf.math.multiply(C_inv,snrs_tf), ord=1, axis=None))-tf.math.multiply(tf.math.multiply(fs*(T_cte - num_sens*tr),tf.keras.backend.transpose(snrs_tf)),tf.keras.backend.transpose(C_inv)) 
      exp2 = tf.linalg.norm(C*snrs_tf, ord=1)* math.sqrt(2*fs*(T_cte - num_sens*tr))
      exp3 = fs*(T_cte - num_sens*tr)*tf.keras.backend.transpose(snrs_tf)*tf.keras.backend.transpose(C_inv)- predicted*tf.linalg.norm(C_inv*snrs_tf, ord=1)
      exp4 = math.sqrt(2*fs*(T_cte - num_sens*tr))*tf.math.sqrt(tf.keras.backend.transpose(snrs_tf)*tf.keras.backend.transpose(C_inv)*snrs_tf)  
      zero_tensor = tf.zeros((640,))
      tf.debugging.assert_non_negative(exp4, message="there is a negative value in the tensor")   
      tf.debugging.assert_none_equal(snrs_tf, zero_tensor, message="there is a null value here in snrs_tf")
      tf.debugging.assert_none_equal(tf.keras.backend.transpose(C_inv), zero_tensor, message="there is a null value here in transpose of C_inv")
      tf.debugging.assert_none_equal(exp4, zero_tensor, message="there is a null value here in exp4")

      p_0 = tf.math.multiply(pi_0,cls.Q(exp1/exp2))
      p_1 = tf.math.multiply(pi_1,cls.Q(exp3/exp4))
      
      #p_1 = tf.math.multiply(pi_1,cls.Q(exp3/exp4))       
      # loss = (p_0 + p_1)
      #loss = (exp1/exp2)+(exp3/exp4)
      # dim=batch_s*num_sens
      # snrs = np.reshape(snrs, [dim])
      # predicted = np.reshape(snrs,[dim])
      # C = np.diag(1+2*snrs)
      # C_inv = np.linalg.inv(C)
      # exp1 = np.dot(predicted, np.linalg.norm(np.dot( C_inv,snrs), ord=1))-np.dot(np.dot(fs*(T_cte - num_sens*tr), snrs.transpose()), C.transpose())
      # exp2 = np.dot(np.linalg.norm(np.dot( C,snrs), ord=1),math.sqrt(2*fs*(T_cte - num_sens*tr)))
      # exp3 = np.dot(np.dot((fs*(T_cte - num_sens)),snrs.transpose()), C_inv.transpose())- np.dot(predicted, np.linalg.norm(np.dot( C_inv,snrs), ord=1))
      # exp4 = np.dot(np.dot(np.dot((2*fs*(T_cte - num_sens*tr)), snrs.transpose()), C_inv.transpose()), snrs)
      #p_0 = pi_0*cls.Q(exp1/exp2)
      #p_1 = pi_1*cls.Q(exp3/exp4)     
      loss = p_0 + p_1
      print(loss)
      
      return loss / batch_s
  
    @staticmethod  
    def Q(x):
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
                model.add(LeakyReLU(alpha=0.05))
                model.add(layers.Dropout(0.2))
                #model.add(BatchNormalization())
                model.add(Dense(64))
                model.add(LeakyReLU(alpha=0.05))
                model.add(layers.Dropout(0.2))
                #model.add(BatchNormalization())
                model.add(Dense(32))
                model.add(LeakyReLU(alpha=0.05))
                model.add(layers.Dropout(0.2))
                #model.add(BatchNormalization())
                model.add(Dense(16))
                model.add(LeakyReLU(alpha=0.05))
                model.add(layers.Dropout(0.2))
                #model.add(BatchNormalization())
                model.add(Dense(num_sens, activation='relu',name='last_layer'))
                model._name = 'dropout'
                
        if choice == "dropout_Rl1":           
                model.add(Dense(num_sens, kernel_initializer=initializer))
                model.add(LeakyReLU(alpha=0.001))
                model.add(layers.Dropout(0.2))
                #model.add(BatchNormalization())
                model.add(Dense(64, kernel_regularizer=regularizers.l1(0.0005)))
                model.add(LeakyReLU(alpha=0.001))
                model.add(layers.Dropout(0.2))
                #model.add(BatchNormalization())
                model.add(Dense(32, kernel_regularizer=regularizers.l1(0.0005)))
                model.add(LeakyReLU(alpha=0.001))
                model.add(layers.Dropout(0.2))
                #model.add(BatchNormalization())
                model.add(Dense(16, kernel_regularizer=regularizers.l1(0.0005)))
                model.add(LeakyReLU(alpha=0.001))
                model.add(layers.Dropout(0.2))
                #model.add(BatchNormalization())
                model.add(Dense(num_sens, activation='relu',name='last_layer'))
                model._name = 'Dropout_Model_Rl1'
     
        if choice == "dropout_Rl2":           
                model.add(Dense(num_sens, kernel_initializer=initializer))
                model.add(LeakyReLU(alpha=0.001))
                model.add(layers.Dropout(0.2))
                #model.add(BatchNormalization())
                model.add(Dense(64,kernel_regularizer=regularizers.l2(0.0005)))
                model.add(LeakyReLU(alpha=0.001))
                model.add(layers.Dropout(0.2))
                #model.add(BatchNormalization())
                model.add(Dense(32,kernel_regularizer=regularizers.l2(0.0005)))
                model.add(LeakyReLU(alpha=0.001))
                model.add(layers.Dropout(0.2))
                #model.add(BatchNormalization())
                model.add(Dense(16,kernel_regularizer=regularizers.l2(0.0005)))
                model.add(LeakyReLU(alpha=0.001))
                model.add(layers.Dropout(0.2))
                #model.add(BatchNormalization())
                model.add(Dense(num_sens, activation='relu',name='last_layer'))
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