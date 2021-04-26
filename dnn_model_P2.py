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
    def loss_fn_P2_with_gain(cls,gain):
        
        global batch_id
        dim=batch_s * num_sens
        gain= gain[0:batch_id,:]
        gain= tf.cast(gain, tf.float32)
        gain_bis = tf.reshape(gain,[dim])
        
        def loss_fn_P2(snrs: np.ndarray, predicted) -> float:
          dim=batch_s * num_sens
          lent = batch_s * num_sens
          index = 0
          count = 0
          predicted_bis = predicted * tf.ones((1,num_sens))
          predicted_bis=tf.reshape(predicted_bis,[dim])
          snrs_tf_bis = tf.reshape(snrs , [dim])
          snrs_tf_bis = tf.math.abs(snrs_tf_bis)
          loss_values = list()
          i = 0
          while count < lent:      
              snrs_tf = tf.slice(snrs_tf_bis,[index],[index+num_sens])
              predicted = tf.slice(predicted_bis,[index],[index+num_sens])
              gain = tf.slice(gain_bis,[index],[index+num_sens])
              count = count + num_sens
              
              tf.debugging.assert_non_negative(snrs_tf, message="there is a negative value in the tensor snrs_tf")
              C = tf.linalg.tensor_diag(1+2*snrs_tf)
              #C =tf.math.abs(C)
              C_inv = tf.linalg.inv(C)
              #C_inv =tf.math.abs(C_inv)
              snrs_tf=tf.reshape(snrs_tf,[num_sens,1])
              predicted=tf.reshape(predicted,[1,num_sens])

              exp1 = predicted*tf.norm(tf.linalg.matmul(C_inv,snrs_tf), ord=1, axis=None)-tf.linalg.matmul(tf.linalg.matmul(fs*(T_cte - num_sens*tr)*snrs_tf,C_inv,transpose_a=True,transpose_b=True),tf.ones([num_sens,1]))
              exp2 = tf.norm(tf.linalg.matmul(C,snrs_tf), ord=1)* math.sqrt(2*fs*(T_cte - num_sens*tr))
              p_0 = pi_0*cls.Q(exp1/exp2)
              
              exp3 = fs*(T_cte - num_sens*tr)*tf.linalg.matmul(snrs_tf,tf.keras.backend.transpose(C_inv),transpose_a=True)*(0.3*gain+tf.ones((num_sens,)))- predicted*tf.linalg.norm(C_inv*snrs_tf, ord=1)
              ext = tf.math.sqrt(tf.linalg.matmul(tf.linalg.matmul(snrs_tf,tf.keras.backend.transpose(C_inv),transpose_a=True),snrs_tf)) 
              exp4 = math.sqrt(2*fs*(T_cte - num_sens*tr))* ext
              
              p_1 = pi_1*cls.Q(exp3/exp4)
              loss=p_0+p_1
              loss_values.append(loss) 
          return tf.reduce_mean(loss_values)
        return loss_fn_P2
  
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
                model.add(Dense(num_sens, kernel_initializer=initializer, kernel_regularizer=regularizers.l1(0.003)))
                model.add(LeakyReLU(alpha=0.05))
                model.add(layers.Dropout(0.4))
                model.add(BatchNormalization())
                model.add(Dense(6, kernel_initializer=initializer, kernel_regularizer=regularizers.l1(0.003)))
                model.add(LeakyReLU(alpha=0.005))
                model.add(layers.Dropout(0.4))
                model.add(BatchNormalization())
                model.add(Dense(6, kernel_initializer=initializer, kernel_regularizer=regularizers.l1(0.003)))
                model.add(LeakyReLU(alpha=0.005))
                model.add(layers.Dropout(0.4))
                model.add(BatchNormalization())
                model.add(Dense(6, kernel_initializer=initializer, kernel_regularizer=regularizers.l1(0.003)))
                model.add(LeakyReLU(alpha=0.005))
                model.add(layers.Dropout(0.4))
                model.add(BatchNormalization())
                model.add(Dense(1, name='last_layer', kernel_initializer=initializer, kernel_regularizer=regularizers.l1(0.003),activation='softplus'))
                #model.add(LeakyReLU(alpha=0.005))
                model._name = 'dropout'
                
        return model


        if choice == "dropout1":            
                model.add(Dense(num_sens, kernel_initializer=initializer))
                model.add(LeakyReLU(alpha=0.05))
                model.add(layers.Dropout(0.3))
                model.add(BatchNormalization())
                model.add(Dense(64, kernel_initializer=initializer))
                model.add(LeakyReLU(alpha=0.005))
                model.add(layers.Dropout(0.3))
                model.add(BatchNormalization())
                model.add(Dense(32, kernel_initializer=initializer))
                model.add(LeakyReLU(alpha=0.005))
                model.add(layers.Dropout(0.3))
                model.add(BatchNormalization())
                model.add(Dense(16, kernel_initializer=initializer))
                model.add(LeakyReLU(alpha=0.005))
                model.add(layers.Dropout(0.3))
                model.add(BatchNormalization())
                model.add(Dense(1, name='last_layer', kernel_initializer=initializer,activation='softplus'))
                #model.add(LeakyReLU(alpha=0.005))
                model._name = 'dropout'  
                
        return model 