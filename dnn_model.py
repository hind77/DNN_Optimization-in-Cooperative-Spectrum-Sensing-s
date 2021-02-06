#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 17:05:48 2021

@author: mac
"""
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras import backend as K
from tensorflow.keras.layers import BatchNormalization
from keras.layers import LeakyReLU
from global_vars import*
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras import layers
from keras import models
from keras import regularizers
from statistics import mean 
import matplotlib.pyplot as plt

class DNNModel:
    
    @staticmethod
    def get_model():
        
        """ 
        This function returns sequential model and the model initilizer
      
        """
        model = Sequential()
        model.add(BatchNormalization(input_shape=(num_sens,)))
        initializer = tf.keras.initializers.GlorotUniform(seed=(1))
        return model, initializer     

    @staticmethod
    def choose_model(init_model, choice: str):
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
                model.add(Dense(num_sens, activation='linear',name='last_layer'))
                model._name = 'Linear_Model'              
                
        return model
    @classmethod
    def train_model(cls, model, X_train: np.ndarray, X_val: np.ndarray,
              y_train: np.ndarray, y_val: np.ndarray) -> np.ndarray:
        """
        This function trains the model. The number of epochs and 
        batch_size are set by the constants in the parameters section.
        
        Parameters:
            model : model with the chosen architecture
            X_train : training features
            y_train : training target
            X_valid : validation features
            Y_valid : validation target
            
        Output:
            model training history    
        
        """
        history = []
        opt = tf.keras.optimizers.Adam(learning_rate=0.001)
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=50) 
        model.compile(loss = cls.loss_fn , optimizer=opt)
        history = model.fit(X_train,y_train,batch_size=batch_s,epochs=200,validation_data=(X_val,y_val),callbacks=[callback])
        for layer in model.layers:
         print(layer.output_shape)
    
        return history
    
    @staticmethod
    def loss_fn(snrs: np.ndarray, predicted) -> float:
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
    def eval_metric(model, history: np.ndarray,
                    metric_name: str) -> np.ndarray:
        '''
        Function to evaluate a trained model on a chosen metric. 
        Training and validation metric are plotted in a
        line chart for each epoch.
        
        Parameters:
            history : model training history
            metric_name : loss or accuracy
        Output:
            line chart with epochs of x-axis and metric on
            y-axis
        '''    
        metric = history.history[metric_name]
        val_metric = history.history['val_' + metric_name] 
        plt.figure(2)
        ax = plt.axes()
        ax.plot(metric, linestyle= 'dotted', color='blue',label='Train Loss')
        ax.plot(val_metric, linestyle='dotted', color='red',label='Val Loss')
        ax.set_title('Loss ' + model.name)
        ax.margins(x=0,y=0)
        ax.grid(False)
        plt.legend()
        ax.set_xlabel('Epoch number')
        ax.set_ylabel(metric_name)
        #plt.show()  
        plt.savefig('Loss ' + model.name +'.pdf')
        plt.close()
    
    @staticmethod    
    def compare_models(model_1, model_2, 
                       model_3, 
                       history_1: np.ndarray,
                       history_2: np.ndarray, 
                       history_3: np.ndarray, 
                      
                       metric: str):
    
        '''
        Function to compare a metric between two models 
        
        Parameters:
            history_1 : training history of model 1
            history_2 : training history of model 2
            metric : metric to compare, loss, acc, val_loss or val_acc
            
        Output:
            plot of metrics of both models
        '''    
        metric_1 = history_1.history[metric]
        metric_2 = history_2.history[metric]
        metric_3 = history_3.history[metric]
        #metric_4 = history_4.history[metric]
        
        metrics_dict = {
            'acc' : 'Training Accuracy',
            'loss' : 'Training Loss',
            'val_acc' : 'Validation accuracy',
            'val_loss' : 'Validation loss'
        } 
        
        metric_label = metrics_dict[metric]
    
        plt.figure(3)
        ax = plt.axes()    
        ax.plot(metric_1, linestyle= 'solid', color='orange', label=model_1.name)
        ax.plot(metric_2, linestyle= 'solid', color='green', label=model_2.name)
        ax.plot(metric_3, linestyle= 'solid', color='blue', label = model_3.name)
        #ax.plot(metric_4, linestyle= 'solid', color='black', label = model_4.name)
        ax.margins(x=0,y=0)
        plt.xlabel('Epoch number')
        plt.ylabel(metric_label)
        plt.title('Comparing ' + metric_label + ' between models')
        plt.legend()
        #plt.show()
        plt.savefig('Comparing ' + metric_label + ' between models' +'.pdf')
        
    