#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 13:01:41 2020

@author: mac
"""

import numpy as np
import random
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras import backend as K
import pandas as pd 
import tensorflow as tf
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import normalize
import seaborn as sns
import math 
from scipy import special as sp
from tensorflow.keras.layers import BatchNormalization
from keras.layers import LeakyReLU
import matplotlib.pyplot as plt
from keras import layers
from keras import models
from keras import regularizers
from statistics import mean 


#----------------------------------------------------------------------------------------

size_area=200.0 # size of area sensors are distributed 

pu_active_prob = 1.0

pl_const = 34.5 # use pathloss constant   (wslee paper)
pl_alpha = 38.0 # use pathloss constant   (wslee paper)

d_ref = 50.0 # reference distance 
sh_sigma = 7.9 # shadow fading constant   (wslee paper)

p_t_dB = 23.0 # tx power - 23dBm
p_t = 10*(p_t_dB/10)

sigma_v = 1 # noice variance


batch_s = 64
num_sens = 10
samples_factor = 10
num_samples = batch_s*samples_factor

sen_loc = size_area*(np.random.rand(num_sens, 2)-0.5)
pri_loc = size_area*(np.random.rand(1, 2)-0.5) #placing sensing entities and primary user randomly

pf = np.arange(0, 1, 0.05)# probability of false alarm 
pd = np.arange(0, 1, 0.05)# probability of detection
thresh = [None] * len(pf) # the threshold

rounds = 4

n=500

local_pds = {k: [] for k in range(num_sens)}
cooperative_pds = list()
   

#-----------------------------------------------------------------------------------------

def get_distances() -> np.ndarray:
    
    '''
    this function generate the random distribution of secondary users and the primary user
    
    '''    
    
    dist_pr_su_vec = pri_loc.reshape(1, 2) - sen_loc # generate PU-SU distance_vector
    dist_pr_su_vec = np.maximum(dist_pr_su_vec, 0.1)
    dist_pr_su_vec = np.linalg.norm(dist_pr_su_vec, axis=1)
    dist_su_su_vec = sen_loc.reshape(num_sens, 1, 2) - sen_loc # generate SU-SU distance_vector
    dist_su_su_vec = np.linalg.norm(dist_su_su_vec, axis=2)
    return dist_pr_su_vec, dist_su_su_vec


def get_channel_gain(dist_pr_su_vec: np.ndarray) -> np.ndarray:
    
    '''
    this function generates the channel gain of each secondary user using the distance
    
    Parameters:
     dist_pr_su_vec : distance between secondary users and the primary user

    Output:
        channel gain
    
    '''  
    
    pu_ch_gain_db = - pl_const - pl_alpha * np.log10(dist_pr_su_vec) # primary channel gain
    return 10 ** (pu_ch_gain_db / 10)


def get_secondary_correlation(dist_su_su_vec: np.ndarray ) -> np.ndarray:
    '''
    this function computes the secondary users correlation using SU-SU distances
    
    Parameters:
     dist_su_su_vec : distances between the secondary users

    Output:
        secondary users correlation
    '''    
    return np.exp(-dist_su_su_vec / d_ref)


def get_shadowing(su_cor: np.ndarray, num_sens: int) -> np.ndarray:
    '''
    this function computes the shadowing using SU-SU correlation 
    
    Parameters:
     num_sens : number of sensing units
     su_cor : the correlation between the secondary users

    Output:
        shadowing    
    
    ''' 
    shadowing_dB = sh_sigma * np.random.multivariate_normal(np.zeros([num_sens]), su_cor)
    return 10 ** (shadowing_dB / 10)


def get_multiPath_Fading(num_sens: int) -> np.ndarray:
    '''
    this function computes the multipath fading 
    
    Parameters:
     num_sens : number of sensing units

    Output:
        multipath fading 
    '''     
    multi_fading = 0.5 * np.random.randn(num_sens) ** 2 + 0.5 * np.random.randn(num_sens) ** 2
    return multi_fading ** 0.5


def ch_gen(num_samples: int) -> np.ndarray:
  """ 
  This function deploy the channel model.
   Parameters:
        num_samples : number of samples

    Output:
       signal noise ratio
  
  """

  returned_list = []
  returned_SNRs = []
  
  
  for i in range(num_samples):

    dist_pr_su_vec, dist_su_su_vec = get_distances()
    pu_ch_gain = get_channel_gain(dist_pr_su_vec)
    su_cor = get_secondary_correlation(dist_su_su_vec)
    shadowing = get_shadowing(su_cor,num_sens)
    pu_power = np.zeros([len(su_cor)]) #pu_power (received power initialization)
    pri_power = p_t #pri_power (transmitted power)
    # test the activity of the primary user 
    if (np.random.rand() < pu_active_prob):
      pu_ch_gain_tot = pu_ch_gain  * shadowing
      pu_power = pu_power +  pri_power*pu_ch_gain_tot
      SNR = pri_power * pow(abs(pu_ch_gain_tot),2)/ sigma_v
    multi_fading = get_multiPath_Fading(num_sens)
    pu_power = pu_power * multi_fading
    returned_list.append(pu_power)
    returned_SNRs.append(SNR)

  return returned_SNRs


def choose_model(init_model, choice):
    """ 
    This function defines the model architecture and returns the model
  
    """   
    model, initializer = init_model()
    
    if choice == "standard":            
            model.add(Dense(num_sens, kernel_initializer=initializer))
            model.add(LeakyReLU(alpha=0.01))
            
            model.add(BatchNormalization())
            model.add(Dense(64))
            model.add(LeakyReLU(alpha=0.01))
            
            model.add(BatchNormalization())
            model.add(Dense(32))
            model.add(LeakyReLU(alpha=0.01))
            
            model.add(BatchNormalization())
            model.add(Dense(16))
            model.add(LeakyReLU(alpha=0.01))
            
            model.add(BatchNormalization())
            model.add(Dense(num_sens, activation='softmax',name='last_layer'))
            model._name = 'dropout'
            
    if choice == "dropout_Rl1":           
            model.add(Dense(num_sens, kernel_initializer=initializer))
            model.add(LeakyReLU(alpha=0.01))
            model.add(layers.Dropout(0.3))
            model.add(BatchNormalization())
            model.add(Dense(64, kernel_regularizer=regularizers.l1(0.0005)))
            model.add(LeakyReLU(alpha=0.01))
            model.add(layers.Dropout(0.3))
            model.add(BatchNormalization())
            model.add(Dense(32, kernel_regularizer=regularizers.l1(0.0005)))
            model.add(LeakyReLU(alpha=0.01))
            model.add(layers.Dropout(0.3))
            model.add(BatchNormalization())
            model.add(Dense(16, kernel_regularizer=regularizers.l1(0.0005)))
            model.add(LeakyReLU(alpha=0.01))
            model.add(layers.Dropout(0.3))
            model.add(BatchNormalization())
            model.add(Dense(num_sens, activation='softmax',name='last_layer'))
            model._name = 'Dropout_Model_Rl1'
 
    if choice == "dropout_Rl2":           
            model.add(Dense(kernel_initializer=initializer))
            model.add(LeakyReLU(alpha=0.01))
            model.add(layers.Dropout(0.3))
            model.add(BatchNormalization())
            model.add(Dense(64, kernel_regularizer=regularizers.l2(0.0005)))
            model.add(LeakyReLU(alpha=0.01))
            model.add(layers.Dropout(0.3))
            model.add(BatchNormalization())
            model.add(Dense(32, kernel_regularizer=regularizers.l2(0.0005)))
            model.add(LeakyReLU(alpha=0.01))
            model.add(layers.Dropout(0.3))
            model.add(BatchNormalization())
            model.add(Dense(16, kernel_regularizer=regularizers.l2(0.0005)))
            model.add(LeakyReLU(alpha=0.01))
            model.add(layers.Dropout(0.3))
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
        

def get_model():
    """ 
    This function returns sequential model and the model initilizer
  
    """
    model = Sequential()
    model.add(BatchNormalization(input_shape=(num_sens,)))
    initializer = tf.keras.initializers.GlorotUniform()
    return model, initializer          
    



def train_model(model, X_train: np.ndarray, X_val: np.ndarray,
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
    opt = tf.keras.optimizers.Adam(learning_rate=0.0003)
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3) 
    model.compile(loss = loss_fn , optimizer=opt)
    history = model.fit(X_train,y_train,batch_size=batch_s,epochs=20,validation_data=(X_val,y_val),callbacks=[callback])
    for layer in model.layers:
     print(layer.output_shape)

    return history
    

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
    
    
def compare_models(model_1, model_2, 
                   model_3, history_1: np.ndarray,
                   history_2: np.ndarray, history_3: np.ndarray, 
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
    


def column(matrix, i):
    return [row[i] for row in matrix]


def define_users_signals(snrs):
    '''
    Function to return a dictionary that representes each user with its signal
    
    '''    
    signals = dict()
    for i in range(0,num_sens):
        signals[i]= column(snrs, i)
        
    return signals

def generate_energy(signals):

  '''
    Function to return a dictionary that representes each user with its Energy
    
  '''       
    
  n = np.random.randn(1,num_samples)#AWGN noise with mean 0 and variance 1
  energy = dict()
  for k,v in signals.items():
    y = list()
    y = v + n
    energy[k] = pow(abs(y),2)
  return energy

def generate_statistic_test(energy):
    
  '''
    Function to return a dictionary that representes each user with its statistic test
    
  '''       
  static = dict()
  for k,v in energy.items():
    static[k] = np.sum(v)*(1/num_samples)
  return static
def get_local_decisions(static,thresh,decisions):
    
  '''
    Function to return a dictionary that representes each user with its decisions
    
  '''       
  for k,v in static.items():
    if v > thresh:
      decisions[k].append(1)
    else:
      decisions[k].append(0)
  #print("loop decisioon",decisions)
  return decisions

def get_cooperative_decision(static,thresh,weights):
  '''
    Function to return a cooperative decision
    
  '''  
  cooperative_static = list()   
  for k,v in static.items():
      
      cooperative_static.append(weights[k]*v)
      
  return int(sum(cooperative_static)> thresh)
 
      
      
def get_cooperatieve_pd(cooperative_decisions):
    
    return sum(cooperative_decisions)/rounds
     
      
          
    

def local_pd(decisions,pds):

      
  '''
    Function to return a dictionary that representes each user with its local probability of detection
    
  '''  
  pd = dict()
  for k,v in decisions.items():
    pd[k] = sum(v)/rounds
    pds[k].append(sum(v)/rounds)
  
  return pd

def get_weights(predictions):
    weights = list()
    for i in range(0,len(predictions)):
        weights.append(abs(mean(predictions[i])))
    return weights

def real(w1):
  return w1.real

def convert(lst): 
    return [[el] for el in lst]

def Compute_PdVSPf(model):
  '''
    Function to return the cooperative probability of detection 
    
  '''  
  for m in range(0,len(pf)):
    local_decisions = {k: [] for k in range(num_sens)}
    cooperative_decisions = list()
    for k in range(0,rounds):#Number of Monte Carlo Simulations
    
        snrs = np.array(ch_gen(num_samples))
        snrs_trick = np.zeros((num_samples,num_sens))
        snrs_trick[0,:] = snrs[k,:]
        signals = define_users_signals(snrs)
        energy = generate_energy(signals)#Energy of received signal over L samples
        val = 1-2*pf[m]
        thresh[m] = ((math.sqrt(2)*sp.erfinv(val))/ math.sqrt(num_samples))+1
        weights = model.predict(snrs_trick).transpose()
        print("this is the weights shape", weights.shape)
        print("weights matrix sum = ", weights[0].sum())
        weights = get_weights(weights)
        weights = [float(i)/sum(weights) for i in weights]
        print('weights',weights)
        print('weights sum',sum(weights))                        
        Statistic_test = generate_statistic_test(energy)
        local_decisions = get_local_decisions(Statistic_test,thresh[m],local_decisions)
        cooperative_decisions.append(get_cooperative_decision(Statistic_test,thresh[m],weights))
            #print(decisions)    
    local_pd(local_decisions,local_pds)
    cooperative_pds.append(get_cooperatieve_pd(cooperative_decisions))
    
  return cooperative_pds

def weights_from_mathematical_model(snrs):
    identity = np.identity(snrs.shape[1])
    print("identity shape", identity.shape)
    diagonal = np.diag(snrs)
    print("diagonal shape", diagonal.shape)
    D = np.sqrt((num_samples*identity)+diagonal)
    print("D shape", D.shape)
    D_inv = np.linalg.inv(D)
    print("D_inv shape", D_inv.shape)
    t1 = np.dot(D_inv,snrs)
    t2 = np.dot(snrs.transpose(),D_inv)
    mat = np.dot(t1,t2)
    v, vects = np.linalg.eig(mat)
    maxcol = list(v).index(max(v))
    q_zero = vects[:,maxcol]
    norm = np.linalg.norm(D_inv*q_zero, ord=2)
    w1 = np.diag(D_inv*q_zero/norm)
    w1_zero = np.sign(snrs.transpose()*w1)*w1
    w1 = real(w1)
    w1 = convert(w1)
    w1 = np.array(w1)
    w1 = [float(i)/sum(w1) for i in w1]
    
    return w1
    
    
    



def main():
    
    #snrs = np.array_split(np.array(ch_gen(num_samples)), 2)
    snrs = np.array(ch_gen(num_samples))
    labels= np.zeros((num_sens))
    X_train, X_val, y_train, y_val = train_test_split(snrs, snrs, test_size=0.30)
    model = choose_model(get_model, "Dropout_Model_Rl2")
    history = train_model(model, X_train, X_val, y_train, y_val)
    eval_metric(model, history, "loss")
    cooperative_pds = Compute_PdVSPf(model)
    print("cooperative_pds",cooperative_pds)
    maths_weights = weights_from_mathematical_model(snrs)
    print("this is the mathematical weights")
    # simulation plots
    
    plt.plot(pf,cooperative_pds)
    plt.title("Pf Vs Pd")
    plt.xlabel("probability of false alarm")
    plt.ylabel("probability of detection");
    plt.show()

   
    
if __name__ == '__main__': 
    main()
