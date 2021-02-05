#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 17:09:52 2021

@author: mac
"""
import numpy as np
from global_vars import *


class ChannelModel:
    
    
    @staticmethod
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
    
    @staticmethod
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
    
    @staticmethod
    def get_secondary_correlation(dist_su_su_vec: np.ndarray ) -> np.ndarray:
        '''
        this function computes the secondary users correlation using SU-SU distances
        
        Parameters:
         dist_su_su_vec : distances between the secondary users
    
        Output:
            secondary users correlation
        '''    
        return np.exp(-dist_su_su_vec / d_ref)

    @staticmethod
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

    @staticmethod
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
    
    @classmethod
    def ch_gen(cls, num_samples: int) -> np.ndarray:
      """ 
      This function deploy the channel model.
       Parameters:
            num_samples : number of samples
    
        Output:
           signal noise ratio
      
      """
    
      returned_power = []
      returned_SNRs = []
      
      
      for i in range(num_samples):
    
        dist_pr_su_vec, dist_su_su_vec = cls.get_distances()
        pu_ch_gain =  cls.get_channel_gain(dist_pr_su_vec)
        su_cor =  cls.get_secondary_correlation(dist_su_su_vec)
        shadowing =  cls.get_shadowing(su_cor,num_sens)
        pu_power = np.zeros([len(su_cor)]) #pu_power (received power initialization)
        pri_power = p_t #pri_power (transmitted power)
        # test the activity of the primary user 
        if (np.random.rand() < pu_active_prob):
          pu_ch_gain_tot = pu_ch_gain  * shadowing
          pu_power = pu_power +  pri_power*pu_ch_gain_tot
          SNR = pri_power * pow(abs(pu_ch_gain_tot),2)/ sigma_v
        multi_fading =  cls.get_multiPath_Fading(num_sens)
        pu_power = pu_power * multi_fading
        returned_power.append(pu_power)
        returned_SNRs.append(SNR)
      output = dict()
      output['snrs'] = returned_SNRs
      output['power'] = returned_power
    
      return output