import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wfdb
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTENC
from sklearn.model_selection import StratifiedKFold
from ecgdetectors import Detectors
from scipy import signal
import matplotlib.pyplot as plt
import neurokit2 as nk
import biosppy
import tensorflow as tf
from keras.models import load_model
model = load_model('C:/Users/chint\OneDrive/Desktop/sample/Projectversion1/models/lstm_model.h5')
print("hi")
final_list_X = []
final_list_y = []  
def filter_data(val):
    ecg = val 
    Fs = 500 
    N = ecg.shape[1]
    t = ((np.linspace(0, N-1, N))/(Fs))
    cover = t.shape[0]
    t = t.reshape(1, cover)  
    n=2 
    Fcutoff_low = 0.5 
    Wn_low = ((2*Fcutoff_low)/(Fs))
    b_low, a_low = signal.butter(n, Wn_low, 'low')
    xn_filtered_LF = signal.filtfilt(b_low, a_low, ecg)

    Fcutoff_high = 40 
    Wn_high = ((2*Fcutoff_high)/(Fs))
    b_high, a_high = signal.butter(n, Wn_high, 'high')
    xn_filtered_HF = signal.filtfilt(b_high, a_high, ecg)
    xn = (ecg-xn_filtered_HF-xn_filtered_LF)
    return xn
def return_peaks(ecg_test):
    cleaned = nk.ecg_clean(ecg_test, sampling_rate = 500)  
    rdet, = biosppy.ecg.hamilton_segmenter(signal = cleaned, sampling_rate = 500)   
    rdet, = biosppy.ecg.correct_rpeaks(signal = cleaned, rpeaks = rdet, sampling_rate = 500, tol = 0.05)
    if(rdet.size<=4):       
        return 'INCOMPLETE'
    rdet = np.delete(rdet, -1)       
    rdet = np.delete(rdet, 0)
    rpeaks = {'ECG_R_Peaks': rdet}   
    cleaned_base = nk.signal_detrend(cleaned, order=0)
    signals, waves = nk.ecg_delineate(cleaned_base, rpeaks, sampling_rate = 500, method = "dwt") 
    rpeakss = rpeaks.copy() 
    temppo = 4-len(rpeakss['ECG_R_Peaks'])
    if temppo>0:
        for i in range(temppo):
            rpeakss['ECG_R_Peaks'] = np.append(rpeakss['ECG_R_Peaks'], rpeakss['ECG_R_Peaks'][-1] + 1)
    signals1, waves1 = nk.ecg_delineate(cleaned_base, rpeakss, sampling_rate = 500, method = "peak")
    if temppo>0:
        for j in range(temppo):
            waves1['ECG_Q_Peaks'] = waves1['ECG_Q_Peaks'][:-1] #remove the last element by slicing.(waves1['ECG_Q_Peaks'] is a list)   
    return (cleaned_base, [waves['ECG_T_Peaks'], waves['ECG_R_Onsets'], waves['ECG_R_Offsets'], waves1['ECG_Q_Peaks']])
def final_data_return(given_list):
    mini = 50; 
    for check_index3 in range(12):
        for second_ind3 in range(4):
            mini = min(mini, len(given_list[check_index3][1][second_ind3]))
    to_take = min(16, mini)
    for x in range(to_take):
        a_temp_list = []  
        flag = -1      
        for y in range(12): 
            if((np.isnan(given_list[y][1][1][x])) or (np.isnan(given_list[y][1][2][x])) or (np.isnan(given_list[y][1][3][x])) or (np.isnan(given_list[y][1][0][x]))):
                a_temp_list = []
                flag = 1
                break 
            first_feat = given_list[y][0][int(given_list[y][1][1][x])] - given_list[y][0][int(given_list[y][1][2][x])]
            second_feat = given_list[y][0][int(given_list[y][1][3][x])]  
            third_feat = given_list[y][0][int(given_list[y][1][0][x])]
            a_temp_list.append(first_feat)
            a_temp_list.append(second_feat)
            a_temp_list.append(third_feat)
        if(flag == -1):
            final_list_X.append(a_temp_list)
def load_raw_data():
    signal, meta_val = wfdb.rdsamp('C:/Users/chint/OneDrive/Desktop/sample/Projectversion1/uploaded/19002_hr')   
    value = signal.T
    temp_list = []
    flag1 = -1
    for ind in range(12):
        val_ind = value[ind]
        tmpp = val_ind.shape[0]
        val_ind = val_ind.reshape(1, tmpp)
        val_filtered = filter_data(val_ind)
        val_filtered = val_filtered.reshape(val_filtered.shape[1], ) 
        a_var = return_peaks(val_filtered)
        if(a_var == 'INCOMPLETE'):
            temp_list = []
            flag1 = 1
            break
        temp_list.append(a_var)
    if(flag1==-1):
        final_data_return(temp_list)
load_raw_data()
pred=np.argmax(model.predict(final_list_X), axis = -1)
print(pred)
