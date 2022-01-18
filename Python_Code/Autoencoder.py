# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 15:10:12 2022

@author: Navin
"""

import os
import pandas as pd
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
import pandas as pd
import keras

from keras.models import Sequential 
from keras.layers import  Dropout, Dense

from keras.models import load_model 
from keras.optimizers import Adam

#%%
data=pd.read_csv('/home/scr/data_ae.csv')
data=data.T
data=data.iloc[1:,::]
#%%
samp=60
step=12
fu=12
tr=31
en=36
k=step+5
##### Train X datat #####
x_train=[]
b=[i for i in range(0,tr)]
for j in range (0,len(b)-1):
    train=data.iloc[b[j]*samp:b[j+1]*samp-fu,::]
    for i in range(step,train.shape[0]+1):
        d=train.iloc[i-step:i,:]
        d=np.array(d) 
        x_train.append(d)
x_train=np.array(x_train)
x_train.shape[1]
x_train=x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])
x_train.shape
train=[]
#%%
##### Validation X datat #####
x_vali=[]
c=[i for i in range(tr-1, en)]
for j in range (0, len(c)-1):
    vali=data.iloc[c[j]*samp:c[j+1]*samp-fu,::]
    for i in range(step,vali.shape[0]+1):
        d=vali.iloc[i-step:i,:]
        d=np.array(d)
        x_vali.append(d)
x_vali=np.array(x_vali)
x_vali=x_vali.reshape(x_vali.shape[0], x_vali.shape[1]*x_vali.shape[2])
vali=[]
#%%
##### Training Y datat #####
y_train=[]
b=[i for i in range(0,tr)]
for j in range (0,len(b)-1):
    train1=data.iloc[b[j]*samp+step+fu-1:b[j+1]*samp,::]
    d=np.array(train1)
    y_train.append(d)   
y_train=np.array(y_train)
y_train=y_train.reshape(y_train.shape[0]*y_train.shape[1], y_train.shape[2])

train1=[]
#%%
##### Vaidation Y datat #####
y_vali=[]
c=[i for i in range(tr-1,en)]
c
for j in range (0, len(c)-1):
    vali1=(data.iloc[c[j]*samp+step+fu-1:c[j+1]*samp,::])
    d=np.array(vali1)
    y_vali.append(d)
y_vali=np.array(y_vali)
y_vali.shape

y_vali=y_vali.reshape(y_vali.shape[0]*y_vali.shape[1], y_vali.shape[2])
#%%
model = Sequential()
model.add(Dense(512, input_dim=4140, activation='relu'))
model.add(Dense(384, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(384, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(345, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(345, activation='relu'))

model.summary()
#%%
opt=Adam(lr=0.00001, beta_1=0.9, beta_2=0.99)
model.compile(optimizer='Adam', loss='mse',metrics=['acc','mse'])
#%%
hist=model.fit(x_train,y_train,epochs=1500,batch_size=24,validation_data=(x_vali,y_vali))
#%%
model.save('/home/navin/ae3.h5')
#%%
autoencoder1=load_model('/home/navin/ae3.h5')
#%%