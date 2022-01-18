#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 09:12:55 2020

@author: navin
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D,Conv2DTranspose, BatchNormalization, Flatten, Reshape, Dropout
from keras.layers import TimeDistributed, LSTM, ConvLSTM2D, Concatenate
from keras.models import Model
import cv2
from keras.models import save_model, load_model
from keras import backend as back
from keras.optimizers import Adam
from keras.utils import plot_model
from keras import metrics
import os
from keras.utils import to_categorical

#%%
data_path='/home/navin/data_d/train/crop/'
img_list=os.listdir(data_path)
img_list.sort()
len(img_list)
img_data_list=[]
test_holder=[]
for img in img_list:
    input_img=cv2.imread(data_path+img)
    test_img=cv2.imread(data_path+img,0)
    input_img=input_img.flatten()
    test_img=test_img.flatten()
    test_holder.append(test_img)
    img_data_list.append(input_img)

#%%
data1=np.array(img_data_list)
data1.shape
data3=np.array(test_holder)
np.unique(data3)
gray=np.zeros_like(data3)
gray[np.logical_and(data3>=0,data3<40)]=0
gray[np.logical_and(data3>=40,data3<140)]=1 
gray[np.logical_and(data3>=190, data3<255)]=2
gray[np.logical_and(data3>=140,data3<190)]=3
data=pd.DataFrame(data1)
data3=pd.DataFrame(gray)
np.unique(data3)
data1=[]
data

samp=60
step=12
fu=12
tr=31
en=36
k=step+5

x_train=[]
b=[i for i in range(0,tr)]
for j in range (0,len(b)-1):
    train=data.iloc[b[j]*samp:b[j+1]*samp-fu,::]
    for i in range(step,train.shape[0]+1):
        d=train.iloc[i-step:i,:]
        d=np.array(d) 
        x_train.append(d)
x_train=np.array(x_train)
##x_train.shape
train=[]

x_vali=[]
c=[i for i in range(tr-1, en)]
for j in range (0, len(c)-1):
    vali=data.iloc[c[j]*samp:c[j+1]*samp-fu,::]
    for i in range(step,vali.shape[0]+1):
        d=vali.iloc[i-step:i,:]
        d=np.array(d)
        x_vali.append(d)
x_vali=np.array(x_vali)
#x_vali
vali=[]

test=[]
row,col,channel=192,448,3
x_train=x_train.reshape(x_train.shape[0],x_train.shape[1],row,col,channel)
x_train.shape

x_vali=x_vali.reshape(x_vali.shape[0],x_vali.shape[1],row,col,channel)
x_vali.shape

#x_test=x_test.reshape(x_test.shape[0],x_test.shape[1], row,col,channel)
#x_test.shape

#%%
row,col,chan=192,448,1
y_tr=[]
b=[i for i in range(0,tr)]
for j in range (0,len(b)-1):
    train1=data3.iloc[b[j]*samp+step+11:b[j+1]*samp,::]
    d=np.array(train1)
    y_tr.append(d)   
y_tr=np.array(y_tr)
y_tr.shape
train1=[]
y_train=y_tr.reshape(y_tr.shape[0]*y_tr.shape[1],row,col,chan)
y_train.shape
y_train1=to_categorical(y_train)
y_train1.shape

y_train=[]

y_va=[]
c=[i for i in range(tr-1, en)]
for j in range (0, len(c)-1):
    vali1=(data3.iloc[c[j]*samp+step+11:c[j+1]*samp,::])
    d=np.array(vali1)
    y_va.append(d)
y_va=np.array(y_va)
vali1
vali1=[]
y_vali=y_va.reshape(y_va.shape[0]*y_va.shape[1],row,col,chan)
y_vali1=to_categorical(y_vali)
y_vali1.shape
y_vali=[]

#%%
back.clear_session()
row,col,channel=192,448,3
row,col=192,448
f1=48
f2=36
f3=24
f4=12
f5=4
input_img = Input(shape=(step,row,col,channel)) 
#input_img = BatchNormalization()(input_img) 
x1 = ConvLSTM2D(f1, (3,3),strides=(1,1),activation='relu', padding='same', kernel_initializer='he_uniform',return_sequences=True,dropout=0.1)(input_img)
x1 = BatchNormalization()(x1)

x2= ConvLSTM2D(f2, (3,3),strides=(1,1),activation='relu', padding='same', kernel_initializer='he_uniform',return_sequences=True,dropout=0.1)(x1)
x2 = BatchNormalization()(x2)

x3 = ConvLSTM2D(f3, (3,3),strides=(1,1),activation='relu', padding='same', kernel_initializer='he_uniform',return_sequences=True ,dropout=0.1)(x2)
x3 = BatchNormalization()(x3)

x4 = ConvLSTM2D(f3, (3,3),strides=(1,1),activation='relu', padding='same', kernel_initializer='he_uniform',return_sequences=True ,dropout=0.1)(x3)
x4 = BatchNormalization()(x4)

x5 = ConvLSTM2D(f4, (3,3),strides=(1,1),activation='relu', padding='same', kernel_initializer='he_uniform',return_sequences=False,dropout=0.1)(x4)
x5 = BatchNormalization()(x5)

x6 = Conv2D(f5, (3,3),strides=(1,1),activation='softmax', padding='same')(x5)
#x5 = BatchNormalization()(x5)
autoencoder = Model(input_img, x6)
autoencoder.summary()
#%%
opt=Adam(lr=0.0001, beta_1=0.9, beta_2=0.999)
autoencoder.compile(optimizer=opt, loss='categorical_crossentropy',metrics=['acc'])
#%%
history=autoencoder.fit(x_train,y_train1,epochs=40,batch_size=1,validation_data=(x_vali,y_vali1))
#%%
autoencoder.save('/home/navin/model1121.h5')
#%%
autoencoder1=load_model('/home/navin/model1120.h5')
# For testing
#%%
x_train=[]
x_vali=[]
y_train=[]
y_vali=[]
x_test=[]
data=[]
data3=[]
autoencoder=[]
#%%
data_path='/home/navin/data_d/test/'
img_list1=os.listdir(data_path)
img_list1.sort()
len(img_list1)


img_data_list1=[]
test_holder=[]
for img in img_list1:
    input_img=cv2.imread(data_path+img)
    test_img=cv2.imread(data_path+img,0)
    input_img=input_img.flatten()
    test_img=test_img.flatten()
    test_holder.append(test_img)
    img_data_list1.append(input_img)
img_data_list1
#%%
data11=np.array(img_data_list1)
data11.shape
data33=np.array(test_holder)
np.unique(data33)
gray=np.zeros_like(data33)
gray[np.logical_and(data33>=0,data33<40)]=0
gray[np.logical_and(data33>=40,data33<140)]=1 
gray[np.logical_and(data33>=190, data33<255)]=2
gray[np.logical_and(data33>=140,data33<190)]=3
data=pd.DataFrame(data11)
data33=pd.DataFrame(gray)
np.unique(data33)
#data11=[]
data
#%%
samp=60
step=12
fu=6
tr=16
en=36
k=step+5
x_test=[]
b=[i for i in range(0,tr)]
for j in range (0,len(b)-1):
    test=data.iloc[b[j]*samp:b[j+1]*samp-fu,::]
    for i in range(step,test.shape[0]+1):
        d=test.iloc[i-step:i,:]
        d=np.array(d) 
        x_test.append(d)
x_test=np.array(x_test)
##x_test.shape
test=[]
row,col,channel=192,448,3
x_test=x_test.reshape(x_test.shape[0],x_test.shape[1],row,col,channel)
x_test.shape

#x_test=x_test.reshape(x_test.shape[0],x_test.shape[1], row,col,channel)
#x_test.shape
#%%

y_te=[]
b=[i for i in range(0,tr)]
for j in range (0,len(b)-1):
    test1=data33.iloc[b[j]*samp+step+fu-1:b[j+1]*samp,::]
    d=np.array(test1)
    y_te.append(d)   
y_te=np.array(y_te)
y_te.shape

#y_test
test1=[]
y_test= y_te.reshape(y_te.shape[0]*y_te.shape[1], row, col, 1)
y_test1=to_categorical(y_test)

y_test=[]
y_test1.shape

#%%
d=autoencoder1.predict(x_test, batch_size=1)
d.shape
#%%
d.shape
a=d[:,:,:,:] 
a.shape
b=y_test1[:,:,:,:]
b.shape
#%%
pred=a.argmax(axis=-1)
pred.shape
#pred=a.copy()
pred_n=np.zeros(shape=(a.shape[0],192,448,3))
pred_n[np.logical_and(pred>=0,pred<1)]=[0,0,0]
pred_n[np.logical_and(pred>=1,pred<2)]=[0,0,255]
pred_n[np.logical_and(pred>=2,pred<3)]=[0,255,255]
pred_n[np.logical_and(pred>=3,pred<=4)]= [0,255,0]
pred_n.shape
#%%
true=b.argmax(axis=-1)
true_n=np.zeros(shape=(a.shape[0],192,448,3))
true_n[np.logical_and(true>=0,true<1)]=[0,0,0]
true_n[np.logical_and(true>=1,true<2)]=[0,0,255]
true_n[np.logical_and(true>=2,true<3)]=[0,255,255]
true_n[np.logical_and(true>=3,true<=4)]= [0,255,0]
#%%
for i in range (0, a.shape[0]):
#    c=np.concatenate([pred_n[i],b[i]],axis=1)
    cv2.imwrite('/home/navin/data_d/clstm/p20/{}.png'.format(i),pred_n[i])
    cv2.imwrite('/home/navin/data_d/clstm/t20/{}.png'.format(i),true_n[i])
    #%%
