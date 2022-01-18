#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 13:10:22 2020

@author: navin
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from keras.layers import Input, Conv2D, Conv2DTranspose, BatchNormalization, Flatten, Reshape, Dropout
from keras.layers import TimeDistributed, LSTM, Concatenate
from keras.models import Model
import cv2
from keras.models import save_model, load_model
from keras import backend as K
from keras.optimizers import Adam
from keras.utils import plot_model
from keras import metrics
import os
from keras.utils import to_categorical

#%%
"""" Read image data from directory and store color image and gray scale image"""
data_path='/home/navin/GAN_thesis/dataset1/'
img_list=os.listdir(data_path)

img_list[0]
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
""" converting congestion class into categories"""
data3=np.array(test_holder)
test_holder = []
gray=np.zeros_like(data3)
gray[np.logical_and(data3>=0,data3<40)]=0
gray[np.logical_and(data3>=40,data3<140)]=1 
gray[np.logical_and(data3>=190, data3<255)]=2
gray[np.logical_and(data3>=140,data3<190)]=3
data3=pd.DataFrame(gray)

#%%
row = 192
col = 448
channel = 3
data_per_day = 60
past_sequence = 12
prediction_horizon = 2
#split data into train and test based on number of day
train_upto_day = 25
total_day = 30
k = past_sequence + 1
#%%
""" TRAIN and VALIDATION x and y data"""
y_train=[]
b=[i for i in range(0,train_upto_day)]
for j in range (0,len(b)-1):
    train1=data3.iloc[b[j]*data_per_day : b[j+1]*data_per_day,::]
    for i in range(past_sequence+prediction_horizon,train1.shape[0]+1):
        d=train1.iloc[i-past_sequence:i,:]
        d=np.array(d)
        y_train.append(d)   
y_train=np.array(y_train)
train1=[]
y_train= y_train.reshape(y_train.shape[0],y_train.shape[1], row, col, 1)

y_vali=[]
b=[i for i in range(train_upto_day-1,total_day)]
for j in range (0,len(b)-1):
    vali1=data3.iloc[b[j]*data_per_day : b[j+1]*data_per_day,::]
    for i in range(past_sequence+prediction_horizon,vali1.shape[0]+1):
        d=vali1.iloc[i-past_sequence:i,:]
        d=np.array(d)
        y_vali.append(d)   
y_vali=np.array(y_vali)
vali1 =[]
y_vali = y_vali.reshape(y_vali.shape[0], y_vali.shape[1],row, col,1)
y_train1=to_categorical(y_train)
y_train1.shape
y_vali1=to_categorical(y_vali)
y_vali1.shape
data3 = []
y_train =[]
y_vali = []

data1=np.array(img_data_list)
img_data_list =[]
data=pd.DataFrame(data1)
x_train=[]
b=[i for i in range(0,train_upto_day)]
for j in range (0,len(b)-1):
    train=data.iloc[b[j]*data_per_day:b[j+1]*data_per_day-prediction_horizon,::]
    for i in range(past_sequence,train.shape[0]+1):
        d=train.iloc[i-past_sequence:i,:]
        d=np.array(d) 
        x_train.append(d)
x_train=np.array(x_train)
x_train=x_train.reshape(x_train.shape[0],x_train.shape[1],row,col,channel)
print(x_train.shape)

x_vali=[]
b=[i for i in range(train_upto_day-1, total_day)]
b
for j in range (0,len(b)-1):
    vali = data.iloc[b[j]*data_per_day:b[j+1]*data_per_day-prediction_horizon,::]
    for i in range(past_sequence,vali.shape[0]+1):
        d = vali.iloc[i-past_sequence:i,:]
        d = np.array(d) 
        x_vali.append(d)
x_vali=np.array(x_vali)
x_vali=x_vali.reshape(x_vali.shape[0],x_vali.shape[1],row,col,channel)
x_vali.shape
data = []
#%%
""" Architecture design of PredNet""" 
step = 12
row,col,channel= 192,448,3
f1=32
f2=64
f3=96
f4=128
f5=160
f6=192
input_img = Input(shape=(step,row,col, channel))  
x1 = TimeDistributed(Conv2D(f1, (3, 3),strides=(1,1),activation='relu', padding='same', kernel_initializer='he_uniform'))(input_img)
x1 = TimeDistributed(BatchNormalization())(x1)
x1 = TimeDistributed(Dropout(0.1))(x1)

x2 = TimeDistributed(Conv2D(f2, (2,2), strides=(2,2),activation='relu', padding='valid', kernel_initializer='he_uniform'))(x1)
x2 = TimeDistributed(BatchNormalization())(x2)
x2 = TimeDistributed(Dropout(0.1))(x2)
x2 = TimeDistributed(Conv2D(f2, (3, 3), strides=(1,1),activation='relu', padding='same', kernel_initializer='he_uniform'))(x2)
x2 = TimeDistributed(BatchNormalization())(x2)
x2 = TimeDistributed(Dropout(0.1))(x2)

x3 = TimeDistributed(Conv2D(f3, (2,2), strides=(2,2),activation='relu', padding='valid', kernel_initializer='he_uniform'))(x2)
x3 = TimeDistributed(BatchNormalization())(x3)
x3 = TimeDistributed(Dropout(0.1))(x3)
x3 = TimeDistributed(Conv2D(f3, (3, 3), strides=(1,1),activation='relu', padding='same', kernel_initializer='he_uniform'))(x3)
x3 = TimeDistributed(BatchNormalization())(x3)
x3 = TimeDistributed(Dropout(0.1))(x3)

x4 = TimeDistributed(Conv2D(f4, (2,2), strides=(2,2),activation='relu', padding='valid', kernel_initializer='he_uniform'))(x3)
x4 = TimeDistributed(BatchNormalization())(x4)
x4 = TimeDistributed(Dropout(0.1))(x4)
x4 = TimeDistributed(Conv2D(f4, (3, 3), strides=(1,1),activation='relu', padding='same', kernel_initializer='he_uniform'))(x4)
x4 = TimeDistributed(BatchNormalization())(x4)
x4 = TimeDistributed(Dropout(0.1))(x4)

x5 = TimeDistributed(Conv2D(f5, (2,2), strides=(2,2),activation='relu', padding='valid', kernel_initializer='he_uniform'))(x4)
x5 = TimeDistributed(BatchNormalization())(x5)
x5 = TimeDistributed(Dropout(0.1))(x5)
x5 = TimeDistributed(Conv2D(f5, (3, 3), strides=(1,1),activation='relu', padding='same', kernel_initializer='he_uniform'))(x5)
x5 = TimeDistributed(BatchNormalization())(x5)
x5 = TimeDistributed(Dropout(0.1))(x5)

x6 = TimeDistributed(Conv2D(f6, (2,2), strides=(2,2),activation='relu', padding='valid', kernel_initializer='he_uniform'))(x5)
x6 = TimeDistributed(BatchNormalization())(x6)
x6 = TimeDistributed(Dropout(0.1))(x6)
x6 = TimeDistributed(Conv2D(8, (3, 3), strides=(1,1),activation='relu', padding='same', kernel_initializer='he_uniform'))(x6)
x6 = TimeDistributed(Flatten())(x5)

encoded = TimeDistributed(Dropout(0.1))(x6)
#
lstm = LSTM(672,return_sequences=True, activation='relu', kernel_initializer='he_uniform')(encoded)
lstm = LSTM(1024,return_sequences=True, activation='relu', kernel_initializer='he_uniform')(lstm)
lstm = Dropout(0.2)(encoded)
lstm = LSTM(672, return_sequences=True, activation='relu', kernel_initializer='he_uniform')(lstm)
lstm = Dropout(0.2)(lstm)
lstm = LSTM(672,return_sequences=True, activation='relu', kernel_initializer='he_uniform')(lstm)
lstm = Dropout(0.2)(lstm)
lstm = LSTM(672, activation='relu',return_sequences=True, kernel_initializer='he_uniform')(lstm)
#
#
d1 = Reshape((step,6,14,8))(lstm)
d1 = TimeDistributed(Conv2DTranspose(f5, (2,2), strides=(2,2),activation='relu', padding='valid', kernel_initializer='he_uniform'))(d1)
d1 = TimeDistributed(BatchNormalization())(d1)
d1 = Concatenate()([d1,x5])
d1 = TimeDistributed(Dropout(0.1))(d1)
d1 = TimeDistributed(Conv2D(f5, (3, 3), strides=(1,1),activation='relu', padding='same', kernel_initializer='he_uniform'))(x5)
d1 = TimeDistributed(BatchNormalization())(d1)
d1 = TimeDistributed(Dropout(0.1))(d1)

d2 = TimeDistributed(Conv2DTranspose(f4, (2,2), strides=(2,2),activation='relu', padding='valid', kernel_initializer='he_uniform'))(d1)
d2 = TimeDistributed(BatchNormalization())(d2)
d2 = Concatenate()([d2,x4])
d2 = TimeDistributed(Dropout(0.1))(d2)
d2 = TimeDistributed(Conv2D(f4, (3, 3), strides=(1,1),activation='relu', padding='same', kernel_initializer='he_uniform'))(d2)
d2 = TimeDistributed(BatchNormalization())(d2)
d2 = TimeDistributed(Dropout(0.1))(d2)

d3 = TimeDistributed(Conv2DTranspose(f3, (2,2), strides=(2,2),activation='relu', padding='valid', kernel_initializer='he_uniform'))(d2)
d3 = TimeDistributed(BatchNormalization())(d3)
d3 = Concatenate()([d3,x3])
d3 = TimeDistributed(Dropout(0.1))(d3)
d3 = TimeDistributed(Conv2D(f3, (3, 3), strides=(1,1),activation='relu', padding='same', kernel_initializer='he_uniform'))(d3)
d3 = TimeDistributed(BatchNormalization())(d3)
d3 = TimeDistributed(Dropout(0.1))(d3)

d4 = TimeDistributed(Conv2DTranspose(f2, (2,2), strides=(2,2),activation='relu', padding='valid', kernel_initializer='he_uniform'))(d3)
d4 = TimeDistributed(BatchNormalization())(d4)
d4 = Concatenate()([d4,x2])
d4 = TimeDistributed(Dropout(0.1))(d4)
d4 = TimeDistributed(Conv2D(f2, (3, 3), strides=(1,1),activation='relu', padding='same', kernel_initializer='he_uniform'))(d4)
d4 = TimeDistributed(BatchNormalization())(d4)
d4 = TimeDistributed(Dropout(0.1))(d4)



d5 = TimeDistributed(Conv2DTranspose(f1, (2,2), strides=(2,2),activation='relu', padding='valid', kernel_initializer='he_uniform'))(d4)
d5 = TimeDistributed(BatchNormalization())(d5)
d5 = Concatenate()([d5,x1])
d5 = TimeDistributed(Dropout(0.1))(d5)
d5 = TimeDistributed(Conv2D(f1, (3, 3), strides=(1,1),activation='relu', padding='same', kernel_initializer='he_uniform'))(d5)
d5 = TimeDistributed(BatchNormalization())(d5)
d5 = TimeDistributed(Dropout(0.1))(d5)

decoded = TimeDistributed(Conv2D(4, (3, 3), activation='softmax', padding='same',kernel_initializer='he_uniform'))(d5)

 
predNet = Model(input_img, decoded) 
predNet.summary()

#%%
#opt=Adam(lr=0.000001, beta_1=0.5, beta_2=0.999)
#from keras import backend as k
#autoencoder.compile(optimizer=opt, loss='mse',metrics=['acc'])
#autoencoder.fit(x_train,y_train, epochs=200, batch_si ze = 1, validation_data=(x_vali, y_vali))
#%%     
from keras.utils import multi_gpu_model
parallel_model = multi_gpu_model(predNet, gpus= 4)
parallel_model.compile(optimizer= 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
parallel_model.fit(x_train,y_train1, epochs=1000, batch_size = 32, validation_data=(x_vali, y_vali1))
#%%
""" TEST data x and Y"""
test_data_dir = '/home/navin/GAN_thesis/dataset1/'
test_holder1 = []
test_list = os.listdir(test_data_dir)
test_list.sort()
for each in test_list:
    test = cv2.imread(test_data_dir + each)
    test = test.flatten()
    test_holder1.append(test)

x_test=[]
b=[i for i in range(30,32)]
for j in range (0,len(b)-1):
    test=data.iloc[b[j]*data_per_day:b[j+1]*data_per_day-prediction_horizon,::]
    for i in range(past_sequence,test.shape[0]+1):
        d=test.iloc[i-past_sequence:i,:]
        d=np.array(d) 
        x_test.append(d)
x_test=np.array(x_test)
x_test=x_test.reshape(x_test.shape[0],x_test.shape[1],row,col,channel)
print(x_test.shape)
#%%
""" Prediction Metrics analysis """
xx = parallel_model.predict(x_test, batch_size = 32)
predict = xx[:,11,:,:,:]
predict = predict.argmax(axis = -1)
predict = predict.reshape(47, 96*224)
predict_n=np.zeros(shape=(predict.shape[0],96,224,3))
predict_n[np.logical_and(predict>=0,predict<1)]=[0,0,0]
predict_n[np.logical_and(predict>=1,predict<2)]=[75,90,255]
predict_n[np.logical_and(predict>=2,predict<3)]=[75,225,250]
predict_n[np.logical_and(predict>=3,predict<=4)]= [100,195,140]
predict.shape
#cv2.imwrite('/home/navin/i1.png', predict_n[0])
#%%
cv2.imshow('i',predict_n[0])
cv2.waitKey(0)
cv2.destroyAllWindows()
#%%
y_test=[]
b=[i for i in range(30,32)]
for j in range (0,len(b)-1):
    test1=data3.iloc[b[j]*data_per_day : b[j+1]*data_per_day,::]
    for i in range(past_sequence+prediction_horizon,test1.shape[0]+1):
        d=test1.iloc[i-past_sequence:i,:]
        d=np.array(d)
        y_test.append(d)   
y_test=np.array(y_test)
test1 =[]
y_test = y_test.reshape(y_test.shape[0], y_test.shape[1],row, col,1)
y_test1=to_categorical(y_test)

y_test1 = y_test[:,11,:,:]
y_test1 = y_test1.reshape(47, 96*224)
#%%
from sklearn.metrics import accuracy_score
def accuracy(true,pred):
    return accuracy_score(true,pred)
#%%
h= []
for i in range(0,x_test.shape[0]):
    x = accuracy_score(y_test1[i],predict[i])
    h.append(x)
    #%%
df = pd.DataFrame(h)
df.to_csv('/home/navin/batch_32.csv')
#%%