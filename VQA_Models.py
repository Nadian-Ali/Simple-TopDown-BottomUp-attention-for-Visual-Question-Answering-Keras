# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 20:24:48 2020

@author: admin
"""


from __future__ import print_function
from __future__ import division
import h5py
from dataset import Dictionary, VQAEntries
import numpy as np 
import tensorflow as tf 
import random
from DateGeneratorClass import DataGenerator
import keras.backend as k
from keras.models import Model
from keras.layers import Input,Dense,Embedding,Dropout,LSTM,Multiply,BatchNormalization,Activation
from keras.utils import plot_model
from keras.layers import RepeatVector, Concatenate, Lambda
from keras.losses import binary_crossentropy
from keras.optimizers import nadam,adam,RMSprop
import keras

def SUM(x):
    import keras.backend as k
    att_v = x[0]*x[1]
    return k.sum(att_v,axis = -2)
    

def Att(d_rate,Vocab_length,num_obj,v_dim ):
   
    v = Input(shape=(num_obj,v_dim),name = 'V')
    
    Q     = Input(shape=(14,),name='question')
    E     = Embedding(19901,300,input_length=14,name='q')(Q)
    LSTM1 = LSTM(1024,return_sequences=False,name='LSTM_l2')(E)
    LSTM1 = Dropout(0.5)(LSTM1)
    q     = Dense(512,activation='relu',name='encoded_question')(LSTM1)
    
    # need a lambda layer to expand q 
    q_expand   = RepeatVector(num_obj,name = 'q_expand')(LSTM1)
    vq         = Concatenate(axis = -1,name = 'vq')([v,q_expand])
    D1         = Dense(256,activation='relu', input_dim=2,name = 'D1')(vq)
    D2         = Dense(1,activation='softmax',input_dim=2,name = 'D2')(D1)
    # L_aat_v    = Lambda(SUM)
    att_scores = Lambda(SUM)([D2,v])

    D3 = Dense(512,activation = 'relu',name = 'v1024')(att_scores)
    D3 = Dropout(0.5)(D3)
    fv = Multiply(name='multiply')([D3,q])
 
    F = Dense(512,activation='relu',name='Dense2')(fv)
    # F = Dropout(0,5)(F)
   

    
    output=Dense(3129,activation='sigmoid',name='output')(F)
    model = Model([v,Q],output)
    # loss = binary_crossentropy(from_logits=False)
    optimizer =nadam(learning_rate=0.0002)
    # metr = keras.metrics(y_true, y_pred, threshold=0.5)
    model.compile(optimizer=optimizer,loss='binary_crossentropy',metrics = ['binary_accuracy'])
    model.summary()
    return model


def Base(d_rate,Vocab_length ):
       
    ImInput = Input(shape=(2048,),name='image_input')
    # ImInput=BatchNormalization()(ImInput)
                 
    ImF =    Dense(512)(ImInput)
    # ImF = BatchNormalization()(ImF)
    ImF = Activation('relu')(ImF)
    
    X = Input(shape=(14,),name='question')
    #E =Embedding(12379,100,weights=[Embedding_matrix], input_length=26, trainable=False)(X)
    E =Embedding(19901,100,input_length=14,name='q')(X)
    # LSTM1 = LSTM(512,return_sequences=True,name='LSTM_l1')(E)
    # LSTM1 = Dropout(0.5,name='dropout1')(LSTM1)
    LSTM1 = LSTM(1024,return_sequences=False,name='LSTM_l2')(E)
    LSTM1 = Dropout(0.5)(LSTM1)
    Ques_D=Dense(1024,activation='relu',name='encoded_question')(LSTM1)
    
    fv = Multiply(name='multiply')([ImF,Ques_D])
    
    F = Dense(1024,activation='relu',name='Dense1')(fv)
    # F = Dropout(0,5)(F)
    # F = Dense(3129,activation='tanh',name='Dense2')(F)
    # F = Dropout(0,5)(F)
    
    output=Dense(3129,activation='softmax',name='output')(F)
    model = Model([ImInput,X],output)
    model.compile(optimizer='nadam',loss='categorical_crossentropy',metrics = ['accuracy'])
    model.summary()
    return model


