# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 07:49:47 2020
@author: admin
"""
from __future__ import print_function
from __future__ import division
import h5py
from     dataset import Dictionary, VQAEntries
import   numpy as np 
import   tensorflow as tf 
import   random
from     DateGeneratorClass_0 import DataGenerator
import   VQA_Models as BaseModel
from     keras.models import Model,load_model,save_model
from     keras.layers import Input,Dense,Embedding,Dropout,LSTM,Multiply,BatchNormalization,Activation
from     keras.layers import RepeatVector,Concatenate,Lambda
from     keras.utils import plot_model
import   keras.backend as k
import   keras
from     keras.optimizers import nadam
from     keras.preprocessing.image import load_img
import   matplotlib.pyplot as plt


def SUM(x):
    import keras.backend as k
    att_v = x[0]*x[1]
    return k.sum(att_v,axis = -2)
    
batch_size=512
        
#read train data        
data_path      = 'H:/VQA_TRAIN_TEST_FEATURES/'
train_file     = 'train36.hdf5'
val_file       = 'val36.hdf5'
train_fetures  = h5py.File(data_path+train_file,'r')        
valid_features = h5py.File(data_path+val_file,'r')        

dictionary     = Dictionary.load_from_file('./data/dictionary.pkl')
entries        = {}
for name in ['train', 'val']:
    entries[name] = VQAEntries(name, dictionary)

Vall  = entries['train']
Val   = entries['train'].entries
question_id = Val['question_id']
image_id    = Val['image_id']
image       = Val['image']
question    = Val['question']
answer      = Val['answer']
q_token     = Val['q_token']
targets     = Vall.target

ids_train    =   np.arange(len(entries['train'].images))
ids_val      =   np.arange(len(entries['val'].images))


EmbedMatrix = np.load('data/glove6b_init_300d.npy')
d_rate = 0.4
Vocab_length = entries['train'].dictionary.ntoken
Num_Hidden = 512
def Att(d_rate,Vocab_length,num_obj,v_dim ):
   
    v = Input(shape=(num_obj,v_dim),name = 'V')
    
    Q     = Input(shape=(14,),name='question')
    E     = Embedding(19901,300,weights=[EmbedMatrix], input_length =14, trainable=False ,name='q')(Q)
   
    
    LSTM1 = LSTM(1024,return_sequences=False,name='LSTM_l2')(E)
    LSTM1 = Dropout(0.4)(LSTM1)

    q     = Dense(1024,activation='relu',name='encoded_question')(LSTM1)
    # q     = Dropout(0.5)(q)
    
    # need a lambda layer to expand q 
    q_expand   = RepeatVector(num_obj,name = 'q_expand')(LSTM1)
    vq         = Concatenate(axis = -1,name = 'vq')([v,q_expand])
    D1         = Dense(1024,activation='relu', input_dim=2,name = 'D1')(vq)
    D1 = Dropout(0.4)(D1)
    D2         = Dense(1,activation='softmax',input_dim=2,name = 'D2')(D1)
    # L_aat_v    = Lambda(SUM)
    att_scores = Lambda(SUM)([D2,v])

    D3 = Dense(1024,activation = 'relu',name = 'v1024')(att_scores)
    # D3 = Dropout(0.5)(D3)
    # 
    fv = Multiply(name='multiply')([D3,q])
 
    F = Dense(4096,activation='relu',name='Dense2')(fv)
    F = Dropout(0,5)(F)
   

    
    output=Dense(3129,activation='sigmoid',name='output')(F)
    model = Model([v,Q],output)
    # loss = binary_crossentropy(from_logits=False)
    optimizer =nadam(learning_rate=0.000001)
    # metr = keras.metrics(y_true, y_pred, threshold=0.5)
    # Score = Scores(y_true,y_pred)
    model.compile(optimizer=optimizer,loss='binary_crossentropy')
    model.summary()
    return model


model = Att(d_rate,Vocab_length,num_obj=36,v_dim=2048)

train_generator  = DataGenerator(ids_train,
                                  entries['train'].images,
                                  train_fetures,
                                  entries['train'].q_token,
                                  entries['train'].target,
                                  batch_size=512,
                                  shuffle = False) 
for i in range(4):
    history = model.fit_generator(train_generator,
                    # validation_data = valid_generator,
                    epochs = 1 ,
                    steps_per_epoch=20,
                    verbose =1,
                    # validation_steps=200
                    )
   
    file_path = '30tir'+str(i)+'.h5'
    print('epoch : ',str(i))

    model.save(file_path)


# model.load_weights('9tir48.h5')
# optimizer =nadam(learning_rate=0.000001)
#     # metr = keras.metrics(y_true, y_pred, threshold=0.5)
#     # Score = Scores(y_true,y_pred)
# model.compile(optimizer=optimizer,loss='binary_crossentropy')
# model.summary()


# from keras.models import load_model
# model = load_model('25khordad69.h5')

