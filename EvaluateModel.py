# -*- coding: utf-8 -*-
"""
Created on Sun May  3 19:45:45 2020

@author: admin
"""
import h5py
from dataset import Dictionary, VQAEntries
import numpy as np 
import tensorflow as tf 
import random
from DateGeneratorClass import ValidGenFromHard
import VQA_Models as BaseModel
from keras.models import Model
from keras.layers import Input,Dense,Embedding,Dropout,LSTM,Multiply,BatchNormalization,Activation
from keras.layers import RepeatVector,Concatenate,Lambda
from keras.utils import plot_model
import keras.backend as k
import keras
from keras.optimizers import nadam
from keras.preprocessing.image import load_img
import matplotlib.pyplot as plt
from keras.models import load_model
import json
from math import ceil

batch_size = 2048 * 2

correct_ans = json.load(open('correct_ans.json','r'))
annotations = json.load(open('annotations/v2_mscoco_val2014_annotations.json','r')) 
annotations  = annotations['annotations']

# correct_answer = json.load(open('correct_ans.json','r'))
model = load_model('D:/python projects/top down vqa keras/14mordad20.h5')

def getData():
    data_path      = 'H:/VQA_TRAIN_TEST_FEATURES/'
    train_file     = 'train36.hdf5'
    val_file       = 'val36.hdf5'
    train_fetures  = h5py.File(data_path+train_file,'r')        
    valid_features = h5py.File(data_path+val_file,'r')        
    dictionary     = Dictionary.load_from_file('./data/dictionary.pkl')
    entries        = {}
    for name in ['train', 'val']:
        entries[name] = VQAEntries(name, dictionary)
    EmbedMatrix = np.load('data/glove6b_init_300d.npy')
    return train_fetures,valid_features,entries,EmbedMatrix 
train_fetures,valid_features,entries,EmbedMatrix  = getData()

ids_val      =   np.arange(len(entries['val'].images))
image_feature_index  =  entries['val'].images     # a vector with 443757 elements where each element is between 1 to 82000
questions            =  entries['val'].q_token   # this is 420 000 questions and targets
targets              =  entries['val'].target    # this is 420 000 questions and targets

Vall  = entries['val']
Val   = entries['val'].entries
question_id = Val['question_id']

Num_val_samples   =  len(question_id)
valid_generator  = ValidGenFromHard(Num_val_samples,image_feature_index,
                                    valid_features,questions,question_id,correct_ans,
                                    targets,batch_size,False)

eval_file = []
count = 0
num_eval_epochs = ceil(214354/batch_size)
for i in range(num_eval_epochs):
    print(i)
    # [[Av,Aq],Aa,question_index,GT,Q_ID]=valid_generator.__getitem__(i)
    [Av,Aq],Q_ID,GT = valid_generator.__getitem__(i)
    Predicted_Ans = model.predict([Av,Aq])
    Predicted_labels= k.eval(k.argmax(Predicted_Ans,1)) 

    for item in range(batch_size):
        count= count+1
        if count<214355 :
            eval_file.append(
            {
            'answer': Vall.label2ans[Predicted_labels[item]],
            'question_id':int( Q_ID[item]),
            'correct_ans': GT[item]
            }
            )
json.dump(eval_file,open('results/v2_OpenEnded_mscoco_val2014_val_results.json','w'))
json.dump(eval_file,open('results/14mordad','w'))    



# #read train data        
# data_path      = 'H:/VQA_TRAIN_TEST_FEATURES/'
# train_file     = 'train36.hdf5'
# val_file       = 'val36.hdf5'
# # train_fetures  = h5py.File(data_path+train_file,'r')        
# valid_features = h5py.File(data_path+val_file,'r')        

# dictionary     = Dictionary.load_from_file('./data/dictionary.pkl')
# entries        = {}
# for name in ['train', 'val']:
#     entries[name] = VQAEntries(name, dictionary)


# image_id    = Val['image_id']
# image       = Val['image']
# question    = Val['question']
# answer      = Val['answer']
# q_token     = Val['q_token']
# targets     = Vall.target
# Ai          = Vall.images


# valid_generator  = ValidGenFromHard(ids_val,
#                                   entries['val'].images,
#                                   valid_features,
#                                   entries['val'].q_token,
#                                   entries['val'].target,
#                                   batch_size=batch_size,
#                                   correct_ans=correct_ans1,
#                                   question_id  = question_id,
#                                   shuffle = False)

# number_of_questions = 214354;

# number_of_steps_per_epoch = np.round(number_of_questions / batch_size)

# B =121
# # batch_size=512
# total_score = 0
# b = 0
# for i in range(27):
    
#     # i   = 100
#     # print(i)
#     [[Av,Aq],Aa,question_index,Ground_Truth,Q_ID]=valid_generator.__getitem__(i)
#     Ans = model.predict([Av,Aq])

#     Predicted_labels = k.argmax(Ans,1)
#     Predicted_labels1=k.eval(Predicted_labels ) 
#     one_hotted = keras.utils.to_categorical(k.eval(Predicted_labels),3129) 
#     S =  Aa * one_hotted
#     matched_answers = np.argmax(S,1)
#     correct_Ans = np.where(S>0)
#     num_true = len(correct_Ans[0])
#     total_score+=100*num_true/batch_size
#     print(100*num_true/batch_size)



    # def EvaluateBatch():
    # question_index : this goes from 0 : number of questions
    # Aa is the true answer one_hottted

