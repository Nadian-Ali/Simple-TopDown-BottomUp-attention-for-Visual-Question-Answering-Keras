# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 09:43:21 2020

@author: admin
"""

import numpy as np
import keras

class DataGenerator(keras.utils.Sequence):
    def __init__(self,train_ids,image_features,questions,image_feature_index,targets,batch_size):
        self.image_features = image_features
        self.questions = questions
        self.targets = targets 
        self.batch_size = batch_size
        self.image_feature_index = image_feature_index
        self.train_ids = train_ids
    def __len__(self):
     'Denotes the number of batches per epoch'
     self.Len = int(np.floor(len(self.train_ids) / self.batch_size))
     print(self.Len)
     return self.Len     

    def __getitem__(self, index):
        'Generate one batch of data'
        # self.V       = np.zeros((self.batch_size,36,2048))
        # self.Q       = np.zeros((self.batch_size,14))
        # self.A       = np.zeros((self.batch_size,3129))
        Indexes = self.train_ids[index*self.batch_size:(index+1)*self.batch_size]
        # print(np.shape(self.V))
        # print(np.shape(Indexes))
        # T= self.feature_index[Indexes]
        # print(np.shape(T))
        # image_features[image_feature_index[Indexes]]
        t = self.image_feature_index[Indexes]
        self.V = self.image_features[self.image_feature_index[Indexes]]
        # print(np.shape(self.V))
        # self.Q = self.questions[Indexes]
        # self.A = self.targets[Indexes]

        return self.V,t,Indexes

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.Len))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)  

class MiniGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, NumData, Features, Questions, Answer, batch_size,shuffle=False):
        'Initialization'
        # self.q_ids = q_ids
        self.Features = Features
        self.batch_size = batch_size
        self.Answer = Answer
        self.Questions = Questions
        self.NumData = NumData
        self.shuffle = shuffle
        self.on_epoch_end()
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        Len = int(np.floor(self.NumData / self.batch_size))
        # print(Len)
        return Len
    
        # return 866

    def __getitem__(self, index):
        'Generate one batch of data'
        # print(index)

        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        imfeatures      = self.Features[indexes]
        question_tokens = self.Questions[indexes]
        answers         = self.Answer[indexes]
        
        return [imfeatures,question_tokens],answers

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.NumData)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)  

class GenerateFromHard(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, NumData, idx, featureH5, Questions, Answer, batch_size,shuffle=False):
        'Initialization'
        # self.q_ids = q_ids
        self.featureH5 = featureH5  
        self.idx = idx  # index to features in image feature vector
        self.batch_size = batch_size
        self.Answer = Answer
        self.Questions = Questions
        self.NumData = NumData
        self.shuffle = shuffle
        self.on_epoch_end()
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        Len = int(np.floor(self.NumData / self.batch_size))
        # print(Len)
        return Len
    
        # return 866

    def __getitem__(self, index):
        'Generate one batch of data'
        # print(index)

        IDX = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        self.imfeatures = np.zeros((self.batch_size,36,2048),dtype=float)
        # read data from hard!
        # print(len(IDX))
        for i,k in enumerate(IDX):
           # print(k)
           self.imfeatures[i] = self.featureH5['image_features'][self.idx[k]]

        # self.Questions       = self.Questions[IDX]
        # self.answers         = self.Answer[IDX]‍
        
        return [self.imfeatures,self.Questions[IDX]],self.Answer[IDX]

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.NumData)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)  





class ValidGenFromHard(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, NumData, idx, featureH5, Questions,question_id,corct_ans, Answer, batch_size,shuffle=False):
        'Initialization'
        # self.q_ids = q_ids
        self.question_id = question_id
        self.featureH5 = featureH5  
        self.idx = idx  # index to features in image feature vector
        self.batch_size = batch_size
        self.Answer = Answer
        self.Questions = Questions
        self.NumData = NumData
        self.shuffle = shuffle
        self.on_epoch_end()
        self.corct_ans = corct_ans
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        Len = int(np.floor(self.NumData / self.batch_size))
        # print(Len)
        return Len
    
        # return 866

    def __getitem__(self, index):
        'Generate one batch of data'
        # print(index)

        IDX = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        self.imfeatures = np.zeros((len(IDX),36,2048),dtype=float)
        # read data from hard!
        # print(len(IDX))
        for i,k in enumerate(IDX):
           # print(k)
           self.imfeatures[i] = self.featureH5['image_features'][self.idx[k]]

        # self.Questions       = self.Questions[IDX]
        # self.answers         = self.Answer[IDX]‍
        
        return [self.imfeatures,self.Questions[IDX]],self.question_id[IDX],self.corct_ans

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.NumData)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)  



'''

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, image_ids, Features, Questions, Answer, batch_size,shuffle=False):
        'Initialization'
        # self.q_ids = q_ids
        self.Features = Features
        self.batch_size = batch_size
        self.Answer = Answer
        self.Questions = Questions
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.on_epoch_end()
        self.image_ids=image_ids
         
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        Len = int(np.floor(len(self.list_IDs) / self.batch_size))
        # print(Len)
        return Len
    
        # return 866

    def __getitem__(self, index):
        'Generate one batch of data'
        # print(index)
        imfeatures      = np.zeros((self.batch_size,36,2048))
        question_tokens = np.zeros((self.batch_size,14))
        answers         = np.zeros((self.batch_size,3129))
        # Q_ids           = np.zeros((self.batch_size,))
        # Generate indexes of the batch
        # tempAns = np.zeros((3129,))
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # self.T.append(indexes)
        # print('indexes...........')
        # list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        for i,k in enumerate(indexes):
            temp                =  self.Features['image_features'][self.image_ids[k]]
            imfeatures[i,]      =  temp
            question_tokens[i,] =  self.Questions[k]

            answers[i,]         = self.Answer[k]
            # Q_ids[i,]=self.q_ids[k]
     
        return [imfeatures,question_tokens],answers,indexes

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)  


class ValidDataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, image_ids, Features, Questions, Answer, batch_size,correct_ans,question_id,shuffle=False):
        'Initialization'
        # self.q_ids = q_ids
        self.Features = Features
        self.batch_size = batch_size
        self.Answer = Answer
        self.Questions = Questions
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.on_epoch_end()
        self.image_ids=image_ids
        self.correct_ans = correct_ans
        self.question_id = question_id 
        # self.indices=indices
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        Len = int(np.floor(len(self.list_IDs) / self.batch_size))
        print(Len)
        return Len
    
        # return 866

    def __getitem__(self, index):
        'Generate one batch of data'
        # print(index)
        imfeatures      = np.zeros((self.batch_size,36,2048))
        question_tokens = np.zeros((self.batch_size,14))
        answers         = np.zeros((self.batch_size,3129))
        Q_ID            = np.zeros((self.batch_size,))
        true_answers           = []
        # Generate indexes of the batch
        # tempAns = np.zeros((3129,))
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # self.T.append(indexes)
        # print('indexes...........')
        # list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        for i,k in enumerate(indexes):
            temp                =  self.Features['image_features'][self.image_ids[k]]
            imfeatures[i,]      =  temp
            question_tokens[i,] =  self.Questions[k]

            answers[i,]         = self.Answer[k]
            true_answers.append(self.correct_ans[k])
            Q_ID[i]=self.question_id[k]
        return [imfeatures,question_tokens],answers,indexes,true_answers,Q_ID

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)  

'''