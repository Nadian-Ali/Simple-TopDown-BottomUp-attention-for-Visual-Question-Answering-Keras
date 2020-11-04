

import numpy as np
import keras

class ValidationDataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, image_ids, Features, Questions, Answer, batch_size,q_ids,shuffle=False):
        'Initialization'
        self.q_ids = q_ids
        self.Features = Features
        self.batch_size = batch_size
        self.Answer = Answer
        self.Questions = Questions
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.on_epoch_end()
        self.image_ids=image_ids
        # self.indices=indices
        
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
        Q_ids           = np.zeros((self.batch_size,))
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
            Q_ids[i,]=self.q_ids[k]
     
        return [imfeatures,question_tokens],answers,indexes,Q_ids

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)  


