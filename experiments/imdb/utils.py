import sys
import os
import pickle as pkl
from pathlib import Path

import tqdm
import numpy as np
import scipy.sparse as sp
import networkx as nx

from keras.datasets import imdb
from keras_preprocessing.sequence import pad_sequences

def vectorize2sequences(sequences, dimension=20000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results


def load_data(dataset_str: str, glove_dir: str, num_words: int, maxlen: int):
    
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(
        num_words=num_words
        )

    # inputs
    train_x = pad_sequences(train_data, maxlen)
    test_x = pad_sequences(test_data, maxlen)

    
    val_x = test_x[:10000]
    test_x = test_x[10000:]

    # labels
    train_y = np.zeros(train_labels.shape + (2, ))
    train_y[range(len(train_labels)),train_labels] = 1

    

    val_y = np.zeros((10000, 2))
    val_y[range(10000), test_labels[:10000]] = 1
    test_y = np.zeros((len(test_labels) - 10000, 2))
    test_y[range(len(test_labels) - 10000), test_labels[10000:]] = 1

    

    # embedding
    embedding_index = {}
    with open(os.path.join(glove_dir, 'glove.6B.100d.txt')) as f:

        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype = 'float32')
            embedding_index[word] = coefs
    
    # get word index 
    word_index = imdb.get_word_index()

    # build embedding matrix: num_words * embedding_dim
    embedding_dim = 100 # glove6B setting

    embedding_matrix = np.zeros((num_words, embedding_dim))
    for word, i in word_index.items():
        if i < num_words:
            embedding_vector = embedding_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector


    def Embedding(x):
        x = embedding_matrix[x.reshape(-1)].reshape(x.shape + (embedding_matrix.shape[1],))
        ''''
        out = []
        for i in tqdm.trange(x.shape[0]):
            entry = []
            for j in range(x.shape[1]):
                entry.append(embedding_matrix[x[i, j]])
            out.append(entry)
        '''
        return x
    

    train_x = Embedding(train_x)
    val_x = Embedding(val_x)
    test_x = Embedding(test_x)
    
    ''' debug
    train_x = train_x[:1000]
    train_y = train_y[:1000]
    val_x = val_x[:1000]
    val_y = val_y[:1000]
    test_x = test_x[:1000]
    test_y = test_y[:1000]
    '''
    
    return (train_x, train_y), (val_x, val_y), (test_x, test_y) ,embedding_matrix,


