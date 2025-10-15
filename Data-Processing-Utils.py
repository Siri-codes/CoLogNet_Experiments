#Data Processing Utils

'''
author: @ishapuri101
'''

import numpy as np # linear algebra
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import pandas as pd #loading data in table form
from torch.utils.data import Dataset
import torch # import main library
import torch.nn as nn # import modules
from torch.autograd import Function # import Function to create custom activations
from torch.nn.parameter import Parameter # import Parameter to create custom activations with learnable parameters
import torch.nn.functional as F # import torch functions
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch.optim as optim
#import random
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt


def process_data(first_column_csv, last_column_csv, web_link = None, data_filename = None):
    #data_filename: filename of data source
    #first_column_csv: index (starting from 0) of first column to include in dataset
    #last_column_csv: index (starting from 0) of last column to include in dataset. Use -1 if you want to include all of the columns.

    import pandas as pd

    if web_link is not None:
        pathname = web_link
    else:
        pathname = data_filename #'datasets/' + data_filename
    df=pd.read_csv(pathname, sep=',',header=0, lineterminator='\r')
    if last_column_csv != -1:
        last_column_csv = last_column_csv + 1
    df = df.sample(frac = 1)
    X = df.iloc[:, first_column_csv : last_column_csv].values

    y = df.iloc[:,-1].values.T




    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(y)

    from sklearn.preprocessing import StandardScaler
    sc = MinMaxScaler(feature_range=(0,1))
    X = sc.fit_transform(X)
    X.argmax()

    seeds = [1, 10, 100, 555, 9897]
    seed = seeds[2]

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size = 0.3,
                                                        random_state = seed,
                                                        shuffle = True)
    X_train, X_val, y_train, y_val = train_test_split(X_train,
                                                        y_train,
                                                        test_size=0.05,
                                                        random_state=seed,
                                                        shuffle = True)

    #CONVERTING TO TENSOR
    tensor_x_train = torch.Tensor(X_train)
    tensor_x_val = torch.Tensor(X_val)
    tensor_x_test = torch.Tensor(X_test)

    tensor_y_val = torch.Tensor(y_val).long()
    tensor_y_train = torch.Tensor(y_train).long()
    tensor_y_test = torch.Tensor(y_test).long()

    return tensor_x_train, tensor_y_train, tensor_x_val, tensor_y_val, tensor_x_test, y_test

class OnlyTabularDataset(Dataset):
    def __init__(self, values, label):
        self.values = values
        self.label = label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        return {
            'tabular': self.values[index].clone().detach().float(),
            'target' :  self.label[index].clone().detach().long()
            }