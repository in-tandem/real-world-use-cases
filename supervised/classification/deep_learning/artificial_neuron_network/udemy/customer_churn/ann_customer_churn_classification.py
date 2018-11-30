# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plot
from sklearn.model_selection import train_test_split
import keras
import pandas as panda
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


PATH = "C:\\Users\\somak\\Documents\\somak_python\\real-world-use-cases\\supervised\\classification\\deep_learning\\artificial_neuron_network\\udemy\\customer_churn\\Churn_Modelling.csv"


def get_data(fully_qualified_file_path):
    
    data = panda.read_csv(fully_qualified_file_path)
    
    print('Data found. Size of data is : ', data.shape)
    print('Columns in data: ', data.columns.tolist())
    
    return data

def get_target_variable(data, column_name):
    
    y = data[column_name]
    
#    another way to extract data : y = data.iloc[:, 13].values (data.iloc[:, 13]  is a pandas.core.series.Series object)
    
    print("shape of target: " , y.shape)
    
    print(y.value_counts())
    ## we see there are two unique values and they are numerical. encoding not required
    ## however classes are imbalanced. we would be using our f1 scores for better evaluation of models
    
    return y

def view_prepare_independent_variables(data):
    
    ## initial understanding would be to take all columns except the target columns
    x = data[data.columns.tolist()[:-1]] ## target column is at the end , ignoring it.
    
    
    print("Description of selected independent variables: ", x.info)
    
    ##we can ignore the columns RowNumber	CustomerId	Surname
    ## we would have to encode the data for the columns Geography and Gender
    
    removed_columns = ['RowNumber', 'CustomerId', 'Surname']
    columns = [i for i in x.columns.tolist() if i not in removed_columns]
    
    x = x[columns]
    
    print(x.head(5))
    
    label_encoder = LabelEncoder()
    
    ## can also be done using apply function on pd dataframe
    ## x[2] = x[2].apply(lambda x : 1 if x=='male' else 0)
    
    x['Geography'] = label_encoder.fit_transform(x['Geography'])
    x['Gender'] = label_encoder.fit_transform(x['Gender']) 
    
    print(x.head(), np.unique(x['Geography'].values))
    
    ##in order to prevent falling into dummy variable trap, we would one hot encode
    ## one hot encoding wil return a numpy array, which we can feed into our models laters

    one_hot_encoder = OneHotEncoder(categorical_features=[1]) ## we only want to encode the geography columns

    x = one_hot_encoder.fit_transform(x).toarray()[:, 1:]

    print(x.shape)    
    
    return x



def run():
    """
    
    Steps undertaken would be as follows:
        
        1. Get the data from the remote file
        2. Check the target of the data. Count the distributions. Encode to numerical if necessary
        3. Check the dependent variables. Select the ones that make sense. ENcode,if necessary
        4. train_test_Split
        5. standard scaling
        6. run it through models        
        
    """
    
    data = get_data(PATH)
    
    y = get_target_variable(data = data, column_name = 'Exited')
    
    x = view_prepare_independent_variables(data)
    
    
    
    
    
run()


    
        