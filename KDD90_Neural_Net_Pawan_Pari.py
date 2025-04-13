#!/usr/bin/env python
# coding: utf-8

# In[1]:


# NEURAL NET
# Pawan Harikrishnan & Parinita Mithepati 


# In[2]:


# STEP 0 - Install pytorch and its dependencies
get_ipython().system('pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118')


# In[3]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms


# In[5]:


# STEP 1 - Import the training and test files for the dataset 
import pandas as pd 
import numpy as np
# training data import
training_data = pd.read_csv('train_kdd_small.csv')
# testing data import
testing_data = pd.read_csv('test_kdd_small.csv')
# testing and training data do not have the same the distribution


# In[7]:


# check the files to make sure theyre being imported correctly 
#print(training_data)
#print(testing_data)


# In[8]:


# STEP 2 - Dataset needs to be modified as not all the columns have numbers as data type 
# Protocol Type - icmp, tcp 
# Service - ecr_i, http 
# Flag - SF
# Label - not_normal, normal 
from sklearn.preprocessing import LabelEncoder

# create the label encoder 
encoder_string_converted_label = LabelEncoder()

# 1.) encode protocol type where icmp = 0, tcp = 1
encoder_string_converted_label.fit(training_data['protocol_type'])
# map to the testing and training data
training_data['protocol_type'] = encoder_string_converted_label.transform(training_data['protocol_type'])
testing_data['protocol_type'] = encoder_string_converted_label.transform(testing_data['protocol_type'])
# check the data to make sure its getting converted 
#print(training_data['protocol_type'])
#print(testing_data['protocol_type'])

# 2.) encode servicve type where ecr_i = 0, http = 1
encoder_string_converted_label.fit(training_data['service'])
# map to the testing and training data
training_data['service'] = encoder_string_converted_label.transform(training_data['service'])
testing_data['service'] = encoder_string_converted_label.transform(testing_data['service'])
# check the data to make sure its getting converted 
#print(training_data['service'])
#print(testing_data['service'])

# 3.) encode flag type where SF=0
encoder_string_converted_label.fit(training_data['flag'])
# map to the testing and training data
training_data['flag'] = encoder_string_converted_label.transform(training_data['flag'])
testing_data['flag'] = encoder_string_converted_label.transform(testing_data['flag'])
# check the data to make sure its getting converted 
#print(training_data['flag'])
#print(testing_data['flag'])


# 4.) encode label type where not_normal = 0, http = 1
encoder_string_converted_label.fit(training_data['label'])
# map to the testing and training data
training_data['label'] = encoder_string_converted_label.transform(training_data['label'])
testing_data['label'] = encoder_string_converted_label.transform(testing_data['label'])
# check the data to make sure its getting converted 
#print(training_data['label'])
#print(testing_data['label'])


# In[9]:


# STEP 3 - The model now needs to be trained using the training set without the labels
from sklearn.model_selection import train_test_split

# NEW TRAINING FILE
x_training_dropped = training_data.drop('label',axis=1)
y_training_final = training_data['label']
# check to make sure it dropped the last column
#print(x_training_final.shape)
#print(x_training_final.head())
#print(y_training_final.head())

# NEW TESTING FILE
x_testing_dropped = testing_data.drop('label',axis=1)
y_testing_final = testing_data['label']
# check to make sure it dropped the last column
#print(x_testing_final.shape)
#print(x_testing_final.head())
#print(y_testing_final.head())

# VALIDATION SETS
x_training_set, x_validation_set, y_training_set, y_validation_set = train_test_split(x_training_dropped, y_training_final,test_size=0.20,random_state=1,stratify=y_training_final)


# In[10]:


# STEP 4 (OPTIMIZATION STEP) 
# currently the model is too good, so its probably running into overfitting issues 
# this is the model output with no optimization steps

# MODEL OUTPUT
# Model Accuracy : % 100.0
# Model Precision: % 100.0
# Model Recall   : % 100.0
# Model F1 Score : % 100.0

# 1.) add the l2 regularzation penalty  

# 2.) choose a really low value for C 

# 3.) add noice to the training data  (skipping for now)

# 4.) normalize the data 
from sklearn.preprocessing import StandardScaler

# create the scalar first
scaler_logistic_regression = StandardScaler()


# normalize x_training_final and x_testing_final from the last step so some features are not too skewed, need to do fit_transform first or it;ll throw an error
x_training_normalized = scaler_logistic_regression.fit_transform(x_training_set)
x_validation_normalized = scaler_logistic_regression.transform(x_validation_set)
x_testing_normalized  = scaler_logistic_regression.transform(x_testing_dropped)


# In[ ]:


# STEP 5 - Create the neural net class now 
# use the neural net module 
class NeuralNetKDD90(nn.Module):
    # start with the init part 
    def __init__(self, input_size, hidden_size, output_size):
        # first call is to super to set the values 
        super(NeuralNetKDD90,self).__init__() 
        # inputs -> hidden layers + the function to add the bias -> outputs 
        
        # incoming inputs need to get mapped to the first set of hidden layers 
        self.fc1 = nn.Linear(input_size,hidden_size)
        # from these layers the get mapped to the outputs 
        self.fc2 = nn.Linear(hidden_size,output_size)
        
        # use ReLU to decide if the input -> hidden layer -> output path should be there
        self.relu = nn.ReLU()
        
    # now it needs to be forwarded ahead
    def forward(self, x):
        forwarded_x = x
        # output needs to be forwarded along as input --> activation --> output
        # so first the weights get multiplied here and the bias is added here
        forwarded_x = self.fc1(forwarded_x)
        forwarded_x  = self.relu(forwarded_x)
        # weights and bias to the last layer before it goes to the output
        forwarded_x = self.fc2(forwarded_x)
        
        # before returning, x needs to be normalized as done before in logistic regression
        forawrded_x = F.log_softmax(forwarded_x, dim=1)
        return forwarded_x
        
    

