#!/usr/bin/env python
# coding: utf-8

# In[1]:


# PART 1
# ------------------------------------------------------------------------------------------------------------------------
# LOGISTIC REGRESSION CLASSIFIER 
# Pawan Harikrishnan & Parinita Mithepati 

# Description:
# This program implements the logisitic regression model to perform binary classification on the KDD90 data set 
# Logistic Regression uses the logistic function to map the prediction between two binary values 0 and 1 
# This program uses the scikit logistic regression model to classify data as either normal or not normal 


# In[3]:


# STEP 1 - Import the training and test files for the dataset 
import pandas as pd 
import numpy as np
# training data import
training_data = pd.read_csv('train_kdd_small.csv')
# testing data import
testing_data = pd.read_csv('test_kdd_small.csv')
# testing and training data do not have the same the distribution

# check the files to make sure theyre being imported correctly 
#print(training_data)
#print(testing_data)


# In[4]:


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


# 4.) encode label type where not_normal = 0, normal = 1
encoder_string_converted_label.fit(training_data['label'])
# map to the testing and training data
training_data['label'] = encoder_string_converted_label.transform(training_data['label'])
testing_data['label'] = encoder_string_converted_label.transform(testing_data['label'])
# check the data to make sure its getting converted 
#print(training_data['label'])
#print(testing_data['label'])


# In[5]:


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
x_training_set, x_validation_set, y_training_set, y_validation_set = train_test_split(x_training_dropped, y_training_final,test_size=0.20,random_state=42,stratify=y_training_final)


# In[6]:


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


# In[7]:


# STEP 5 - Create and run the model on the modified dataset with the normalized values 
from sklearn.linear_model import LogisticRegression

# set the model type to be the logistic regression model 
model = LogisticRegression(penalty='l2',C=0.0001, solver='liblinear',random_state=42) 

# fit the model using the normalized training and testing created above
model.fit(x_training_normalized,y_training_set)

# FINAL VALIDATION PREDICTIONS
y_final_validation_prediction = model.predict(x_validation_normalized)

# FINAL MODEL PREDICTIONS 
y_final_prediction = model.predict(x_testing_normalized)


# FINAL SCORES 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Validation scores
model_validation_accuracy = accuracy_score(y_validation_set, y_final_validation_prediction)
model_validation_precision = precision_score(y_validation_set, y_final_validation_prediction)
model_validation_recall = recall_score(y_validation_set, y_final_validation_prediction)
model_validation_f1_score = f1_score(y_validation_set, y_final_validation_prediction)

# Test scores 
model_accuracy = accuracy_score(y_testing_final, y_final_prediction)
model_precision = precision_score(y_testing_final, y_final_prediction)
model_recall = recall_score(y_testing_final, y_final_prediction)
model_f1_score = f1_score(y_testing_final, y_final_prediction)

print("Final Validation Metrics: ")

print("Model Accuracy : %", model_validation_accuracy*100)
print("Model Precision: %", model_validation_precision*100)
print("Model Recall   : %", model_validation_recall*100)
print("Model F1 Score : %", model_validation_f1_score*100)

print("The validation error is: %", (1-model_validation_accuracy)*100)

print("Final Model Metrics: ")

print("Model Accuracy : %", model_accuracy*100)
print("Model Precision: %", model_precision*100)
print("Model Recall   : %", model_recall*100)
print("Model F1 Score : %", model_f1_score*100)

print("The testing error is: %", (1-model_accuracy)*100)


# In[8]:


# SVM CLASSIFIER 
# Pawan Harikrishnan & Parinita Mithepati 

# Description:
# This program implements the SVM (Support Vector Machine) model to perform binary classification on the KDD90 data set 
# SVM will find a split on the plane containing the data, so it find a plane that splits the data into two parts 
# This program uses the scikit SVM model to classify data as either normal or not normal 


# In[9]:


# STEP 1 - Import the training and test files for the dataset 
import pandas as pd 
import numpy as np
# training data import
training_data = pd.read_csv('train_kdd_small.csv')
# testing data import
testing_data = pd.read_csv('test_kdd_small.csv')
# testing and training data do not have the same the distribution

# check the files to make sure theyre being imported correctly 
#print(training_data)
#print(testing_data)


# In[10]:


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


# 4.) encode label type where not_normal = 0, normal = 1
encoder_string_converted_label.fit(training_data['label'])
# map to the testing and training data
training_data['label'] = encoder_string_converted_label.transform(training_data['label'])
testing_data['label'] = encoder_string_converted_label.transform(testing_data['label'])
# check the data to make sure its getting converted 
#print(training_data['label'])
#print(testing_data['label'])


# In[11]:


# STEP 3 - The model now needs to be trained using the training set without the labels
from sklearn.model_selection import train_test_split

# NEW TRAINING FILE
x_training_final = training_data.drop('label',axis=1)
y_training_final = training_data['label']
# check to make sure it dropped the last column
#print(x_training_final.shape)
#print(x_training_final.head())
#print(y_training_final.head())

# NEW TESTING FILE
x_testing_final = testing_data.drop('label',axis=1)
y_testing_final = testing_data['label']
# check to make sure it dropped the last column
#print(x_testing_final.shape)
#print(x_testing_final.head())
#print(y_testing_final.head())


# In[12]:


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
scaler_svm = StandardScaler()


# normalize x_training_final and x_testing_final from the last step so some features are not too skewed, need to do fit_transform first or it;ll throw an error
x_training_normalized = scaler_svm.fit_transform(x_training_final)
x_testing_normalized  = scaler_svm.transform(x_testing_final)


# In[13]:


# STEP 5 - Create and run the model on the modified dataset with the normalized values 
from sklearn import svm
# set the model type to be the SVM model
model = svm.SVC(kernel='rbf', C=0.1)

# fit the model using the normalized training and testing created above
model.fit(x_training_normalized,y_training_final)

# FINAL MODEL PREDICTIONS 
y_final_prediction = model.predict(x_testing_normalized)


# FINAL SCORES 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

model_accuracy = accuracy_score(y_testing_final, y_final_prediction)
model_precision = precision_score(y_testing_final, y_final_prediction)
model_recall = recall_score(y_testing_final, y_final_prediction)
model_f1_score = f1_score(y_testing_final, y_final_prediction)\

print("Final Model Metrics: ")

print("Model Accuracy : %", model_accuracy*100)
print("Model Precision: %", model_precision*100)
print("Model Recall   : %", model_recall*100)
print("Model F1 Score : %", model_f1_score*100)


# In[14]:


# Random Forests CLASSIFIER 
# Pawan Harikrishnan & Parinita Mithepati 

# Description:
# This program implements the Random Forests model to perform binary classification on the KDD90 data set 
# Random Forests use decision tress to keep splitting on a feature to classify data
# This program uses the scikit random forests model to classify data as either normal or not normal 


# In[15]:


# STEP 1 - Import the training and test files for the dataset 
import pandas as pd 
import numpy as np
# training data import
training_data = pd.read_csv('train_kdd_small.csv')
# testing data import
testing_data = pd.read_csv('test_kdd_small.csv')
# testing and training data do not have the same the distribution

# check the files to make sure theyre being imported correctly 
#print(training_data)
#print(testing_data)


# In[16]:


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


# 4.) encode label type where not_normal = 0, normal = 1
encoder_string_converted_label.fit(training_data['label'])
# map to the testing and training data
training_data['label'] = encoder_string_converted_label.transform(training_data['label'])
testing_data['label'] = encoder_string_converted_label.transform(testing_data['label'])
# check the data to make sure its getting converted 
#print(training_data['label'])
#print(testing_data['label'])


# In[17]:


# STEP 3 - The model now needs to be trained using the training set without the labels
from sklearn.model_selection import train_test_split

# NEW TRAINING FILE
x_training_final = training_data.drop('label',axis=1)
y_training_final = training_data['label']
# check to make sure it dropped the last column
#print(x_training_final.shape)
#print(x_training_final.head())
#print(y_training_final.head())

# NEW TESTING FILE
x_testing_final = testing_data.drop('label',axis=1)
y_testing_final = testing_data['label']
# check to make sure it dropped the last column
#print(x_testing_final.shape)
#print(x_testing_final.head())
#print(y_testing_final.head())


# In[18]:


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
scaler_random_forests = StandardScaler()


# normalize x_training_final and x_testing_final from the last step so some features are not too skewed, need to do fit_transform first or it;ll throw an error
x_training_normalized = scaler_random_forests.fit_transform(x_training_final)
x_testing_normalized  = scaler_random_forests.transform(x_testing_final)


# In[19]:


# STEP 5 - Create and run the model on the modified dataset with the normalized values 
from sklearn.ensemble import RandomForestClassifier
# set the model type to be the SVM model
model = RandomForestClassifier(max_depth=100,random_state=42,n_estimators=1000,min_samples_leaf=900)

# fit the model using the normalized training and testing created above
model.fit(x_training_normalized,y_training_final)

# FINAL MODEL PREDICTIONS 
y_final_prediction = model.predict(x_testing_normalized)


# FINAL SCORES 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

model_accuracy = accuracy_score(y_testing_final, y_final_prediction)
model_precision = precision_score(y_testing_final, y_final_prediction)
model_recall = recall_score(y_testing_final, y_final_prediction)
model_f1_score = f1_score(y_testing_final, y_final_prediction)\

print("Final Model Metrics: ")

print("Model Accuracy : %", model_accuracy*100)
print("Model Precision: %", model_precision*100)
print("Model Recall   : %", model_recall*100)
print("Model F1 Score : %", model_f1_score*100)


# In[20]:


# PART 2 
# ---------------------------------------------------------------------------------------------------------------------------
# NEURAL NET
# Pawan Harikrishnan & Parinita Mithepati 


# In[21]:


# STEP 0 - Install pytorch and its dependencies
get_ipython().system('pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118')


# In[22]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms


# In[23]:


# STEP 1 - Import the training and test files for the dataset 
import pandas as pd 
import numpy as np
# training data import
training_data = pd.read_csv('train_kdd_small.csv')
# testing data import
testing_data = pd.read_csv('test_kdd_small.csv')
# testing and training data do not have the same the distribution


# In[24]:


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


# 4.) encode label type where not_normal = 0, normal= 1
encoder_string_converted_label.fit(training_data['label'])
# map to the testing and training data
training_data['label'] = encoder_string_converted_label.transform(training_data['label'])
testing_data['label'] = encoder_string_converted_label.transform(testing_data['label'])
# check the data to make sure its getting converted 
#print(training_data['label'])
#print(testing_data['label'])


# In[25]:


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


# In[26]:


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


# In[27]:


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
        forward_x = self.fc2(forward_x)
        return x


# In[ ]:




