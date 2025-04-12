#!/usr/bin/env python
# coding: utf-8

# In[1]:


# LOGISTIC REGRESSION CLASSIFIER 
# Pawan Harikrishnan & Parinita Mithepati 

# Description:
# This program implements the logisitic regression model to perform binary classification on the KDD90 data set 
# Logistic Regression uses the logistic function to map the prediction between two binary values 0 and 1 
# This program uses the scikit logistic regression model to classify data as either normal or not normal 


# In[2]:


# STEP 1 - Import the training and test files for the dataset 
import pandas as pd 
# training data import
training_data = pd.read_csv('train_kdd_small.csv')
# testing data import
testing_data = pd.read_csv('test_kdd_small.csv')
# testing and training data do not have the same the distribution


# In[10]:


# check the files to make sure theyre being imported correctly 
#print(training_data)
#print(testing_data)


# In[65]:


# STEP 2 - Dataset needs to be modified as not all the columns have numbers as data type 
# Protocol Type - icmp, tcp 
# Service - ecr_i, http 
# Flag - SF
# Label - not_normal, normal 
from sklearn.preprocessing import OrdinalEncoder


# 1.) encode protocol type where icmp = 0, tcp = 1
encoder_string_converted_label = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=2,max_categories=3,dtype=int)
#new_protocol_maps = [['icmp', 0], ['tcp', 1]]
encoder_string_converted_label.fit(training_data[['protocol_type']])
# map to the testing and training data
training_data['protocol_type'] = encoder_string_converted_label.transform(training_data[['protocol_type']])
testing_data['protocol_type'] = encoder_string_converted_label.transform(testing_data[['protocol_type']])
# check the data to make sure its getting converted 
#print(training_data['protocol_type'])
#print(testing_data['protocol_type'])

# 2.) encode service type where ecr_i = 0, http = 1
encoder_string_converted_label = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=2,max_categories=3,dtype=int)
#new_protocol_maps = [['ecr_i', 0], ['http', 1]]
encoder_string_converted_label.fit(training_data[['service']])
# map to the testing and training data
training_data['service'] = encoder_string_converted_label.transform(training_data[['service']])
testing_data['service'] = encoder_string_converted_label.transform(testing_data[['service']])
# check the data to make sure its getting converted 
#print(training_data['service'])
#print(testing_data['service'])

# 3.) encode flag type where SF = 0
encoder_string_converted_label = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=1,max_categories=2,dtype=int)
#new_protocol_maps = [['SF', 0], ['everything else', 1]]
encoder_string_converted_label.fit(training_data[['flag']])
# map to the testing and training data
training_data['flag'] = encoder_string_converted_label.transform(training_data[['flag']])
testing_data['flag'] = encoder_string_converted_label.transform(testing_data[['flag']])
# check the data to make sure its getting converted 
#print(training_data['flag'])
#print(testing_data['flag'])

# 4.) enode label type where not_normal = 0, normal = 1
encoder_string_converted_label = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=2,max_categories=3,dtype=int)
#new_protocol_maps = [['normal', 1], ['not_normal', 0]]
encoder_string_converted_label.fit(training_data[['label']])
# map to the testing and training data
training_data['label'] = encoder_string_converted_label.transform(training_data[['label']])
testing_data['label'] = encoder_string_converted_label.transform(testing_data[['label']])
# check the data to make sure its getting converted 
#print(training_data['label'])
#print(testing_data['label'])


# In[134]:


# STEP 3 - The model now needs to be trained using the training set without the labels
from sklearn.model_selection import train_test_split

# this should be everything in the training set
print("Original training feature shape: ",training_data.shape)
# create the final training set which is everything except the label column feature
x_training_final = training_data.drop(columns=['label'])
# this should be everything except the last column 
print("New training feature shape: ",x_training_final.shape)


# create the final label feature column for the training set
y_training_final = training_data['label']
# this should be just the last column 
print("New final feature shape: ",y_training_final.shape)


# this should be everything in the test set
print("Original testing feature shape: ",testing_data.shape)
# create the final training set which is everything except the label column feature
x_testing_final = testing_data.drop(columns=['label'])
# this should be everything except the last column 
print("New testing feature shape: ",x_testing_final.shape)


# create the final label feature column for the testing set
y_testing_final = testing_data['label']
# this should be just the last column 
print("New final feature shape: ",y_testing_final.shape)


# In[127]:


# STEP 4 (OPTIMIZATION STEP) - Dataset needs to be normalized as some features range from 0-1 and some from 300-1000 
# logisitic regression uses the sigmoid function to classifiy data as either 0 or 1
# when the data range is too big like in src_bytes the data can range from 1032 to 201 
# so this data has to be normalized, normalization will subtract the value from the mean so it will be on a scale from 0 to 1 
# this will make sure that the sigmoid function does not lean towards 1 for values that are too big and 0 for values that are too small 
from sklearn.preprocessing import StandardScaler

# create the scaler
scaler_logistic_regression = StandardScaler()

# normalize Xtrain and Xtest from the last step so some features are not too skewed, need to do fit_transform first or it;ll throw an error
Xtrain_normalized = scaler_logistic_regression.fit_transform(x_training_final)
Xtest_normalized  = scaler_logistic_regression.transform(x_testing_final)

# Standard deviation needs to be about 1, this proved that thje data was normalized 
print(Xtrain_normalized.std())
print(Xtest_normalized.std())



# In[138]:


# STEP 5 - Create and run the model on the modified dataset with the normalized values 
from sklearn.linear_model import LogisticRegression

# set the model type to be the logistic regression model 
model = LogisticRegression(penalty='l2', solver='liblinear', max_iter=1000, random_state=42)

# fit the model using the normalized training and testing created above
model.fit(x_training_final,y_training_final)


# FINAL MODEL PREDICTIONS 
y_pred = model.predict(x_testing_final)


# FINAL SCORES 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("Accuracy :", accuracy_score(y_testing_final, y_pred))
print("Precision:", precision_score(y_testing_final, y_pred))
print("Recall   :", recall_score(y_testing_final, y_pred))
print("F1 Score :", f1_score(y_testing_final, y_pred))


# In[ ]:




