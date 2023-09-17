#!/usr/bin/env python
# coding: utf-8

# # CS 4622 - Machine Learning - Lab 01
# 
# ## Feature Engineering

# The task in this lab was to develop a model to predict 4 label outputs, by applying feature engineering techniques.
# We were provided with 2 datasets : "train.csv" and "valid.csv" for this lab.
# 
# Given below is the general outline of how the implementation has been done in this notebook:
# <ol>
# <li> Read given data into pandas dataframe </li> 
# <li> Fill missing values </li> 
# <li> For all 4 labels, </li> 
#     <ol>
#     <li> Train a SVM model for all 256 features </li> 
#     <li> Get accuracy for the trained model </li> 
#     <li> Perform feature selection and train a new model for the reduced number of features </li> 
#     <li> Get accuracy scores for the new model and evaluate the performance </li> 
#     </ol>
# </ol>
#     
# For 'label_4' an additional step of preprocessing was done to oversample the data and ensure an equal distribution.

# In[1]:


# Import necessary libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import RandomOverSampler


# In[2]:


# Read csv data
train = pd.read_csv("train.csv")
valid = pd.read_csv("valid.csv")


# In[3]:


# Display information about the train and valid datasets
train_original = train.copy()
valid_original = valid.copy()

print(train.shape)
print(valid.shape)


# In[4]:


# Display the first few rows of 'train' dataframe
train.head()


# In[5]:


# Display the first few rows of 'valid' dataframe
valid.head()


# In[6]:


# Find out the number of rows with missing values for each label
def check_missing_values(label):
    print("Missing values in {label}: {val}".format(label=label, val=train[label].isna().sum()))

for label in ['label_1', 'label_2', 'label_3', 'label_4']:
    check_missing_values(label)


# In[7]:


# Handling missing values in Label 2
train['label_2'].fillna(train['label_2'].mean().round(), inplace=True)
valid['label_2'].fillna(valid['label_2'].mean().round(), inplace=True)

train = train.astype({'label_2':'int'})
valid = valid.astype({'label_2':'int'})

# Confirm that the values have been filled
train.head()


# In[8]:


# Confirm whether the missing values are all completed now

for label in ['label_1', 'label_2', 'label_3', 'label_4']:
    check_missing_values(label)


# In[9]:


# Separate features and target labels
X_train = train.iloc[:, :256]
y_train = train[['label_1', 'label_2', 'label_3', 'label_4']]

X_valid = valid.iloc[:, :256]
y_valid = valid[['label_1', 'label_2', 'label_3', 'label_4']]

# Check the dimensions to confirm
print(X_train.shape, X_valid.shape)
print(y_train.shape, y_valid.shape)


# ## 1. Modelling Speaker ID
# 
# Speaker ID is labeled as 'label_1' in the datasets. There were no missing values for this label. This can be considered a categorical variable. 
# 
# First, a model will be developed for all 256 features. I have used a SVM classifier for this model.

# In[10]:


svm_model_1 = SVC(kernel='linear', C=1.0, random_state=42)
svm_model_1.fit(X_train, y_train['label_1'])


# In[11]:


# Evaluate the model on valid dataset
from sklearn.metrics import accuracy_score

y_pred_1 = svm_model_1.predict(X_valid)
accuracy = accuracy_score(y_valid['label_1'], y_pred_1)
print("Accuracy:", accuracy)


# Now, some feature engineering techniques will be applied to select the most impactful features from the 256 features. 

# In[12]:


# Feature Selection using SelectKBest and ANOVA F-value test
# k = num of features to select

selector1 = SelectKBest(score_func=f_classif, k=100)
X_train_selected = selector1.fit_transform(X_train, y_train['label_1'])
X_valid_selected = selector1.transform(X_valid)


# In[13]:


# Print the dimensions of X_train_selected
X_train_selected.shape


# In[14]:


# Retrain a new SVM Classifier for the selected feature set

svm_model_1_sel = SVC(kernel='linear', C=1.0, random_state=42)
svm_model_1_sel.fit(X_train_selected, y_train['label_1'])


# In[15]:


# Evaluate the new model on valid dataset
y_pred_1_sel = svm_model_1_sel.predict(X_valid_selected)
accuracy = accuracy_score(y_valid['label_1'], y_pred_1_sel)
print("Accuracy:", accuracy)


# Although the accuracy has dropped a little bit, it was possible to achieve a high accuracy rate with only 100 features. 
# 
# For label_1, <br>
# Initial Model (trained with 256 features) : svm_model_1 <br>
# Model after feature selection : svm_model_1_sel

# ## Modelling Speaker Age
# 
# Speaker age has been labeled as 'label_2'. The missing values were filled earlier with the mean age. 

# In[16]:


svm_model_2 = SVC(kernel='linear', C=1.0, random_state=42)
svm_model_2.fit(X_train, y_train['label_2'])


# In[17]:


y_pred_2 = svm_model_2.predict(X_valid)
accuracy = accuracy_score(y_valid['label_2'], y_pred_2)
print("Accuracy:", accuracy)


# In[18]:


# Feature Selection using SelectKBest and ANOVA F-value test
# k = num of features to select

selector2 = SelectKBest(score_func=f_classif, k=150)
X_train_selected = selector2.fit_transform(X_train, y_train['label_2'])
X_valid_selected = selector2.transform(X_valid)


# In[19]:


# Retrain a new SVM Classifier for the selected feature set

svm_model_2_sel = SVC(kernel='linear', C=1.0, random_state=42)
svm_model_2_sel.fit(X_train_selected, y_train['label_2'])


# In[20]:


# Evaluate the new model on valid dataset
y_pred_2_sel = svm_model_2_sel.predict(X_valid_selected)
accuracy = accuracy_score(y_valid['label_2'], y_pred_2_sel)
print("Accuracy:", accuracy)


# For label_2, <br>
# Initial model : svm_model_2 <br>
# Model after feature selection : svm_model_2_sel

# ## Modelling Speaker Gender
# 
# Speaker gender is labeled as 'label_3'. This is a binary classification.

# In[21]:


# Train an SVM model for all 256 features
svm_model_3 = SVC(kernel='linear', C=1.0, random_state=42)
svm_model_3.fit(X_train, y_train['label_3'])

# Evaluate with the valid dataset
y_pred_3 = svm_model_3.predict(X_valid)
accuracy = accuracy_score(y_valid['label_3'], y_pred_3)
print("Accuracy:", accuracy)


# In[22]:


# Feature Selection using SelectKBest and ANOVA F-value test
# k = num of features to select
selector3 = SelectKBest(score_func=f_classif, k=50)
X_train_selected = selector3.fit_transform(X_train, y_train['label_3'])
X_valid_selected = selector3.transform(X_valid)


# In[23]:


# Retrain a new SVM Classifier for the selected feature set
svm_model_3_sel = SVC(kernel='linear', C=1.0, random_state=42)
svm_model_3_sel.fit(X_train_selected, y_train['label_3'])

# Evaluate the new model on valid dataset
y_pred_3_sel = svm_model_3_sel.predict(X_valid_selected)
accuracy = accuracy_score(y_valid['label_3'], y_pred_3_sel)
print("Accuracy:", accuracy)


# A very high accuracy has been achieved with 50 selected features.
# 
# For label_3, <br>
# Initial model : svm_model_2 <br>
# Model after feature selection : svm_model_2_sel

# ## Modelling Speaker Accent
# 
# Speaker accent is labelled as 'label_4'. The data in this column are not equally distributed. Thus, an oversampling technique has to be applied first before training a model. I have used RandomOverSampler() for this.

# In[24]:


# Oversample the minority class using RandomOverSampler
oversampler = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train, y_train['label_4'])


# For label_4, I developed a KNN model.

# In[25]:


# From the square root rule, get the k value
k = int(X_train.shape[0] ** (1/4))
print(k)


# In[26]:


# build the KNN classifier
knn_model = KNeighborsClassifier(n_neighbors=k)
knn_model.fit(X_train_resampled, y_train_resampled)

# Oversample the valid dataset
X_valid_resampled, y_valid_resampled = oversampler.fit_resample(X_valid, y_valid['label_4'])

y_pred_4 = knn_model.predict(X_valid_resampled)
accuracy = accuracy_score(y_valid_resampled, y_pred_4)
print("Accuracy:", accuracy)


# In[27]:


# Feature Selection using SelectKBest and ANOVA F-value test

selector4 = SelectKBest(score_func=f_classif, k=100)
X_train_selected = selector4.fit_transform(X_train_resampled, y_train_resampled)
X_valid_selected = selector4.transform(X_valid_resampled)


# In[28]:


# Retrain the model

knn_model_sel = KNeighborsClassifier(n_neighbors=k)
knn_model_sel.fit(X_train_selected, y_train_resampled)

y_pred_4_sel = knn_model_sel.predict(X_valid_selected)
accuracy = accuracy_score(y_valid_resampled, y_pred_4_sel)
print("Accuracy:", accuracy)


# With only 100 features 97.6% accuracy has been achieved.
# 
# For label_4, <br>
# Initial model : knn_model, <br>
# Model after feature selection : knn_model_sel

# ## Testing the models
# 
# The accuracy of the above 4 models will be tested with a test dataset : "test.csv"

# In[33]:


# Read test dataset file
test = pd.read_csv("test.csv")


# In[34]:


X_test = test.iloc[:, :256]
print(X_test.shape)


# In[35]:


# Preprocess to input to the models
test_1 = selector1.transform(X_test)
test_2 = selector2.transform(X_test)
test_3 = selector3.transform(X_test)
test_4 = selector4.transform(X_test)


# In[36]:


print(test_1.shape)
print(test_2.shape)
print(test_3.shape)
print(test_4.shape)


# In[38]:


# Predict with the models before feature engineering 
y_test_1_bf = svm_model_1.predict(X_test)
y_test_2_bf = svm_model_2.predict(X_test)
y_test_3_bf = svm_model_3.predict(X_test)
y_test_4_bf = knn_model.predict(X_test)


# In[39]:


# Predict with the models after feature engineering
y_test_1 = svm_model_1_sel.predict(test_1)
y_test_2 = svm_model_2_sel.predict(test_2)
y_test_3 = svm_model_3_sel.predict(test_3)
y_test_4 = knn_model_sel.predict(test_4)


# In[40]:


# Write the transformed datasets into csv files

import csv

header = ["Predicted labels before feature engineering", "Predicted labels after feature engineering", "No of new features"]
for i in range(1, 257):
    header.append("new_feature_" + str(i))

with open("190547P_label_1.csv", "w", newline='') as f1:
    writer = csv.writer(f1)
    writer.writerow(header)
    for i in range(750):
        datarow = [y_test_1_bf[i], y_test_1[i], test_1.shape[1]] + test_1[i].tolist()
        writer.writerow(datarow)

with open("190547P_label_2.csv", "w", newline='') as f2:
    writer = csv.writer(f2)
    writer.writerow(header)
    for i in range(750):
        datarow = [y_test_2_bf[i], y_test_2[i], test_2.shape[1]] + test_2[i].tolist()
        writer.writerow(datarow)
        
with open("190547P_label_3.csv", "w", newline='') as f3:
    writer = csv.writer(f3)
    writer.writerow(header)
    for i in range(750):
        datarow = [y_test_3_bf[i], y_test_3[i], test_3.shape[1]] + test_3[i].tolist()
        writer.writerow(datarow)
        
with open("190547P_label_4.csv", "w", newline='') as f4:
    writer = csv.writer(f4)
    writer.writerow(header)
    for i in range(750):
        datarow = [y_test_4_bf[i], y_test_4[i], test_4.shape[1]] + test_4[i].tolist()
        writer.writerow(datarow)

