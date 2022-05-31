#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Load Required Libraries 

# In[3]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold


# In[4]:


input_data = pd.read_csv('C:\\Users\\LENOVO LEGION\\Desktop\\Machine Learning\\Untitled Folder\\train.csv')       #Read csv Training file.


# In[5]:


input_data.head()


# In[6]:


print(input_data.shape)


# In[7]:


print(input_data[input_data.columns[1:19]].describe())
input_data[input_data.columns[1:19]].hist(figsize=(20, 20), bins=100, xlabelsize=8, ylabelsize=8);


# In[8]:


X = input_data.values[:, 1:19]
Y = input_data.values[:, 19:]
print(X)


# In[9]:


#X_train,X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)


# In[10]:


kf= KFold(n_splits = 5, random_state = 42, shuffle = True)
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index,],X[test_index]
    Y_train, Y_test = Y[train_index],Y[test_index]


# In[11]:


sc=StandardScaler()
scaler = sc.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[12]:


estimators = [('mlp', MLPClassifier(hidden_layer_sizes=(100,100), random_state = 42,
                        max_iter = 300,activation = 'relu',
                        solver = 'adam')),
('rf',RandomForestClassifier(criterion = 'entropy', random_state=42, max_depth = 16, n_estimators=100, n_jobs=-1))]
clf = VotingClassifier(estimators=estimators, voting = 'soft')
clf.fit(X_train_scaled, Y_train)


# In[ ]:


#While only MLP gives slightly higher accuracy, Voting classifier should have a higher accuracy in a larger dataset.

#clf = MLPClassifier(hidden_layer_sizes=(60,60), random_state = 42,
#                        max_iter = 300,activation = 'relu',
#                        solver = 'adam')
#clf.fit(X_train_scaled, Y_train)


# In[13]:


Y_pred = clf.predict(X_test_scaled)
Y_pred


# In[14]:


print("Accuracy is ", accuracy_score(Y_test,Y_pred))


# In[15]:


test_data = pd.read_csv('C:\\Users\\LENOVO LEGION\\Desktop\\Machine Learning\\Untitled Folder\\test.csv')


# In[16]:


X1 = test_data.values[:, 1:19]


# In[17]:


sc=StandardScaler()
scaler = sc.fit(X1)
X1_test_scaled = scaler.transform(X1)


# In[18]:


preds = clf.predict(X1_test_scaled)


# In[19]:


pd.Series(preds).value_counts()


# In[20]:


preds = pd.DataFrame(preds)


# In[21]:


preds


# In[22]:


preds.rename({0: 'class'}, axis=1, inplace=True)


# In[23]:


print(preds)


# In[24]:


pd.DataFrame(preds).to_csv('submisssion_file_7.csv', index_label='Id')


# In[ ]:




