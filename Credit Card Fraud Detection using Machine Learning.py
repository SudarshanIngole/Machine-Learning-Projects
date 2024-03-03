#!/usr/bin/env python
# coding: utf-8

# # Importing the Important Libraries

# In[5]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[6]:


data=pd.read_csv('creditcard.csv')


# In[7]:


data.head()


# In[8]:


data


# In[9]:


data.tail()


# In[10]:


# Dataset information
data.info()


# In[11]:


#checking the number of missing column
data.isnull().sum()


# In[14]:


#distribution of legit transactions and fraudulent transactions
data['Class'].value_counts()


# In[15]:


# This dataset is highly unbalanced
#  0 --> Normal Transactions
#  1 --> Fraudulent Transactions


# In[23]:


#separating the Data for Analysis 
legit  = data[data.Class == 0]

fraud = data[data.Class == 1]


# In[25]:


legit['Class'].value_counts()


# In[26]:


fraud


# In[28]:


print(legit.shape)
print(fraud.shape)


# In[29]:


legit.Amount.describe()


# In[30]:


fraud.Amount.describe()


# In[33]:


# Compare the Values for Both Transactions

data.groupby('Class').mean()


# In[34]:


# Number of Fraudulent Transactions = 492
# using Random Sampling take 492 random values from legit dataset


# In[35]:


legit_sample= legit.sample(n=492)


# In[36]:


legit_sample


# In[43]:


new_data=pd.concat([legit_sample, fraud], axis=0)


# In[44]:


new_data['Class'].value_counts()


# In[45]:


new_data.head()


# In[46]:


new_data.groupby('Class').mean()


# In[47]:


# Splitting the Data into features and Target


# In[49]:


x= new_data.drop(columns='Class', axis=1)
y=new_data['Class']


# In[50]:


x


# In[51]:


y


# In[52]:


# Split the data into Trainnig data and Testing Data


# In[57]:


x_train, x_test, y_train, y_test=train_test_split(x,y, test_size=0.2, random_state=10, stratify = y)


# In[58]:


y_test


# In[61]:


print(x.shape, x_train.shape, x_test.shape, y_train.shape)


# In[63]:


model=LogisticRegression()


# In[64]:


model.fit(x_train, y_train)


# In[65]:


# Model Evaluation


# In[67]:


x_train_predict = model.predict(x_train)
accuracy_score(x_train_predict, y_train)


# In[68]:


x_test_predict = model.predict(x_test)
accuracy_score(x_test_predict, y_test)


# In[69]:


# The Accuracy Score is about 92% which is a very good score


# In[ ]:




