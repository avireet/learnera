#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
housing= fetch_california_housing()


# In[2]:


housing


# In[3]:


housing.keys()


# In[5]:


x= pd.DataFrame(housing.data , columns =housing.feature_names)


# In[6]:


y= pd.DataFrame(housing.target, columns=housing.target_names)


# In[7]:


x.head()


# In[8]:


y.head()


# In[9]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


# In[10]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=69)


# In[11]:


model= LinearRegression()
model


# In[12]:


model.fit(x_train,y_train)


# In[13]:


pred = model.predict(x_test)


# In[14]:


from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error


# In[15]:


r2_score(y_test,pred)


# In[16]:


mean_absolute_error(y_test,pred)


# In[17]:


mean_squared_error(y_test,pred)


# In[18]:


scaler = StandardScaler()

x_train_scaler = scaler.fit_transform(x_train)

x_test_scaler = scaler.transform(x_test)


# In[19]:


model.fit(x_train_scaler,y_train)


# In[22]:


pred_ = model.predict(x_test_scaler)


# In[23]:


r2_score(y_test,pred_)


# In[ ]:




