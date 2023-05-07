#!/usr/bin/env python
# coding: utf-8

# ## importing libraries 

# In[25]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 


# ## importing files 

# In[26]:


df= pd.read_csv('50_Startups.csv')
X=df.iloc[:,:-1].values
y=df.iloc[:,-1].values


# ## one hot encoder to change categorical variables and fitting on X 

# In[27]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct=ColumnTransformer(transformers=[('encoder', OneHotEncoder(),[3])], remainder='passthrough')
X=np.array(ct.fit_transform(X))


# In[28]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.2,random_state=0)


# ## importing linear regression libraries and fitting it 

# In[29]:


from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(X_train, y_train)


# ## Comparing predicted values to the actual values 

# In[30]:


y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# In[ ]:




