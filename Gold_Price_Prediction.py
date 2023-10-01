#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder,OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt


# In[14]:


data=pd.read_csv('gold_price_data.csv')
data.head()


# In[15]:


data.dtypes


# In[16]:


label_encoder = LabelEncoder()
data['Date'] = label_encoder.fit_transform(data['Date'])
print(data)


# In[17]:


X=data.drop(columns=['GLD'])
y=data['GLD']


# In[18]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
reg_model = RandomForestRegressor(n_estimators=100,random_state=101)
reg_model.fit(X_train,y_train)
y_pred = reg_model.predict(X_test)


# In[19]:


mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Abosulte Error is {mae}")


# In[20]:


plt.figure(figsize=(5,5))
plt.scatter(y_test, y_pred, c='crimson')
p1 = max(max(y_pred), max(y_test))
p2 = min(min(y_pred), min(y_test))
plt.plot([p1, p2], [p1, p2],c='blue')
plt.xlabel('Actual Values', fontsize=15)
plt.ylabel('Predicted Values', fontsize=15)
plt.show()


# In[21]:


g=plt.plot(y_test - y_pred,marker='o',linestyle='-')


# In[ ]:




