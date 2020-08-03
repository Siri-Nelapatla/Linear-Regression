#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt


# In[18]:


dataset = pd.read_csv('Desktop/Salary_Data.csv')
print(dataset.shape)
dataset.head()


# In[20]:


plt.scatter(dataset['YearsExperience'],dataset['Salary'])


# In[22]:


np.corrcoef(dataset['YearsExperience'],dataset['Salary'])[0,1]


# In[23]:


dataset['Salary'].describe()


# In[24]:


dataset.boxplot(column = 'Salary')


# In[25]:


X = dataset['YearsExperience'].values
Y = dataset['Salary'].values


# In[37]:


mean_x = np.mean(X)
mean_y = np.mean(Y)
n = len(X)

numer = 0
denom = 0
for i in range(n):
    numer += (X[i]-mean_x)*(Y[i]-mean_y)
    denom += (X[i]-mean_x)**2
    
b1 = numer/denom

b0 = mean_y-(b1*mean_x)

print(b0,b1)


# In[40]:


y_pred = b0 + b1 * X
plt.plot(X,y_pred,label='regression line')
plt.scatter(X,Y,c='green')
plt.xlabel('Salary')
plt.ylabel('YearsExperience')
plt.legend()
plt.show()


# In[41]:


#cost function

mse=0
for i in range(n):
    y_pred = b0 + b1*X[i]
    mse += (Y[i]-y_pred)**2
rmse = np.sqrt(mse/n)
print(rmse)


# In[43]:


ss_t = 0
ss_r = 0

for i in range(n):
    y_pred=b0+b1*X[i]
    ss_t += (Y[i]-mean_y)**2
    ss_t += (Y[i]-y_pred)**2
    
r2 = 1-(ss_r/ss_t)
print(r2)


# In[46]:


new_x = 10
y_new_pred = b0 +b1*new_x
y_new_pred


# ### Using Ski-kit

# In[51]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# In[54]:


X = X.reshape((n,1))

model = LinearRegression()
model = model.fit(X,Y)
y_pred = model.predict(X)


# In[55]:


model.score(X,Y)


# In[56]:


mse = mean_squared_error(Y,y_pred)
rmse = np.sqrt(mse)
rmse


# In[57]:


model.predict([[10]])


# In[ ]:




