#!/usr/bin/env python
# coding: utf-8

# For this project I'll be using data obatined from Kaggle that will be used to predict the outcome of a response. 
# The dataset for this project can be found at the following site. 
# 
# Marketing Campaign Positive Response Prediction
# https://www.kaggle.com/datasets/sujithmandala/marketing-campaign-positive-response-prediction
# 
# 

# Importing the necessary packages that could be needed for this project. 

# In[502]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[503]:


hf = pd.read_csv('campaign_responses.csv')
hf


# In[504]:


hf.info()


# In[505]:


hf.describe()


# In[506]:


hf.isna().sum()


# In[507]:


hf['responded'] = hf['responded'].replace(['Yes', 'No'], [1, 0])


# In[508]:


hf


# In[509]:


sns.histplot(hf, x = 'age')
plt.xlabel('Patient Ages')
plt.ylabel('Total Count')
plt.title('Total Age Count')


# In[510]:


sns.histplot(hf, x = 'annual_income')
plt.xlabel('Annual Income')
plt.ylabel('Total Count')
plt.title('Annual Income Count')


# In[511]:


sns.histplot(hf, x = 'credit_score')
plt.xlabel('Credit Score')
plt.ylabel('Total Count')
plt.title('Credit Score Count')


# In[512]:


corr = hf.corr()


# In[513]:


plt.figure(figsize = (10, 5))
sns.heatmap(corr, cmap = 'coolwarm', annot = True)
plt.show()


# In[514]:


hf_x = hf.drop(columns = 'responded')
hf_x.info()


# In[515]:


hf_x.shape


# In[516]:


hf_x = pd.get_dummies(hf_x)


# In[517]:


hf_x.head()


# In[518]:


hf_y = hf.iloc[:, -1].values
hf_y


# In[519]:


hf_y.shape


# In[520]:


hf_y1 = pd.DataFrame()


# In[521]:


hf_y1 = hf_y


# In[522]:


print(type(hf_y1))


# In[523]:


hf_y


# In[524]:


x_train, x_test, y_train, y_test = train_test_split(hf_x, hf_y, test_size = .30, random_state = 58)


# In[525]:


log = LogisticRegression()
log


# In[526]:


log.fit(x_train, y_train)


# In[527]:


x_train.head()


# In[528]:


y_test


# In[529]:


y_pred = log.predict(x_test)


# In[530]:


print(log.coef_)


# In[531]:


print(log.intercept_)


# In[537]:


accuracy_score(y_test, y_pred, normalize = True)


# In[ ]:




