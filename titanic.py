#!/usr/bin/env python
# coding: utf-8

# # Imporitng libraries
# 

# In[2]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.ensemble import RandomForestClassifier


# # Importing data

# In[3]:


train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
train_data.head()


# In[4]:


train_data


# In[5]:


test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
test_data.head()


# In[6]:


test_data


# # EDA

# In[18]:


train_data['Survived'].value_counts().plot(kind='bar')


# 

# In[19]:


train_data['Sex'].value_counts().plot(kind='bar')


# In[20]:


train_data['Embarked'].value_counts().plot(kind='bar')


# In[21]:


train_data[['Age', 'Fare']].hist(bins=50, figsize=(12,5))


# # Number of women survived 
# code

# In[7]:


women = train_data[train_data.Sex=="female"].Survived
women


# In[8]:


#rate of women survived
rate_women = sum(women)/len(women)
print("Rate of women survived:", rate_women) 


# In[9]:


#percentage of women survived
rate_women * 100


# In[10]:


men = train_data[train_data.Sex=="male"].Survived
men


# In[11]:


rate_men = sum(men)/len(men)
print("Rate of men survived: ", rate_men)


# In[12]:


target = train_data["Survived"]
features = ["Pclass", "Sex", "SibSp", "Parch","Age"]


# In[13]:


# X-training_data
X = pd.get_dummies(train_data[features].fillna(-1))
# X_test - testing_data
X_test = pd.get_dummies(test_data[features].fillna(-1))


# In[14]:


clf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=1)
clf.fit(X, target)
#creating predictions
predictions = clf.predict(X_test)


# In[15]:


predictions


# In[16]:


output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)


# In[17]:


output.head()


# In[ ]:




