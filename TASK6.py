#!/usr/bin/env python
# coding: utf-8

# # NAME-SONALI PATIL
# # TASK 6- -Prediction using Decision Tree Algorithm(Level - Intermediate)

# In[16]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree
from sklearn import tree
import seaborn as sns


# In[2]:


#import dataset
data = load_iris()
df = pd.DataFrame(data.data, columns = data.feature_names)
df['target'] = data.target


# In[3]:


#checking for null values
df.isnull().sum()


# In[4]:


# Showing first 5 values
df.head()


# In[5]:


df.tail()


# In[7]:


df.shape


# In[8]:


df.columns


# In[11]:


df.info()


# In[12]:


df.describe()


# In[18]:


# Showing only target data (Dependent Variable)
print(df['target'])


# In[19]:


# splitting data

fc = [x for x in df.columns if x!="target"]
x= df[fc]
y= df["target"]
X_train, X_test, Y_train, Y_test = train_test_split(x,y, random_state = 100, test_size = 0.30)


# In[20]:


# Display of data
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# In[21]:



#building Desicion tree model

model1 = DecisionTreeClassifier()


# In[22]:



model1.fit(X_train,Y_train)


# In[23]:



Y_pred = model1.predict(X_test)


# In[24]:



data2 = pd.DataFrame({"Actual":Y_test,"Predicted":Y_pred})
data2.head()


# In[25]:


# Testing the accuracy of model prediction
accuracy_score(Y_test,Y_pred)


# In[26]:


# Plotting
f_n = ["Sepal length", "Sepal width", "Petal length", "Petal width"]
c_n = ["Setosa", "Versicolor", "Virginica"]
plot_tree(model1,feature_names = f_n, class_names = c_n , filled = True)


# In[27]:



modelx= DecisionTreeClassifier().fit(x,y)


# In[28]:


plt.figure(figsize = (20,15))
tree = tree.plot_tree(modelx, feature_names = f_n, class_names = c_n, filled = True)


# In[ ]:




