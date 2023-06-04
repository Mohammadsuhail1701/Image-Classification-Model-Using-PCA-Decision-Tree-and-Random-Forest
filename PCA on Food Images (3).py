#!/usr/bin/env python
# coding: utf-8

# # Creating Image Classification Model Using PCA,Decision Tree and Random Forest

# ## Importing Required Libraries 

# In[1]:


import os
import glob
import matplotlib.pyplot as plt
from matplotlib.image import imread
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import pandas as pd
import seaborn as sns
from pathlib import Path
import os.path
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# In[2]:


import PIL
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras import datasets,layers,models
import pathlib
import cv2
from imutils import paths


# ### Python's pathlib module provides a modern and Pythonic way of working with file paths, making code more readable and maintainable. With pathlib , you can represent file paths with dedicated Path objects instead of plain strings.

# In[3]:


image_dir = pathlib.Path("D:/Indian Food Images")


# ### The glob module, which is short for global, is a function that's used to search for files that match a specific file pattern or name. 

# In[4]:


list(image_dir.glob('*/*.jpg'))


# In[5]:


len(list(image_dir.glob('*/*.jpg')))


# ### Visualisation

# In[20]:


adhirasam = list(image_dir.glob('adhirasam/*'))
adhirasam[:5]


# In[21]:


PIL.Image.open(str(adhirasam[3]))


# In[22]:


aloo_gobi = list(image_dir.glob('aloo_gobi/*'))
aloo_gobi[:3]


# In[23]:


PIL.Image.open(str(aloo_gobi[20]))


# ### Map in Python is a function that works as an iterator to return a result after applying a function to every item of an iterable (tuple, lists, etc.). It is used when you want to apply a single transformation function to all the iterable elements. The iterable and function are passed as arguments to the map in Python.

# ### Python OS module provides the facility to establish the interaction between the user and the operating system. It offers many useful OS functions that are used to perform OS-based tasks and get related information about operating system. The OS comes under Python's standard utility modules.

# In[6]:


filepath = list(image_dir.glob(r'**/*.jpg'))
label = list(map(lambda x : os.path.split(os.path.split(x)[0])[1], filepath))


# ### Creating DataFrame Using Filepath(image paths) as one column and label(names of images) as other column Using Pandas  

# In[7]:


filepath = pd.Series(filepath, name = 'Filepath').astype(str)
label = pd.Series(label, name = 'Label')
image_df = pd.concat([filepath, label], axis = 1).sample(frac = 1.0, random_state = 1).reset_index(drop = True)
image_df.head()


# In[8]:


len(image_df['Label'].unique())


# In[9]:


image_df.shape


# In[10]:


image_df


# In[11]:


data1=pd.DataFrame(image_df)


# In[12]:


data1


# ### LabelEncoder can be used to normalize labels. It can also be used to transform non-numerical labels (as long as they are hashable and comparable) to numerical labels.

# In[13]:


le = LabelEncoder()
image_df['Label'] = le.fit_transform(image_df['Label'])
image_df['Label'].value_counts()


# In[14]:


image_df['Filepath']= le.fit_transform(image_df['Filepath'])
image_df['Filepath'].value_counts()


# In[15]:


image_df.head(10)


# ### Assigning X and Y 

# In[16]:


x=image_df['Filepath']
y=image_df['Label']


# In[17]:


x


# In[18]:


y


# ### Splitting x and y into x_train, x_test and y_train, y_test Ising Train_Test_Splitter From Sklearn.Model selection

# In[19]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(image_df,y,train_size=0.8,random_state=90)


# In[20]:


x_train.shape


# In[21]:


x_test.shape


# ### Importing PCA from Sklearn

# In[22]:


from sklearn.decomposition import PCA


# ### Fitting and Transforming X_Train and X_Test into PCA Using 90% Accuracy Components

# In[23]:


pca=PCA(n_components=0.90)
pca_x_train=pca.fit_transform(x_train)
pca_x_test=pca.transform(x_test)


# ### Importing DecisionTree Classifier and Fitting x_train and x_test into it.

# #### Decision Trees (DTs) are a non-parametric supervised learning method used for classification and regression. The goal is to create a model that predicts the value of a target variable by learning simple decision rules inferred from the data features.

# In[25]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import*


# In[28]:


dt=DecisionTreeClassifier().fit(pca_x_train,y_train)


# In[29]:


predicted=dt.predict(pca_x_test)


# ### Checking Predicted Values 

# In[30]:


predicted


# ### Actual Values 

# In[31]:


np.array(y_test)


# ### Checking Accuracy Score

# In[32]:


accuracy_score(predicted,y_test)


# ### Creating a Function for Misclassified Images

# In[33]:


def get_misclassified_index(y_pred,y_test):
    misclassification=[]
    for index,(predicted,actual) in enumerate(zip(y_pred,y_test)):
        if predicted!=actual:
            misclassification.append(index)
    return misclassification


# In[34]:


misclassification=get_misclassified_index(predicted,y_test)


# In[35]:


len(misclassification)


# In[36]:


misclassification[:]


# In[52]:


# CLASSIFICATION REPORT

print("Classification Reporrt: \n", classification_report(y_test,predicted))


# ### As We Get Accuracy 0.98875 around 98% which is good. But to Improve the Accuracy We are going to Check RandomForestClassifier

# #### Random forest is a supervised Machine Learning algorithm. This algorithm creates a set of decision trees from a few randomly selected subsets of the training set and picks predictions from each tree. Then by means of voting, the random forest algorithm selects the best solution.

# In[37]:


from sklearn.ensemble import RandomForestClassifier


# In[38]:


rf = RandomForestClassifier(n_estimators = 1000)


# In[50]:


rf=rf.fit(x_train,y_train)


# In[51]:


rf_pred = rf.predict(x_test)


# ### Checking the Predictions Given by RandomForestClassifier 

# In[52]:


rf_pred


# In[53]:


len(rf_pred)


# ### Actual Values

# In[54]:


np.array(y_test)


# ### Checking Accuracy Score of RandomForestClassifier 

# In[56]:


accuracy_score(rf_pred,y_test)


# In[57]:


def get_misclassified_index(rf_pred,y_test):
    misclassification=[]
    for index,(predicted,actual) in enumerate(zip(rf_pred,y_test)):
        if predicted!=actual:
            misclassification.append(index)
    return misclassification


# In[58]:


misclassification=get_misclassified_index(rf_pred,y_test)


# In[59]:


len(misclassification)


# In[60]:


misclassification[:]


# ### CONCLUSION

# #### When we used DECISIONTREE CLASSIFIER we got 98% accuracy which is good, but when we check the misclassification report using defined function we got 9 misclassified indices which the model predicted wrong.
# #### To Get less misclassified images and more accuracy we use RANDOMFOREST  CLASSIFIER to improve the model.When We used RF model and predicted the values based on that we got 99.8% Accuracy which is very good.When we check Misclassified Indices we got only '1' Misclassified Index.
# #### By Which We Can Conclude that Both models are working good to predict the output without any overfitting and  underfitting.

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




