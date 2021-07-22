#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[18]:


df = pd.read_csv(r"C:\Users\Venkatesh\Downloads\Iris.csv")


# In[19]:


df.head()


# In[20]:


df.tail()


# In[21]:


df.shape


# In[22]:


df.isnull().sum()


# In[23]:


df.describe()


# In[24]:


df.info()


# In[25]:


df.corr()


# In[26]:


plt.figure(figsize=(20,20))
sns.heatmap(df.corr(),annot=True,cmap='tab20',vmin=0,vmax=1)


# In[27]:


import warnings 
warnings.filterwarnings('ignore')


# In[28]:


a = df.hist(figsize=(10,10) , bins = 10)


# In[29]:


import seaborn as sns
a = df.columns
for i in a:
    sns.distplot(a = df[i])
    plt.show()


# In[32]:


X= df[['Id','SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']].values


# In[33]:


y=df['Species']


# In[35]:



from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


# In[36]:



test = SelectKBest(score_func=chi2, k=5)
fit = test.fit(X, y)

# Summarize scores
np.set_printoptions(precision=5)
print(fit.scores_)

features = fit.transform(X)
# Summarize selected features
print(features[0:10,:])


# In[37]:


from sklearn import preprocessing

X = np.asarray(X)
  
X = preprocessing.StandardScaler().fit(X).transform(X)


# In[38]:


X


# In[39]:


y=df['Species']


# In[40]:


y


# In[41]:


y.unique()


# In[44]:


y.value_counts()


# In[43]:


from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.2 , random_state = 0)


# In[45]:


X_train.shape


# In[47]:


X_test.shape


# In[48]:


X_train


# In[49]:


y_train.shape


# In[51]:


y_test


# In[104]:


from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


# KNN ALGORITHM

# In[105]:


def knn_classifier(X_train, X_test, y_train, y_test):
    classifier_knn = KNeighborsClassifier(metric = 'minkowski', p = 5)
    classifier_knn.fit(X_train, y_train)

    y_pred = classifier_knn.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    
    return print(f"Train score : {classifier_knn.score(X_train, y_train)}\nTest score : {classifier_knn.score(X_test, y_test)}\nAccuracy score:{accuracy_score(y_test,y_pred)}\nCR:{classification_report(y_test,y_pred)}")


# DECISION TREE

# In[106]:


def tree_classifier(X_train, X_test, y_train, y_test):
    
    classifier_tree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
    classifier_tree.fit(X_train, y_train)

    y_pred = classifier_tree.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)

    return print(f"Train score : {classifier_tree.score(X_train, y_train)}\nTest score : {classifier_tree.score(X_test, y_test)}\nAccuracy score :{accuracy_score(y_test, y_pred)}\nCR:{classification_report(y_test,y_pred)}")


# In[107]:


def print_score(X_train, X_test, y_train, y_test):
    print("KNN:\n")
    knn_classifier(X_train, X_test, y_train, y_test)
    
    print("-"*100)
    print()
    
    print("Decision Tree:\n")
    tree_classifier(X_train, X_test, y_train, y_test)
    


# In[108]:


print_score(X_train, X_test, y_train, y_test)


# By performing classification analsyis on iris dataset among KNN algorthm and Decision Tree, KNN algorithm gives the BEST ACCURACY with 98.3 %

# In[ ]:




