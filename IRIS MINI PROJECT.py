#!/usr/bin/env python
# coding: utf-8

# In[109]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[110]:


df = pd.read_csv(r"C:\Users\Venkatesh\Downloads\Iris.csv")


# In[111]:


df.head()


# In[112]:


df.tail()


# In[113]:


df.shape


# In[114]:


df.isnull().sum()


# In[115]:


df.describe()


# In[116]:


df.info()


# In[117]:


df.corr()


# In[118]:


plt.figure(figsize=(20,20))
sns.heatmap(df.corr(),annot=True,cmap='tab20',vmin=0,vmax=1)


# In[119]:


import warnings 
warnings.filterwarnings('ignore')


# In[120]:


a = df.hist(figsize=(10,10) , bins = 10)


# In[121]:


import seaborn as sns
a = df.columns
for i in a:
    sns.distplot(a = df[i])
    plt.show()


# In[122]:


X= df[['Id','SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']].values


# In[123]:


y=df['Species']


# In[124]:



from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


# In[125]:



test = SelectKBest(score_func=chi2, k=5)
fit = test.fit(X, y)

# Summarize scores
np.set_printoptions(precision=5)
print(fit.scores_)

features = fit.transform(X)
# Summarize selected features
print(features[0:10,:])


# In[126]:


from sklearn import preprocessing

X = np.asarray(X)
  
X = preprocessing.StandardScaler().fit(X).transform(X)


# In[127]:


X


# In[128]:


y=df['Species']


# In[129]:


y


# In[130]:


y.unique()


# In[131]:


y.value_counts()


# In[132]:


from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.2 , random_state = 0)


# In[133]:


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




