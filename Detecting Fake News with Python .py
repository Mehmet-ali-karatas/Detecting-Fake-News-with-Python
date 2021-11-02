#!/usr/bin/env python
# coding: utf-8

# In[13]:


# Steps for detecting fake news with Python
# Follow the below steps for detecting fake news and complete your first advanced Python Project –

# The fake news Dataset
#The dataset we’ll use for this python project- we’ll call it news.csv. This dataset has a shape of 7796×4. The first column identifies the news, the second and third are the title and text, and the fourth column has labels denoting whether the news is REAL or FAKE.
#DATA SET 
#https://drive.google.com/file/d/1er9NJTLUA3qnRuyhfzuN0XUsoIC4a-_q/view


# In[9]:


# Make necessary imports:
import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# In[14]:


# Now, let’s read the data into a DataFrame, and get the shape of the data and the first 5 records.

#Read the data
df=pd.read_csv('Desktop/news.csv')

#Get shape and head
df.shape
df.head()


# In[15]:


# And get the labels from the DataFrame.

#DataFlair - Get the labels
labels=df.label
labels.head()


# In[16]:


# Split the dataset into training and testing sets.
#DataFlair - Split the dataset
x_train,x_test,y_train,y_test=train_test_split(df['text'], labels, test_size=0.2, random_state=7)


# In[17]:


# Now, fit and transform the vectorizer on the train set, and transform the vectorizer on the test set.

#DataFlair - Initialize a TfidfVectorizer
tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)

#DataFlair - Fit and transform train set, transform test set
tfidf_train=tfidf_vectorizer.fit_transform(x_train) 
tfidf_test=tfidf_vectorizer.transform(x_test)


# In[18]:


#  Next, we’ll initialize a PassiveAggressiveClassifier. This is. We’ll fit this on tfidf_train and y_train.

# Then, we’ll predict on the test set from the TfidfVectorizer and calculate the accuracy with accuracy_score() from sklearn.metrics.

#DataFlair - Initialize a PassiveAggressiveClassifier
pac=PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)

#DataFlair - Predict on the test set and calculate accuracy
y_pred=pac.predict(tfidf_test)
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')


# In[19]:


# We got an accuracy of 92.9% with this model. Finally, let’s print out a confusion matrix to gain insight into the number of false and true negatives and positives.

#DataFlair - Build confusion matrix
confusion_matrix(y_test,y_pred, labels=['FAKE','REAL'])


# In[ ]:


# So with this model, we have 589 true positives, 588 true negatives, 41 false positives, and 49 false negatives.

# Today, we learned to detect fake news with Python. We took a political dataset, implemented a TfidfVectorizer, initialized a PassiveAggressiveClassifier, and fit our model. We ended up obtaining an accuracy of 92.9% in magnitude.



