#!/usr/bin/env python
# coding: utf-8

# In[76]:


print('*******Flight_Data_Project********')


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
import seaborn as sn
from sklearn.metrics import confusion_matrix
import re
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import model_selection, preprocessing, naive_bayes,metrics
from sklearn.feature_extraction.text import TfidfVectorizer


# In[4]:


df_flights = pd.read_csv('/Users/jaswanthjerripothula/Downloads/flights_data.csv')
print(df_flights.shape)


# In[5]:


df_flights.head(11)


# In[6]:


df_flights.tail(11)


# In[12]:


df_flights.count()


# In[13]:


df_flights.groupby('Airline').describe()


# In[14]:


df_flights.groupby('Airline').count()


# In[15]:


category_count = pd.DataFrame()
category_count['count'] = df_flights['Airline'].value_counts()


# In[16]:


category_count['count']


# In[36]:


fig , ax = plt.subplots(figsize = (14,6))
sn.barplot (x = category_count.index , y = category_count['count'],ax = ax)
ax.set_ylabel ('Number of Flights',fontsize = 12)
ax.set_xlabel ('Airlines',fontsize = 12)
ax.tick_params(labelsize = 10)
plt.xticks(rotation = 90)


# In[26]:


sn.countplot(data = df_flights, x = 'Source')
plt.ylabel('Number of Flights',fontsize = 10)
plt.xlabel('Source',fontsize = 10)
ax.tick_params(labelsize = 10)


# In[27]:


base_color = sn.color_palette()[0]
sn.countplot(data = df_flights, x = 'Source', color = base_color)
plt.xticks(rotation = 30)


# In[28]:


base_color = sn.color_palette()[1]
general_order = df_flights['Source'].value_counts().index
sn.countplot(data = df_flights, x = 'Source', color = base_color, order = general_order)


# In[38]:


base_color = sn.color_palette()[2]
sn.countplot(data = df_flights, x = 'Airline', color = base_color)
plt.xticks(rotation = 270)


# In[40]:


base_color = sn.color_palette()[3]
sn.countplot(data = df_flights, x = 'Airline', color = base_color)
plt.xticks(rotation = 90)


# In[41]:


df_flights.isna().sum()


# In[45]:


ns_counts = df_flights.isna().sum()
base_color = sn.color_palette()[0]
sn.barplot(ns_counts.index.values, ns_counts, color = base_color)
plt.ylabel('Number of missing values',fontsize = 10)
plt.xticks(rotation = 90)


# In[47]:


#Preprocessing Tweets for removing punctuations(! , '), @ , # , https , special characters
def processMessage(tweet):
    from string import punctuation
    tweet = re.sub(r'\&\w*;','',tweet)
    tweet = re.sub('@[^\s]+','',tweet)
    tweet = re.sub(r'\$\w*','',tweet)
    tweet = tweet.lower()
    tweet = re.sub(r'https?:\/\/.*\/\w*','',tweet)
    tweet = re.sub(r'#\w*','',tweet)
    tweet = re.sub(r'['+punctuation.replace('@','')+']+','',tweet)
    tweet = re.sub(r'\b\w{1,2}\b','',tweet)
    tweet = re.sub(r'\s\s+',' ',tweet)
    tweet = tweet.lstrip(' ')
    tweet = ''.join(c for c in tweet if c <= '\uFFFF')
    return tweet


# In[58]:


sort_counts = df_flights['Destination'].value_counts()
plt.pie(sort_counts, labels = sort_counts.index, startangle = 90, counterclock = False);
plt.axis('square')
plt.title('Flight Destination\n')


# In[59]:


sort_counts = df_flights['Source'].value_counts()
plt.pie(sort_counts, labels = sort_counts.index, startangle = 90, counterclock = False);
plt.axis('square')
plt.title('Flight Source\n')


# In[60]:


plt.hist(data = df_flights, x = 'Duration(minutes)')


# In[65]:


plt.hist(data = df_flights, x = 'Price', bins = 30)


# In[74]:


sort_counts = df_flights['Total_Stops'].value_counts()
plt.pie(sort_counts, labels = sort_counts.index, startangle = 90, counterclock = False);
plt.axis('square')
plt.title('Total Stops\n')


# In[75]:


print('********Flight_Data_Project is done successfully**********')

