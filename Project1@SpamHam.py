#!/usr/bin/env python
# coding: utf-8

# In[22]:


print('             SpamHam Project               ')

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


# In[5]:


df_inputdata = pd.read_csv('/Users/jaswanthjerripothula/Desktop/SpamHam.csv',usecols = [0,1],encoding = 'latin-1')


# In[7]:


df_inputdata.head()


# In[11]:


df_inputdata.rename(columns = {'v1': 'Category','v2': 'Message'}, inplace = True)


# In[12]:


df_inputdata.head()


# In[13]:


df_inputdata.count()


# In[14]:


df_inputdata.groupby('Category').describe()


# In[15]:


df_inputdata.groupby('Category').count()


# In[18]:


category_count = pd.DataFrame()
category_count['count'] = df_inputdata['Category'].value_counts()


# In[19]:


category_count['count']


# In[34]:


fig , ax = plt.subplots(figsize = (12,6))
sn.barplot (x = category_count.index , y = category_count['count'],ax = ax)
ax.set_ylabel ('Count',fontsize = 20)
ax.set_xlabel ('Category',fontsize = 20)
ax.tick_params(labelsize = 20)


# In[33]:


#Looking at the above numbers of hams and spams there are more number of hams than spams 


# In[35]:


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


# In[37]:


df_inputdata['Message'] = df_inputdata['Message'].apply(processMessage)


# In[38]:


df_inputdata['Message'].head(7)


# In[39]:


df_inputdata['Message'].tail(7)


# In[40]:


df_inputdata['Category'].value_counts()


# In[42]:


#convert the labels from text to numbers
label_encoder = preprocessing.LabelEncoder()
df_inputdata['Category'] = label_encoder.fit_transform(df_inputdata['Category'])


# In[43]:


X = df_inputdata.Message
y = df_inputdata.Category


# In[44]:


df_inputdata['Category'].head(7)


# In[45]:


#Split the dataser into 80% and 20% for training respectively
X_train , X_test , y_train , y_test = train_test_split(X,y,test_size = 0.20)


# In[46]:


type (X_train)


# In[47]:


#convert the raw document into a matrix of TF-IDF features
tfidf_vect = TfidfVectorizer(analyzer = 'word', token_pattern = r'\w{1,}', max_features = 20000)
#Create TF-IDF with X_train
tfidf_vect.fit(X_train)


# In[48]:


#the TF-IDF created with X_train for transforming X_train and X_test
xtrain_tfidf = tfidf_vect.transform(X_train)
xvalid_tfidf = tfidf_vect.transform(X_test)


# In[50]:


#Create a model for NaiveBaye's Model
model = naive_bayes.MultinomialNB()


# In[51]:


#Create a model for NaiveBaye's Model
model = naive_bayes.MultinomialNB()


# In[53]:


#Get the prediction for X_test which is transformed with TF-IDF 
y_pred = model.predict(xvalid_tfidf)


# In[57]:


#Get accuracy for the model
metrics.accuracy_score(y_test,y_pred)


# In[63]:


#Get the confusion Matrix
cm = confusion_matrix(y_test,y_pred)
conf_matrix = pd.DataFrame(data = cm , columns = ['Predicted : 0','Predicted : 1'],
                          index = ['Actual : 0','Actual : 1'])
plt.figure(figsize = (8,5))
sn.heatmap(conf_matrix, annot=True , fmt = 'd', cmap = "YlGnBu")


# In[69]:


#Ham input for testing
myHamData = np.array(["Nah I don't think he goes to usf, he lives around here though"])
#Spam input for testing 
mySpamData = np.array(["URGENT! You have won a 1 week FREE membership in our £100,000 Prize Jackpot! Txt the word: CLAIM to No: 81010 T&C www.dbuk.net LCCLTD POBOX 4403LDNW1A7RW18"])


# In[70]:


myHamData = tfidf_vect.transform(myHamData)


# In[101]:


y_result1 = model.predict(myHamData)


# In[102]:


y_result1[0]


# In[103]:


hamvalue = label_encoder.inverse_transform([y_result1[0]])


# In[105]:


hamvalue[0]


# In[121]:


#Spam,Ham
myMultiplesData=["Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's","Had your mobile 11 months or more? U R entitled to Update to the latest colour mobiles with camera for Free! Call The Mobile Update Co FREE on 08002986030",'I HAVE A DATE ON SUNDAY WITH WILL!!']


# In[122]:


df_myMultiplesData=pd.DataFrame(myMultiplesData,columns={'Message'})


# In[123]:


df_myMultiplesData['Message'] = df_myMultiplesData['Message'].apply(processMessage)


# In[124]:


myMultiData = tfidf_vect.transform(df_myMultiplesData['Message'])


# In[125]:


y_predlabels=model.predict(myMultiData)


# In[126]:


y_predlabels.shape


# In[127]:


#y_predlabels=y_prelabels.reshape(-1)


# In[128]:


vals = label_encoder.inverse_transform(y_predlabels)


# In[129]:


for val in vals:
    print(val)


# In[130]:


from sklearn.metrics import classification_report


# In[131]:


report = classification_report(y_test,y_pred,labels=[0,1])


# In[132]:


print(report)


# In[133]:


type(report)


# In[134]:


#recall=TP/(TP+FN)
#precision=TP/(TP+FP)
#f1-score = 2*(precision*recall)/(precision+recall)


# In[146]:


TP = cm[0,0]
FN = cm[0,1]
FP = cm[1,0]
TN = cm[1,1]


# In[137]:


recall = TP/(TP+FN)


# In[138]:


recall


# In[141]:


precision = TP/(TP+FP)


# In[142]:


precision


# In[143]:


f1score = 2*(precision*recall)/(precision+recall)


# In[144]:


f1score


# In[145]:


print('     HAMSPAM Project completed successfully      ')


# In[ ]:




