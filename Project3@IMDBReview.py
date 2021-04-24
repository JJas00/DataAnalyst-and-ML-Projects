#!/usr/bin/env python
# coding: utf-8

# In[1]:


print('             IMDB Review Project               ')

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


# In[2]:


inputdata_df = pd.read_csv('/Users/jaswanthjerripothula/Desktop/IMDBDataset.csv',usecols = [0,1],encoding = 'latin-1')


# In[3]:


inputdata_df.head(10)


# In[4]:


inputdata_df.tail(10)


# In[42]:


inputdata_df.rename(columns = {'audience opinion': 'opinion','sentiment': 'Result'}, inplace = True)


# In[43]:


inputdata_df.head(11)


# In[41]:


inputdata_df.count()


# In[28]:


inputdata_df.groupby('Result').describe()


# In[29]:


inputdata_df.groupby('Result').count()


# In[30]:


category_count = pd.DataFrame()
category_count['count'] = inputdata_df['Result'].value_counts()


# In[31]:


category_count['count']


# In[32]:


fig , ax = plt.subplots(figsize = (12,6))
sn.barplot (x = category_count.index , y = category_count['count'], ax = ax)
ax.set_ylabel ('Count',fontsize = 25)
ax.set_xlabel ('Result',fontsize = 25)
ax.tick_params(labelsize = 20)


# In[16]:


#Looking at the above numbers of hams and spams there are more number of positive than negative 


# In[44]:


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


# In[45]:


inputdata_df['opinion'] = inputdata_df['opinion'].apply(processMessage)


# In[46]:


inputdata_df['opinion'].head(11)


# In[21]:


inputdata_df['Result'].value_counts()


# In[22]:


#convert the labels from text to numbers
label_encoder = preprocessing.LabelEncoder()
inputdata_df['Result'] = label_encoder.fit_transform(inputdata_df['Result'])


# In[47]:


a = inputdata_df.opinion
b = inputdata_df.Result


# In[48]:


inputdata_df['Result'].head(11)


# In[51]:


#Split the dataser into 80% and 20% for training respectively
a_train , a_test , b_train , b_test = train_test_split(a,b,test_size = 0.20)


# In[52]:


type(a_train)


# In[53]:


#convert the raw document into a matrix of TF-IDF features
tfidf_vect = TfidfVectorizer(analyzer = 'word', token_pattern = r'\w{1,}', max_features = 20000)
#Create TF-IDF with a_train
tfidf_vect.fit(a_train)


# In[54]:


#the TF-IDF created with a_train for transforming a_train and a_test
atrain_tfidf = tfidf_vect.transform(a_train)
avalid_tfidf = tfidf_vect.transform(a_test)


# In[55]:


#Create a model for NaiveBaye's Model
model = naive_bayes.MultinomialNB()


# In[56]:


#Train the model with X_train and y_train 
model.fit(atrain_tfidf,b_train)


# In[57]:


#Get the prediction for X_test which is transformed with TF-IDF 
b_pred = model.predict(avalid_tfidf)


# In[58]:


#Get accuracy for the model
metrics.accuracy_score(b_test,b_pred)


# In[60]:


#Get the confusion Matrix
cm = confusion_matrix(b_test,b_pred)
conf_matrix = pd.DataFrame(data = cm , columns = ['Predicted : 0','Predicted : 1'],
                          index = ['Actual : 0','Actual : 1'])
plt.figure(figsize = (8,5))
sn.heatmap(conf_matrix, annot=True , fmt = 'd', cmap = "YlGnBu")


# In[61]:


#Positive input for testing
myPositiveData = np.array(["I thought this was a wonderful way to spend time on a too hot summer weekend, sitting in the air conditioned theater and watching a light-hearted comedy. The plot is simplistic, but the dialogue is witty and the characters are likable (even the well bread suspected serial killer). While some may be disappointed when they realize this is not Match Point 2: Risk Addiction, I thought it was proof that Woody Allen is still fully in control of the style many of us have grown to love.<br /><br />This was the most I'd laughed at one of Woody's comedies in years (dare I say a decade?). While I've never been impressed with Scarlet Johanson, in this she managed to tone down her ""sexy"" image and jumped right into a average, but spirited young woman.<br /><br />This may not be the crown jewel of his career, but it was wittier than ""Devil Wears Prada"" and more interesting than ""Superman"" a great comedy to go see with friends."])


# In[63]:


myPositiveData = tfidf_vect.transform(myPositiveData)


# In[65]:


b_result = model.predict(myPositiveData)


# In[66]:


b_result[0]


# In[67]:


positivevalue = label_encoder.inverse_transform([b_result[0]])


# In[68]:


positivevalue[0]


# In[69]:


#Multiple data like both positive and negative results
myMultipleData = (["Basically there's a family where a little boy (Jake) thinks there's a zombie in his closet & his parents are fighting all the time.<br /><br />This movie is slower than a soap opera... and suddenly, Jake decides to become Rambo and kill the zombie.<br /><br />OK, first of all when you're going to make a film you must Decide if its a thriller or a drama! As a drama the movie is watchable. Parents are divorcing & arguing like in real life. And then we have Jake with his closet which totally ruins all the film! I expected to see a BOOGEYMAN similar movie, and instead i watched a drama with some meaningless thriller spots.<br /><br />3 out of 10 just for the well playing parents & descent dialogs. As for the shots with Jake: just ignore them.","I remember this film,it was the first film i had watched at the cinema the picture was dark in places i was very nervous it was back in 74/75 my Dad took me my brother & sister to Newbury cinema in Newbury Berkshire England. I recall the tigers and the lots of snow in the film also the appearance of Grizzly Adams actor Dan Haggery i think one of the tigers gets shot and dies. If anyone knows where to find this on DVD etc please let me know.The cinema now has been turned in a fitness club which is a very big shame as the nearest cinema now is 20 miles away, would love to hear from others who have seen this film or any other like it."])


# In[70]:


myMultipleData_df = pd.DataFrame(myMultipleData, columns={'opinion'})


# In[71]:


myMultipleData_df['opinion'] = myMultipleData_df['opinion'].apply(processMessage)


# In[72]:


myMultiData = tfidf_vect.transform(myMultipleData_df['opinion'])


# In[73]:


b_predlabels = model.predict(myMultiData)


# In[74]:


b_predlabels.shape


# In[75]:


#b_predlabels=y_prelabels.reshape(-1)


# In[77]:


vals = label_encoder.inverse_transform(b_predlabels)


# In[78]:


for val in vals:
    print(val)


# In[79]:


from sklearn.metrics import classification_report


# In[81]:


report = classification_report(b_test,b_pred,labels = [0,1])
print('Report :: \n')
print(report)


# In[82]:


TP = cm[0,0]
FN = cm[0,1]
FP = cm[1,0]
TN = cm[1,1]


# In[86]:


print('Recall or True Positive Rate = TP / (TP + FN)                     :: ',TP / float(TP + FN),'\n',
'Positive Precision value = TP / (TP + FP)                        :: ',TP / float(TP + FP),'\n',
'Negative Predictive value = TN / (TN + FN)                       :: ',TN / float(TN + FN))


# In[87]:


recall = TP/ ( TP + FN)


# In[88]:


recall


# In[89]:


precision = TP / ( TP + FP)


# In[90]:


precision


# In[91]:


f1score = 2 * ( precision * recall ) / ( precision + recall )


# In[92]:


f1score


# In[93]:


print('        IMDB Review Project Completed Successfully       ')


# In[ ]:




