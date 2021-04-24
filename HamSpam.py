# -*- coding: utf-8 -*-
# author by Jaswanth Jerripothula

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
from sklearn import model_selection, preprocessing, naive_bayes,metrics
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import string

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

class NavieBays:
    accuracy = 0
    precision = 0
    fmeasure = 0
    recall = 0
    imagepath = 0
    imagepath = ''
    tfidf_vect = TfidfVectorizer(analyzer = 'word', token_pattern = r'\w{1,}', max_features = 20000)
    label_encoder=LabelEncoder()
    
    def trainModel(self,filepath):
        print('Inside trainModel')
        df_inputdata = pd.read_csv('/Users/jaswanthjerripothula/Desktop/SpamHam.csv',usecols = [0,1],encoding = 'latin-1')
        #print(df_inputdata.head())
        df_inputdata.rename(columns = {'v1':'Category','v2':'Message'},inplace = True)
        #df_inputdata['Message'] = df_inputdata['Message'].apply(processMessage)
        df_inputdata['Message'] = df_inputdata['Message'].apply(processMessage)
        #convert the labels from text to numbers
        NavieBays.label_encoder = preprocessing.LabelEncoder()
        NavieBays.label_encoder.fit(df_inputdata['Category'])
        df_inputdata['Category']=NavieBays.label_encoder.transform(df_inputdata['Category'])
        X=df_inputdata.Message
        y=df_inputdata.Category
        #split the dataset into 80% and 20% for training and testing resprctively
        X_train , X_test , y_train , y_test = train_test_split(X,y,test_size = 0.20)
        NavieBays.tfidf_vect.fit(X_train)
        xtrain_tfidf = NavieBays.tfidf_vect.transform(X_train)
        xvalid_tfidf = NavieBays.tfidf_vect.transform(X_test)
        model = naive_bayes.MultinomialNB()
        model.fit(xtrain_tfidf,y_train)
        y_pred = model.predict(xvalid_tfidf)
        NavieBays.accuracy = metrics.accuracy_score(y_test,y_pred)
        # save the trained model into hard disk
        modelfilename = 'NavieBayesModels.sav'
        pickle.dump(model,open(modelfilename,'wb'))
        # confusion metrix
        # get the confusion matrix
        cm = confusion_matrix(y_test,y_pred)
        conf_matrix = pd.DataFrame(data = cm , columns = ['Predicted : 0','Predicted : 1'],index = ['Actual : 0','Actual : 1'])
        plt.figure(figsize = (8,5))
        sn.heatmap(conf_matrix, annot=True , fmt = 'd', cmap = "YlGnBu")
        imageFile = 'NavieBays_confusion.jpg'
        plt.savefig(imageFile)
        #perfomance parameters
        imagePath = ''
        print('Image file path ',imagePath)
        NavieBays.precision =  metrics.precision_score(y_test, y_pred,average=None)
        NavieBays.recall= metrics.recall_score( y_test, y_pred,average=None)
        NavieBays().fmeasure=metrics.f1_score(y_test, y_pred,average=None)
        
        print('NaiveBayes Model','Training Completed')
    def getAccuracy(self):
        return NavieBays.accuracy
    def getPerfmatrix(self):
        return NavieBays.precision,NavieBays.recall,NavieBays.fmeasure,NavieBays.imagepath
    
    def getPrediction(self,inputTweet):
        myData = np.array([inputTweet])
        myData = NavieBays.tfidf_vect.transform(myData)
        filename = 'NavieBayesModels.sav'
        loaded_model=pickle.load(open(filename,'rb'))
        y_pred=loaded_model.predict(myData)
        print(y_pred)
        vals= NavieBays.label_encoder.inverse_transform([y_pred[0]])
        print(vals[0])
        return vals[0]
        #return NavieBays.label_encoder.inverse_transform([y_pred[0]])        