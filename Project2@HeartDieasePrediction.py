#!/usr/bin/env python
# coding: utf-8

# In[2]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import numpy as np
#import statsmodels.api as sm
#import scipy.stats as st
import seaborn as sn
from sklearn.metrics import confusion_matrix
#import matplotlib.mtab as mlab
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


heart_df = pd.read_csv('/Users/jaswanthjerripothula/Desktop/framingham.csv')


# In[4]:


heart_df.head()


# In[5]:


heart_df.tail()


# In[6]:


heart_df.isnull().sum()


# In[7]:


count = 0
for i in heart_df.isnull().sum(axis = 1):
    if i > 0:
        count = count + 1
print('Total number of rows with missing values is :: ',count)
print('It is ',round((count / len(heart_df.index))*100),'percent of the entire dataset the rows with missing values are imputed.')


# In[9]:


#Imputation :- The NAN or NULL values in the records will be replaced with proper values depending on the nature of data
#Find the null values
#repeat of 6
heart_df.isnull().sum()


# In[10]:


#repeat of 7
count = 0
for i in heart_df.isnull().sum(axis = 1):
    if i > 0:
        count = count + 1
print('Total number of rows with missing values is :: ',count)
print('Percentage of rows with missing values :: ',round((count / len(heart_df.index))*100),'%')


# In[11]:


#The data is missing is 14% of the total data which us high we need to impute the values


# In[16]:


heart_df['cigsPerDay'].isnull().sum()


# In[17]:


#There are 29 records with 'cigsPerDay' and we will replace them with mean value
import math
mean_value = heart_df['cigsPerDay'].mean()
mean_value = math.floor(mean_value)
heart_df['cigsPerDay'] = heart_df['cigsPerDay'].fillna(mean_value)


# In[18]:


#NuLL Values have been replaced by mean value
heart_df['cigsPerDay'].isnull().sum()


# In[19]:


#Find the education counts of different types
heart_df['education'].value_counts()


# In[20]:


#the education 1.0 is maximum and we can replace null values for education with modee value
heart_df['education'].fillna(heart_df['education'].mode()[0], inplace = True)


# In[21]:


heart_df['education'].value_counts()


# In[22]:


#Fill the null values in glucose column with mean value
mean_value = heart_df['glucose'].mean()
heart_df['glucose'] = heart_df['glucose'].fillna(mean_value)


# In[23]:


heart_df.isnull().sum()


# In[24]:


#Fill the BPmeds column which has null values with mode
heart_df['BPMeds'].fillna(heart_df['BPMeds'].mode()[0], inplace = True)
#Fill the heartRate column which has null values with mode
heart_df['heartRate'].fillna(heart_df['heartRate'].mode()[0], inplace = True)


# In[25]:


heart_df.isnull().sum()


# In[26]:


#Fill the BMI column which has null values with median
heart_df['BMI'] = heart_df.fillna(heart_df['BMI'].median())
#Fill the totChol column which has null values with mean
mean_value = heart_df['totChol'].mean()
heart_df['totChol'] = heart_df['totChol'].fillna(mean_value)
heart_df.isnull().sum()


# In[27]:


#Imputation is completed as we have no columns with null values in the data or records


# In[30]:


heart_df.head(10)


# In[31]:


heart_df.tail(10)


# In[37]:


def draw_histograms(dataframe, features, rows, cols):
    fig = plt.figure(figsize = (20,20))
    for i, feature in enumerate(features):
        ax = fig.add_subplot(rows,cols,i+1)
        dataframe[feature].hist(bins = 20, ax = ax, facecolor = 'midnightblue')
        ax.set_title(feature+"Distribution",color = 'DarkRed')
        
    fig.tight_layout()
    plt.show()


# In[38]:


draw_histograms(heart_df, heart_df.columns ,6,3)


# In[39]:


heart_df.TenYearCHD.value_counts()


# In[40]:


sn.countplot(x = 'TenYearCHD',data = heart_df)


# In[41]:


heart_df.describe()


# In[42]:


import sklearn
new_features = heart_df[['age','male','cigsPerDay','totChol','sysBP','glucose','TenYearCHD']]
x = new_features.iloc[:,:-1]
y = new_features.iloc[:,-1]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 5)


# In[45]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(x_train,y_train)
y_pred = logreg.predict(x_test)


# In[46]:


sklearn.metrics.accuracy_score(y_test,y_pred)


# In[47]:


#Get the confusion Matrix
cm = confusion_matrix(y_test,y_pred)
conf_matrix = pd.DataFrame(data = cm , columns = ['Predicted : 0','Predicted : 1'],
                          index = ['Actual : 0','Actual : 1'])
plt.figure(figsize = (8,5))
sn.heatmap(conf_matrix, annot=True , fmt = 'd', cmap = "YlGnBu")


# In[49]:


TP = cm[0,0]
FP = cm[0,1]
FN = cm[1,0]
TN = cm[1,1]
sensitivity = TP / float(TP + FN)
specificity = TN / float(TN + FP)


# In[59]:


print(
'The accuracy of the model = TP + TN / (TP + TN + FP + FN)        :: ',(TP + TN)/ float(TP + TN + FP + FN),'\n',
'The Missclassification = 1 - Accuracy                           :: ',1 - ((TP + TN) / float(TP + TN + FP + FN)),'\n',
'Sensitivity or True Positive Rate = TP / (TP + FN)              :: ',TP / float(TP + FN),'\n',
'Specificity or True Negative Rate = TN / (TN + FP)              :: ',TN / float(TN + FP),'\n',
'Positive Predictive value = TP / (TP + FP)                      :: ',TP / float(TP + FP),'\n',
'Negative Predictive value = TN / (TN + FN)                      :: ',TN / float(TN + FN),'\n',
'Positive Likelihood Ratio = Sensitivity / (1 - specificity)     :: ',sensitivity / (1 - specificity),'\n',
'Negative Likelihood Ratio = (1 - Sensitivity) / Specificity     :: ',(1 - sensitivity) / specificity)


# In[64]:


from sklearn.metrics import classification_report, confusion_matrix
print('Confusion_matrix :: \n',confusion_matrix(y_test, y_pred))
print('Report :: \n',classification_report(y_test, y_pred))


# In[65]:


print('      Framing Ham Project completed successfully     ')


# In[ ]:




