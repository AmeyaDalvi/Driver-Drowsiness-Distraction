#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
#

# In[2]:


og_data=pd.read_csv("C:\\Users\\bhushan\\Downloads\\train2016.csv")
og_data.head()


# new_data=og_data['YOB','Gender','Income','HouseholdStatus','EducationLevel','Party','Q100562','Q102089','Q102674','Q106388','Q108343','Q109244','Q109367','Q114152','Q115611','Q117193','Q118232','Q118233','Q121011','Q122771','Q123464','Q123621']]
# 
# new_data.head()

# In[3]:


new_data=og_data[['YOB','Gender','Income','HouseholdStatus','EducationLevel','Party','Q100562','Q102089','Q102674','Q106388','Q108343','Q109244','Q109367','Q114152','Q115611','Q117193','Q118232','Q118233','Q121011','Q122771','Q123464','Q123621']]

new_data.head()


# In[4]:


#sns.heatmap(new_data.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[5]:


#sns.heatmap(new_data.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[6]:


#new_data.head(10)


# In[7]:


sex = pd.get_dummies(new_data['Gender'],drop_first=True)
demo=pd.get_dummies(new_data['Party'],drop_first=True)
ideal=pd.get_dummies(new_data['Q118232'],drop_first=True)
new_data=pd.concat([new_data,sex,demo,ideal],axis=1)


# In[8]:


BetterLife=pd.get_dummies(new_data['Q100562'],drop_first=True)
new_data=pd.concat([new_data,BetterLife],axis=1)
new_data.rename(columns={'Yes':'BetterLife'},inplace=True)


# In[9]:

#
work_hours=pd.get_dummies(new_data['Q106388'],drop_first=True)
new_data=pd.concat([new_data,work_hours],axis=1)
new_data.rename(columns={'Yes':'WorkHours'},inplace=True)


# In[10]:


debt=pd.get_dummies(new_data['Q102674'],drop_first=True)
new_data=pd.concat([new_data,debt],axis=1)
new_data.rename(columns={'Yes':'CreditDebt'},inplace=True)


# In[11]:


rent=pd.get_dummies(new_data['Q102089'],drop_first=True)
new_data=pd.concat([new_data,rent],axis=1)
new_data.rename(columns={'Yes':'Rent'},inplace=True)


# In[12]:


job=pd.get_dummies(new_data['Q123621'],drop_first=True)
new_data=pd.concat([new_data,job],axis=1)
new_data.rename(columns={'Yes':'Job'},inplace=True)


# In[13]:


wage=pd.get_dummies(new_data['Q123464'],drop_first=True)
new_data=pd.concat([new_data,wage],axis=1)
new_data.rename(columns={'Yes':'MinimumWage'},inplace=True)


# In[14]:


school=pd.get_dummies(new_data['Q122771'],drop_first=True)
new_data=pd.concat([new_data,school],axis=1)
new_data.rename(columns={'Yes':'School'},inplace=True)


# In[15]:


stress=pd.get_dummies(new_data['Q121011'],drop_first=True)
new_data=pd.concat([new_data,stress],axis=1)
new_data.rename(columns={'Yes':'Stress'},inplace=True)


# In[16]:


violence=pd.get_dummies(new_data['Q118233'],drop_first=True)
new_data=pd.concat([new_data,violence],axis=1)
new_data.rename(columns={'Yes':'IntViolence'},inplace=True)


# In[17]:


schedule=pd.get_dummies(new_data['Q117193'],drop_first=True)
new_data=pd.concat([new_data,schedule],axis=1)
new_data.rename(columns={'Yes':'Schedule'},inplace=True)


# In[18]:


beenpoor=pd.get_dummies(new_data['Q109367'],drop_first=True)
new_data=pd.concat([new_data,beenpoor],axis=1)
new_data.rename(columns={'Yes':'BeenPoor'},inplace=True)


# In[19]:


gun=pd.get_dummies(new_data['Q115611'],drop_first=True)
new_data=pd.concat([new_data,gun],axis=1)
new_data.rename(columns={'Yes':'Gun'},inplace=True)


# In[20]:


feminist=pd.get_dummies(new_data['Q109244'],drop_first=True)
new_data=pd.concat([new_data,feminist],axis=1)
new_data.rename(columns={'Yes':'Feminist'},inplace=True)
new_data.head(20)


# In[21]:


employed=pd.get_dummies(new_data['Q123621'],drop_first=True)
new_data=pd.concat([new_data,employed],axis=1)
new_data.rename(columns={'Yes':'Employed'},inplace=True)
#new_data.drop(['Employeed'],axis=1,inplace=True)
new_data.head(20)


# In[22]:


#new_data.drop(['Gender','Party','Q118232','Q109244','Q123621'],axis=1,inplace=True)


# In[23]:


#sex = pd.get_dummies(new_data['Gender'],drop_first=True)
#demo=pd.get_dummies(new_data['Party'],drop_first=True)
#ideal=pd.get_dummies(new_data['Q118232'],drop_first=True)
#feminist=pd.get_dummies(new_data['Q109244'],drop_first=True)
#employeed=pd.get_dummies(new_data['Q123621'],drop_first=True)
#new_data.drop(['Gender','Party','Q118232','Q109244','Q123621'],axis=1,inplace=True)
#new_data=pd.concat([new_data,sex,demo,ideal,feminist,employeed],axis=1)


#new_data.rename(columns={'Yes':'Feminist'},inplace=True)


# In[24]:


##sns.set_style('whitegrid')
#sns.countplot(x='Republica#n',hue='Employed',data=new_data)


# In[25]:


##sns.set_style('whitegrid')
#sns.countplot(x='Republica#n',hue='Male',data=new_data)


# In[26]:


##sns.heatmap(new_data.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[27]:


##sns.set_style('whitegrid')
#sns.countplot(x='Education#Level',hue='Income',data=new_data)


# In[28]:


nulls=new_data['Income'].isna().sum()    
print(nulls)


# In[29]:


#Dealing with YOB 
mean_yob=np.mean(new_data['YOB'])
print (mean_yob)

def fill_yob(yob):
    if(pd.isnull(yob[0])):
        return mean_yob
    return yob[0]

new_data['YOB']=new_data[['YOB','Male']].apply(fill_yob,axis=1)
#new_data.head(1000)

#categorizing YOB into 5 groups

def Age(cols):
    yob=cols[0]
    age=np.abs(2016-yob)
    return age

new_data['YOB']=new_data[['YOB','Male']].apply(Age,axis=1)

new_data.rename(columns={'YOB':'Age'},inplace=True)
new_data.head()


# In[30]:


def income_category(cols):
    income=cols[0]
    if(pd.isnull(income)):
        return income
    elif(income=='under $25,000'):
        return 1
    elif(income=='$25,001 - $50,000'):
        return 2
    elif(income=='$50,000 - $74,999'):
        return 3
    elif(income=='$75,000 - $100,000'):
        return 4
    elif(income=='$100,001 - $150,000'):
        return 5
    else :
        return 6
    
new_data['Income']=new_data[['Income','Male']].apply(income_category,axis=1)    


# In[31]:


new_data.head()


# In[32]:


def edu(cols):
    ed=cols[0]
    if(pd.isnull(ed)):
        return 1
    elif(ed=='Current K-12'):
        return 1
    elif(ed=="Master's Degree"):
        return 2
    elif(ed=="Bachelor's Degree"):
        return 3
    elif(ed=="Current Undergraduate"):
        return 4
    elif(ed=="High School Diploma"):
        return 5
    elif(ed=="Doctoral Degree"):
        return 6
    else:
        return 7


new_data['EducationLevel']=new_data[['EducationLevel','Male']].apply(edu,axis=1)    


# In[33]:


new_data.head(20)


# In[34]:


##sns.set_style('whitegrid')
#sns.countplot(x='Education#Level',hue='Income',data=new_data)


# In[35]:


##sns.set_style('whitegrid')
#sns.countplot(x='Income',h#ue='EducationLevel',data=new_data)


# In[36]:


def fill_income(cols):
    inc=cols[0]
    ed=cols[1]
    if(pd.isnull(inc)):
        if(ed==1):
            return 1
        elif(ed==2):
            return 6
        elif(ed==3):
            return 3
        elif(ed==4):
            return 1
        elif(ed==5):
            return 2
        elif(ed==6):
            return 6
        else :
            return 2
    return inc


new_data['Income']=new_data[['Income','EducationLevel']].apply(fill_income,axis=1)    


# In[49]:


new_data.head(20)


# In[38]:


##sns.heatmap(new_data.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[39]:


def age_category(cols):
    age=cols[0]
    
    if age<25:
        return 1
    elif (age>=25 and age<36):
        return 2
    elif(age>=36 and age<49):
        return 3
    elif(age>=49 and age<64):
        return 4
    else: return 5

new_data['Age'] = new_data[['Age','Male']].apply(age_category,axis=1)


# In[40]:


##sns.set_style('whitegrid')
#sns.countplot(x='Republica#n',hue='Age',data=new_data)


# In[41]:


new_data['Age'].head()


# In[42]:


from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(new_data[['Age','EducationLevel','Male','Employed','Feminist','Pragmatist','Income','BetterLife','Rent','CreditDebt','MinimumWage','Stress','IntViolence','Standard hours','Gun','BeenPoor']],new_data['Republican'],test_size=0.25,random_state=101)


# In[43]:


from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression(solver='lbfgs')
logmodel.fit(X_train,Y_train)

predictions = logmodel.predict(X_test)


# In[44]:


from sklearn.metrics import classification_report
from sklearn import metrics

print(classification_report(Y_test,predictions))
print(metrics.f1_score(Y_test, predictions, average='weighted', labels=np.unique(predictions)))


# In[45]:


from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=10,random_state=0)
rfc.fit(X_test, Y_test)
rfc_predict=rfc.predict(X_test)
print(rfc.feature_importances_)
#print(classification_report(Y_test,clf))
#print(metrics.f1_score(Y_test, clf, average='weighted', labels=np.unique(clf)))


# In[46]:


print("=== Classification Report ===")
print(classification_report(Y_test, rfc_predict))
print('\n')


# In[47]:


from sklearn import svm
clf = svm.SVC(gamma='scale')
clf.fit(X_test, Y_test)  

clf_predict=clf.predict(X_test)

print("=== Classification Report ===")
print(classification_report(Y_test, clf_predict))
print('\n')


# In[48]:


print(rfc.predict([[1,3,1,1,0,1,2,1,0,1,3,0,1,1,1,1]]))

