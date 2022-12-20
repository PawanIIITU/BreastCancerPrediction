

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv("Breastcancer.csv")
df.head(6)
df.shape

df.isna().sum()
df=df.dropna(axis=1)
df.shape
df['diagnosis'].value_counts()
sns.countplot(df['diagnosis'],label='count')   #visualize the count
df.dtypes
from sklearn.preprocessing import LabelEncoder
LabelEncoder_Y=LabelEncoder()
df.iloc[:,1]=LabelEncoder_Y.fit_transform(df.iloc[:,1].values)
df.iloc[:,1]
sns.pairplot(df.iloc[:,1:5],hue='diagnosis')
df.head(5)
plt.figure(figsize=(10,10))
sns.heatmap(df.iloc[:,1:12].corr(),annot=True)
X=df.iloc[:,2:31].values   
Y=df.iloc[:,1].values
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=0)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)
X_train
X_test

def models(X_train,Y_train):
    from sklearn.linear_model import LogisticRegression 
    log=LogisticRegression(random_state=0)
    log.fit(X_train,Y_train)
    from sklearn.tree import DecisionTreeClassifier
    tree=DecisionTreeClassifier(criterion='entropy',random_state=0)
    tree.fit(X_train,Y_train)
    from sklearn import svm
    model=svm.SVC(kernel='linear',gamma="auto",C=0.3)
    model.fit(X_train,Y_train)
    print('[0]Logistic Regression Training Accuracy:',log.score(X_train,Y_train))
    print('[1]Decision Tree Classifier Training Accuracy:',tree.score(X_train,Y_train))
    print('[2]Support Vector Machine training Accuracy',model.score(X_train,Y_train))
    return log,tree,model
model=models(X_train,Y_train)
from sklearn.metrics import confusion_matrix
for i in range(len(model)):
    print('model',i)
    cm=confusion_matrix(Y_test,model[i].predict(X_test))
    TP=cm[0][0]
    TN=cm[1][1]
    FN=cm[1][0]
    FP=cm[0][1]
    print(cm)
    print('Testing Accuracy=',(TP+TN)/(TP+TN+FP+FN))
    print()
    from sklearn.metrics import classification_report
    from sklearn.metrics import accuracy_score
    for i in range(len(model)):
        print('model',i)
        print(classification_report(Y_test,model[i].predict(X_test)))
        print(accuracy_score(Y_test,model[i].predict(X_test)))
        print()
        
    
    
    


