# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 19:09:05 2018

@author: admin
"""

import numpy as np
import pandas as pd

df = pd.read_csv('customer_churn.csv')
df.head()
X = df.loc[:,['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']]
y = df.Exited

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder1 = LabelEncoder()
X.Geography= labelencoder1.fit_transform(X.Geography)
labelencoder2 = LabelEncoder()
X.Gender = labelencoder2.fit_transform(X.Gender)

onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

X = np.delete(X, [0], 1)

y = y[:,np.newaxis]

onehotencoder = OneHotEncoder()
y = onehotencoder.fit_transform(y).toarray()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

len(X_train)
#normalization
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))