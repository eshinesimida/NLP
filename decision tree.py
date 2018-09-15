# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 20:44:59 2018

@author: admin
"""

import pandas as pd 
#dataframe

df = pd.read_csv('loans.csv')
df.head()
X = df.drop('safe_loans', axis=1) 
y = df.safe_loans 

from sklearn.preprocessing import LabelEncoder 
from collections import defaultdict 
d = defaultdict(LabelEncoder) 
X_trans = X.apply(lambda x: d[x.name].fit_transform(x)) 
X_trans.head() 

from sklearn.cross_validation import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X_trans, y, random_state=1) 

from sklearn import tree 
clf = tree.DecisionTreeClassifier(max_depth=3) 
clf = clf.fit(X_train, y_train) 

with open("safe-loans.dot", 'w') as f:     
    f = tree.export_graphviz(clf,out_file=f,max_depth = 3,
                             impurity = True,
                             feature_names = list(X_train),
                             class_names = ['not safe', 'safe'],
                             rounded = True,
                             filled= True ) 
from subprocess import check_call 
check_call(['dot','-Tpng','safe-loans.dot','-o','safe-loans.png']) 
from IPython.display import Image as PImage 
from PIL import Image, ImageDraw, ImageFont 
img = Image.open("safe-loans.png") 
draw = ImageDraw.Draw(img) 
img.save('output.png') 
PImage("output.png")

test_rec = X_test.iloc[1,:] 
clf.predict([test_rec]) 

from sklearn.metrics import accuracy_score 
accuracy_score(y_test, clf.predict(X_test)) 