#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 17:33:58 2020

@author: apple
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from itertools import permutations
from sklearn import tree

wno_df=pd.read_excel('/Users/apple/Desktop/Desktop/Viacom/Object Detection/CTR3and7_WNO_features.xlsx', sheet_name='Sheet2')

col_names=pd.read_excel('/Users/apple/Desktop/Desktop/Viacom/Object Detection/CTR3and7_WNO_features.xlsx', sheet_name='Sheet11')

filter_col = [col for col in wno_df.columns if col not in list(col_names['column'])]

X=wno_df[filter_col]
#X=wno_df
y=pd.DataFrame(wno_df['CTR day7'])

X.drop(columns=['video_id','video_title_text','CTR day7', 'CTR day3', 'post_url',
               'date_published','thumbnail_url', 'Image_text'],inplace=True)

quantile=y['CTR day7'].quantile([.25,.5,.75,1])
uppr_limit=quantile[.75]+1.5*(quantile[.75]-quantile[.25])
lwr_limit=quantile[.25]-1.5*(quantile[.75]-quantile[.25])
y['CTR day7'][y['CTR day7']>uppr_limit]=uppr_limit
y['CTR day7'][y['CTR day7']<lwr_limit]=lwr_limit

y['CTR day 7 Bins']="2"
y['CTR day 7 Bins'][y['CTR day7']<=y['CTR day7'].quantile(.33)]="1"
y['CTR day 7 Bins'][y['CTR day7']>y['CTR day7'].quantile(.66)]="3"

y.drop(columns=["CTR day7"],inplace=True)
X.fillna(0,inplace=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

cases={}
values=[]

perm = permutations([2, 3, 4, 5, 6, 7, 8, 9, 10, 11], 4)
for i in list(perm):
    print(i)
    clf = DecisionTreeClassifier(max_depth = i[0], min_samples_split=i[1] ,min_samples_leaf=i[2], 
                                 random_state = 0, max_features=i[3])
    clf.fit(X_train, y_train)
    y_pred=clf.predict(X_test)    
    accuracy=metrics.accuracy_score(y_test, y_pred)
    cases={'max_depth':i[0], 'min_samples_split':i[1], 'min_samples_leaf':i[2], 'max_features':i[3],'Accuracy':accuracy}
    values.append(cases)
    
df=pd.DataFrame(values)


clf = DecisionTreeClassifier(max_depth =7 , min_samples_split=8 ,min_samples_leaf=2, 
                             random_state = 0, max_features=6)
clf.fit(X_train, y_train)
y_pred=clf.predict(X_test) 

fn=X_train.columns
cn=['1', '2', '3']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (6,2), dpi=300)
tree.plot_tree(clf,
               feature_names = fn, 
               class_names=cn,
               filled = True);
fig.savefig('imagename.png')