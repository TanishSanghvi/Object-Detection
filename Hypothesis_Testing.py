#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 16:45:13 2020

@author: apple
"""

import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import pearsonr

wno_df=pd.read_excel('CTR3and7_WNO_features.xlsx') 
wno_df['z']=''

#Enter the variables you want to test
x='object_cap'
y='label_Song'
#a='object_Microphone'
#b='Color_of_skin 1'


for i in range(0,139):
    if wno_df[x][i]>0 or wno_df[y][i]>0:
        wno_df['z'][i]='Yes'
    else:
        wno_df['z'][i]='No'
        
#for i in range(0,139):
#    if wno_df[x][i]>28572:
#        wno_df['z'][i]='Yes'
#    else:
#        wno_df['z'][i]='No'

wno_df=wno_df[['CTR day7','z']]
print(wno_df['z'].value_counts()) # Making sure count is not far apart
wno_df.rename(columns={'CTR day7':'CTRday7'}, inplace=True)
yes=wno_df[wno_df['z']=='Yes']
print("Yes", yes['CTRday7'].mean()) #Making sure means are not far apart
no=wno_df[wno_df['z']=='No']
print("No", no['CTRday7'].mean())

#Performing Anova
mod = ols('CTRday7 ~ z' ,data=wno_df).fit()
aov_table = sm.stats.anova_lm(mod, typ=2)
print(aov_table['PR(>F)'][0])


wno_df=wno_df[['CTR day7','area_covered_by_text','area_covered_by_faces']]

list1 = wno_df['CTR day7'] 
list2 = wno_df['avg_area_covered_by_a_face'].quantile([0.25,0.5,0.75]) #Bucketing for correlation

# Apply the pearsonr() 
corr, _ = pearsonr(list1, list2) 
#corr2, _ = pearsonr(list1, list3) 

