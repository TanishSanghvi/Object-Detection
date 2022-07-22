
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
from itertools import permutations
from sklearn import tree

wno_df=pd.read_excel('Object Detection/Segregated Files/CTR3and7_WNO_features_image_text.xlsx')

X=wno_df.copy()
y=pd.DataFrame(wno_df['CTR day7'])

X.drop(columns=['video_id','CTR day7', 'Image_text'],inplace=True)

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

TOP_FEATURES = 100

forest = ExtraTreesClassifier(n_estimators=250, max_depth=5, random_state=1)
forest.fit(X,y)

importances = forest.feature_importances_
std = np.std(
    [tree.feature_importances_ for tree in forest.estimators_],
    axis=0
)
indices = np.argsort(importances)[::-1]
indices = indices[:TOP_FEATURES]

imp=[]
score=[]
print('Top features:')
for f in range(TOP_FEATURES):
    print('%d. feature %d (%f)' % (f + 1, indices[f], importances[indices[f]]))
    imp.append(indices[f])
    score.append(importances[indices[f]])
    

X=X.iloc[:,imp]
feats=list(X.columns)








