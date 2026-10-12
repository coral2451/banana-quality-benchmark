import sys, warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from banana_revision_analysis import FEATURES, SEED
df=pd.read_csv(sys.argv[1]); X=df[FEATURES].to_numpy(float); y=(df.Quality.str.lower()=='good').astype(int).to_numpy()
tr,te=train_test_split(np.arange(len(y)),test_size=0.2,stratify=y,random_state=SEED)
sc=StandardScaler().fit(X[tr])
dt=DecisionTreeClassifier(criterion='entropy',max_depth=4,random_state=SEED).fit(sc.transform(X[tr]),y[tr])
print('acc',accuracy_score(y[te],dt.predict(sc.transform(X[te]))))
fig,ax=plt.subplots(figsize=(16,6.7))
plot_tree(dt,feature_names=FEATURES,class_names=['Bad','Good'],filled=True,rounded=True,fontsize=7,ax=ax,impurity=True,precision=2)
fig.tight_layout(); fig.savefig(sys.argv[2]+'/figS13_tree.png',dpi=200)
