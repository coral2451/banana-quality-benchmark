import sys, warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from banana_revision_analysis import FEATURES, SEED
BLUE, ORANGE, INK, MUTED, GRID = '#2a78d6', '#eb6834', '#1f1f1e', '#6b6a63', '#e6e5e0'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.edgecolor':MUTED,'axes.labelcolor':INK,'xtick.color':MUTED,'ytick.color':MUTED,'axes.spines.top':False,'axes.spines.right':False})
df=pd.read_csv(sys.argv[1]); out=sys.argv[2]
X=df[FEATURES].to_numpy(float); y=(df.Quality.str.lower()=='good').astype(int).to_numpy()
tr,te=train_test_split(np.arange(len(y)),test_size=0.2,stratify=y,random_state=SEED); Xtr,ytr=X[tr],y[tr]
cv=StratifiedKFold(5,shuffle=True,random_state=SEED)
def cvs(est): return cross_val_score(Pipeline([('s',StandardScaler()),('c',est)]),Xtr,ytr,cv=cv,n_jobs=-1)
depths=list(range(1,21))
d=np.array([cvs(DecisionTreeClassifier(criterion='entropy',max_depth=k,random_state=SEED)) for k in depths])
fig,ax=plt.subplots(figsize=(7.5,3.7))
ax.errorbar(depths,d.mean(1),yerr=d.std(1,ddof=1),color=BLUE,marker='o',ms=4,lw=2,capsize=2)
ax.axvline(12,color=MUTED,ls='--',lw=0.8); ax.text(12.3,d.mean(1).min()+0.01,'selected: 12',color=MUTED,fontsize=8)
ax.set_xlabel('max_depth'); ax.set_ylabel('Five-fold CV accuracy (training set)'); ax.grid(True,color=GRID); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig(f'{out}/figS5_dt_depth_cv.png',dpi=250); plt.close(fig)
ks=list(range(1,32,2))
u=np.array([cvs(KNeighborsClassifier(n_neighbors=k)) for k in ks]); w=np.array([cvs(KNeighborsClassifier(n_neighbors=k,weights='distance')) for k in ks])
fig,ax=plt.subplots(figsize=(7.5,3.7))
ax.errorbar(ks,u.mean(1),yerr=u.std(1,ddof=1),color=BLUE,marker='o',ms=4,lw=2,capsize=2,label='Uniform weights')
ax.errorbar(ks,w.mean(1),yerr=w.std(1,ddof=1),color=ORANGE,marker='s',ms=4,lw=2,capsize=2,ls='--',label='Distance weights')
ax.axvline(19,color=MUTED,ls='--',lw=0.8); ax.text(19.3,min(u.mean(1).min(),w.mean(1).min())+0.001,'selected: k = 19, distance',color=MUTED,fontsize=8)
ax.legend(frameon=False,loc='lower right'); ax.set_xlabel('k (number of neighbors)'); ax.set_ylabel('Five-fold CV accuracy (training set)')
ax.grid(True,color=GRID); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig(f'{out}/figS6_knn_k_cv.png',dpi=250); plt.close(fig)
print('dt plateau', d.mean(1)[8:13].round(4), 'knn dist19', w.mean(1)[9].round(4))
