from pathlib import Path
import json
import pandas as pd,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
R=Path('/home/moshtasa/Research/phd-recsys');D=R/'First_paper_material/analysis/score_components';O=D/'prepared';O.mkdir(exist_ok=True)
f=pd.read_csv(D/'components.csv')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':14,'axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False,'axes.titleweight':'bold','axes.labelcolor':'#163248','text.color':'#163248','xtick.color':'#536471','ytick.color':'#536471','figure.facecolor':'white','savefig.facecolor':'white'})
colors={'mean_prediction':'#009E73','mean_bi':'#142D4E','mean_bu':'#61B8E8','mean_interaction':'#0072B2','mu':'#D7AA00'}
labels={'mean_prediction':r'Predicted rating $\hat r_{ui}$','mean_bi':r'Book bias $b_i$','mean_bu':r'User bias $b_u$','mean_interaction':r'Vector dot product $p_u^\top q_i$','mu':r'Global rating mean $\mu$'}
names={'real_unseen_prediction':('01_real_users_unseen_thriller_prediction.png','Prediction | Real users, unrated Thriller books','Equal weight per user; average over all unrated Thriller books, not only Top-K.'),'synthetic_training':('02_synthetic_users_thriller_training.png','Training fit | Synthetic users, Thriller books','Recorded training rating = 5; averages over synthetic users and Thriller books.'),'real_observed_training':('03_real_users_thriller_training.png','Training fit | Real users, rated Thriller books','Average over original observed Thriller ratings; recorded ratings may differ.')}
for view,(name,title,note) in names.items():
 fig,axs=plt.subplots(1,2,figsize=(16,7.5),sharey=True)
 data=f[(f.view==view)&(f.group=='target')]
 for ax,model in zip(axs,['svd','nmf']):
  d=data[data.model==model].sort_values('n_fake');assert len(d)==(8 if view=='synthetic_training' else 9)
  assert np.allclose(d.mean_prediction,d.mu+d.mean_bu+d.mean_bi+d.mean_interaction,atol=1e-10)
  x=np.arange(len(d))
  for key in ['mu','mean_bi','mean_bu','mean_interaction','mean_prediction']:
   ax.plot(x,d[key],color=colors[key],label=labels[key],lw=4.2 if key=='mean_prediction' else 2.2,marker='o' if key=='mean_prediction' else None,ms=5,zorder=5 if key=='mean_prediction' else 3,ls='--' if key=='mu' else '-')
  ax.axhline(0,color='#A6B2BB',lw=.7,zorder=1);ax.set_title(model.upper(),loc='left',pad=14,fontsize=20)
  ax.set_xticks(x,['Base' if n==0 else str(n) for n in d.n_fake]);ax.set_xlabel('Injected users',labelpad=10);ax.set_ylabel('Mean raw score component');ax.grid(axis='y',color='#DEE5EA',lw=.7);ax.tick_params(length=0,pad=7,labelleft=True)
 fig.suptitle(title,x=.065,ha='left',y=.97,fontsize=25,fontweight='bold')
 fig.text(.5,.866,r'$\hat r_{ui}=\mu+b_u+b_i+p_u^\top q_i$',ha='center',fontsize=23)
 handles=[Line2D([0],[0],color=colors[k],lw=4 if k=='mean_prediction' else 2.2,ls='--' if k=='mu' else '-',label=labels[k]) for k in colors]
 fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.51,.805),ncol=5,frameon=False,fontsize=12,columnspacing=1.6)
 fig.text(.5,.04,note,ha='center',fontsize=12,color='#536471')
 fig.subplots_adjust(left=.065,right=.98,top=.68,bottom=.16,wspace=.16)
 fig.savefig(O/name,dpi=400);fig.savefig(O/name.replace('.png','_preview.png'),dpi=100);plt.close(fig)
 print(name,flush=True)
