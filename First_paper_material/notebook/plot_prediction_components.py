from pathlib import Path
import json
import pandas as pd,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
R=Path('/home/moshtasa/Research/phd-recsys');D=R/'First_paper_material/notebook';O=R/'First_paper_material/figures'
f=pd.read_csv(D/'score_components.csv')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':14,'axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False,'axes.titleweight':'bold','axes.labelcolor':'#163248','text.color':'#163248','xtick.color':'#536471','ytick.color':'#536471','figure.facecolor':'white','savefig.facecolor':'white'})
colors={'mean_prediction':'#009E73','mean_bi':'#D55E00','mean_bu':'#61B8E8','mean_interaction':'#0072B2','mu':'#D7AA00'}
labels={'mean_prediction':r'Predicted rating $\hat r_{ui}$','mean_bi':r'Book bias $b_i$','mean_bu':r'User bias $b_u$','mean_interaction':r'Vector dot product $p_u^\top q_i$','mu':r'Global rating mean $\mu$'}

fig,axs=plt.subplots(2,2,figsize=(16,11.5),sharey='row')
for row,group in enumerate(['target','non_target']):
 for col,model in enumerate(['svd','nmf']):
  ax=axs[row,col];d=f[(f.view=='real_unseen_prediction')&(f.group==group)&(f.model==model)].sort_values('n_fake')
  assert d.n_fake.tolist()==[0,2,4,6,25,50,100,200,500]
  assert np.allclose(d.mean_prediction,d.mu+d.mean_bu+d.mean_bi+d.mean_interaction,atol=1e-10)
  x=np.arange(len(d))
  for key in ['mu','mean_bi','mean_bu','mean_interaction','mean_prediction']:
   ax.plot(x,d[key],color=colors[key],lw=4.2 if key=='mean_prediction' else 2.2,marker='o' if key=='mean_prediction' else None,ms=5,zorder=5 if key=='mean_prediction' else 3,ls='--' if key=='mu' else '-')
  ax.axhline(0,color='#A6B2BB',lw=.7,zorder=1)
  ax.set_title(model.upper()+' | '+('Thriller books' if group=='target' else 'Non-Thriller books'),loc='left',pad=16,fontsize=19)
  ax.set_xticks(x,['Base' if n==0 else str(n) for n in d.n_fake]);ax.set_xlabel('Injected users',labelpad=8);ax.set_ylabel('Mean raw score component');ax.grid(axis='y',color='#DEE5EA',lw=.7);ax.tick_params(length=0,pad=7,labelleft=True)
fig.suptitle('Prediction | Real users, unrated books | Thriller injection',x=.065,ha='left',y=.975,fontsize=24,fontweight='bold')
fig.text(.5,.91,r'$\hat r_{ui}=\mu+b_u+b_i+p_u^\top q_i$',ha='center',fontsize=23)
handles=[Line2D([0],[0],color=colors[k],lw=4 if k=='mean_prediction' else 2.2,ls='--' if k=='mu' else '-',label=labels[k]) for k in colors]
fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.51,.855),ncol=5,frameon=False,fontsize=12,columnspacing=1.6)
fig.text(.5,.025,'Equal weight per real user; average over all unrated books in each group (not only Top-K). Raw scores; no observed ratings for MAE.',ha='center',fontsize=11,color='#536471')
fig.subplots_adjust(left=.065,right=.98,top=.75,bottom=.09,wspace=.16,hspace=.46)
name='prediction_real_users_thriller_non_thriller_4panel.png'
fig.savefig(O/name,dpi=400);fig.savefig(O/name.replace('.png','_preview.png'),dpi=100);plt.close(fig)
print(O/name)
