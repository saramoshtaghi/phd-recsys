"""Render all seven Shopping figures after diagnostic measurements complete."""
from pathlib import Path
import os,json
os.environ['MPLBACKEND']='Agg'
import pandas as pd,numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
R=Path('/home/moshtasa/Research/phd-recsys/First_paper_material');O=R/'figures/first_reviewer/shopping';P=O/'paper_figures';P.mkdir(exist_ok=True)
f=pd.read_csv(R/'notebook/shopping_reviewer_components.csv')
C={'svd':'#0072B2','nmf':'#D55E00'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7.5,'axes.titlesize':8,'axes.titleweight':'bold','axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','savefig.facecolor':'white'})
manifest=[]
for view,stem,title in [('all_training','training_shopping_all_users','Training Fit: All Users — Shopping'),('synthetic_training','training_shopping_synthetic_only','Training Fit: Synthetic Users — Shopping'),('real_unseen_prediction','prediction_real_users_shopping','Recommendations for Real Users — Shopping')]:
 d=f[f.view==view]; ns=sorted(d.n_fake.unique()); xx=np.arange(len(ns))
 panels=[('mae','Training MAE'),('mean_bi',r'App bias $b_i$'),('mean_interaction',r'Dot product $p_u^\top q_i$'),('mean_bu',r'User bias $b_u$'),('mean_prediction',r'Prediction $\hat r_{ui}$')]
 if view=='real_unseen_prediction':panels=[panels[-1]]+panels[1:4]
 for compact in [False,True]:
  fig=plt.figure(figsize=(7.2,3.75) if compact else (10,7))
  grid=fig.add_gridspec(2,6 if len(panels)==5 else 2)
  slices=[(0,slice(0,2)),(0,slice(2,4)),(0,slice(4,6)),(1,slice(1,3)),(1,slice(3,5))] if len(panels)==5 else [(0,0),(0,1),(1,0),(1,1)]
  for pi,((key,label),slot) in enumerate(zip(panels,slices)):
   ax=fig.add_subplot(grid[slot])
   for model in C:
    for group,ls,mk in [('target','-','o'),('non_target','--','s')]:
     a=d[(d.model==model)&(d.group==group)].set_index('n_fake').loc[ns]
     assert np.allclose(a.mean_prediction,a.mu+a.mean_bu+a.mean_bi+a.mean_interaction)
     ax.plot(xx,a[key],color=C[model],ls=ls,marker=mk,ms=2.3,lw=1.7 if key=='mean_prediction' else 1.2)
   if key=='mean_prediction' and view!='real_unseen_prediction':
    for group,ls in [('target','-'),('non_target','--')]:
     a=d[(d.model=='svd')&(d.group==group)].set_index('n_fake').loc[ns];ax.plot(xx,a.mean_recorded_rating,color='#777777',ls=ls,lw=.9)
   ax.set_title(f'({chr(65+pi)}) {label}',loc='left');ax.set_xticks(xx,[str(n) for n in ns]);ax.set_xlabel('Injected users');ax.set_ylabel('Mean absolute error' if key=='mae' else 'Mean score component' if key!='mean_prediction' else 'Mean predicted rating');ax.set_axisbelow(True);ax.grid(axis='y',color='#E1E7EB',lw=.5)
  fig.suptitle(title,fontsize=10,fontweight='bold',y=.995)
  handles=[Line2D([],[],color=C[m],label=m.upper()) for m in C]+[Line2D([],[],color='#536471',ls=ls,marker=mk,label=g) for g,ls,mk in [('Shopping','-','o'),('Non-Shopping','--','s')]]
  if view!='real_unseen_prediction':handles.append(Line2D([],[],color='#777777',label='Recorded rating (E)'))
  fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.94),ncol=len(handles),frameon=False,fontsize=7)
  fig.subplots_adjust(left=.09,right=.98,bottom=.105,top=.80,hspace=.95,wspace=1.05 if len(panels)==5 else .4)
  name=stem+'_5panel_paper.png' if compact else (stem+'_overview_'+('4panel' if view=='all_training' else '5panel')+'.png' if view!='real_unseen_prediction' else 'prediction_real_users_shopping_non_shopping_4panel.png')
  path=(P if compact else O)/name;fig.savefig(path,dpi=600 if compact else 300);plt.close(fig);manifest.append(str(path))
keys=['mu','mean_bi','mean_bu','mean_interaction','mean_prediction','mae'];cols=['#D7AA00','#D55E00','#61B8E8','#0072B2','#009E73','#D62728']
fig,axs=plt.subplots(4,2,figsize=(7.2,7))
for section,view in enumerate(['all_training','synthetic_training']):
 for row,group in enumerate(['target','non_target']):
  for col,model in enumerate(C):
   ax=axs[section*2+row,col];d=f[(f.view==view)&(f.group==group)&(f.model==model)].sort_values('n_fake');x=np.arange(len(d))
   for key,color in zip(keys,cols):ax.plot(x,d[key],color=color,lw=1.8 if key=='mean_prediction' else 1,ls='--' if key=='mu' else '-')
   ax.set_title(f'{model.upper()} | '+('Shopping' if group=='target' else 'Non-Shopping')+' | '+('All users' if section==0 else 'Synthetic only'),fontsize=8,loc='left');ax.set_xticks(x,[str(n) for n in d.n_fake]);ax.tick_params(labelsize=6);ax.set_axisbelow(True);ax.grid(axis='y',color='#E1E7EB',lw=.5)
fig.suptitle('Training Fit: All and Synthetic Users — Shopping',fontsize=11,fontweight='bold')
fig.legend(handles=[Line2D([],[],color=c,label=l) for c,l in zip(cols,[r'$\mu$',r'$b_i$',r'$b_u$',r'$p_u^\top q_i$',r'$\hat r_{ui}$','MAE'])],ncol=6,loc='upper center',bbox_to_anchor=(.5,.95),frameon=False)
fig.subplots_adjust(top=.87,hspace=.6,wspace=.3,bottom=.06)
path=O/'training_shopping_all_and_synthetic_8panel_paper.png';fig.savefig(path,dpi=600);plt.close(fig);manifest.append(str(path))
(R/'notebook/shopping_reviewer_figures_manifest.json').write_text(json.dumps(manifest,indent=2))
print('\n'.join(manifest))
