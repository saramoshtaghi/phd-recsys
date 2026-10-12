from pathlib import Path
import argparse, shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
r=a.root
identity=pd.read_csv(r/'tables/controlled/genre_summary.csv')
taste=pd.read_csv(r/'tables/neighborhood_taste/genre_table_old_new.csv')
blue='#2563A6';green='#218B58';orange='#D97732'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.titlesize':17,'axes.labelsize':13,'savefig.dpi':200,'axes.spines.top':False,'axes.spines.right':False})
for model in ['nmf','svd']:
 for target in ['Drama','Thriller']:
  out=r/'result/final_results'/model.upper()/target;out.mkdir(parents=True,exist_ok=True)
  d=identity[(identity.model==model)&(identity.injected_genre==target)&(identity.metric=='cosine')&(identity.genre==target)&(identity.n_fake<=500)].sort_values('n_fake')
  doses=[0]+d.n_fake.astype(int).tolist();turn=np.r_[0,d.mean_turnover.to_numpy()]*100
  assert len(doses)==len(set(doses))
  x=np.arange(len(doses));labels=['Baseline']+[str(n) for n in doses[1:]]
  fig,axs=plt.subplots(1,2,figsize=(16,8),sharey=True)
  for ax,v,title,color in zip(axs,[100-turn,turn],['Original neighbors retained','Original neighbors replaced'],[blue,orange]):
   ax.bar(x,v,color=color,width=.68)
   for xx,y in zip(x,v):ax.text(xx,y+1.5,f'{y:.1f}%',ha='center',fontsize=11)
   ax.set_xticks(x,labels,rotation=35,ha='right');ax.set_ylim(0,112);ax.set_yticks(range(0,101,20));ax.set_title(title);ax.set_xlabel('Injected users');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
  axs[0].set_ylabel('Mean percentage of the 20 real neighbors')
  fig.suptitle(f'{model.upper()} | {target} injection: who stays and who changes?',fontsize=22,y=.97)
  fig.text(.5,.045,'Cosine neighbors in the original embedding space • Same real users compared before and after • Fake users excluded from neighbor lists',ha='center',fontsize=11)
  fig.subplots_adjust(top=.85,bottom=.22,wspace=.12)
  fig.savefig(out/'neighbor_identity_up_to_500.png',facecolor='white');plt.close(fig)
  t=taste[(taste.model==model)&(taste.target==target)&(taste.n_fake<=500)&(taste.genre.str.lower()!='unknown')]
  z=t[t.genre==target].set_index('n_fake').loc[doses]
  assert np.allclose(z.old_contribution_pp+z.new_contribution_pp,z.total_percent)
  assert np.allclose(z.neighbor_turnover_percent.to_numpy(),turn,atol=.01)
  fig=plt.figure(figsize=(16,10));gs=fig.add_gridspec(2,1,height_ratios=[1,1.6],hspace=.5)
  ax=fig.add_subplot(gs[0]);old=z.old_contribution_pp.to_numpy();new=z.new_contribution_pp.to_numpy();total=z.total_percent.to_numpy()
  ax.bar(x,old,color=blue,label='Retained real neighbors');ax.bar(x,new,bottom=old,color=green,label='Incoming real neighbors')
  for xx,y,o,n in zip(x,total,z.old_share_of_total_percent,z.new_share_of_total_percent):
   ax.text(xx,y+1.5,f'{y:.1f}%\nOld {o:.0f}%\nNew {n:.0f}%',ha='center',fontsize=10)
  ax.set_ylim(0,min(115,max(total)*1.3+8));ax.set_xticks(x,labels);ax.set_ylabel(f'{target} share (%)');ax.set_title('Target genre in neighbors’ own Top-20 recommendations');ax.legend(loc='upper left',bbox_to_anchor=(0,1.3),ncol=2,frameon=False,fontsize=11);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
  ax2=fig.add_subplot(gs[1]);matrix=t.pivot(index='genre',columns='n_fake',values='total_percent').loc[:,doses]
  order=[target]+[g for g in matrix.index if g!=target];matrix=matrix.loc[order]
  im=ax2.imshow(matrix.to_numpy(),cmap='YlGnBu',vmin=0,vmax=100,aspect='auto')
  ax2.set_xticks(x,labels);ax2.set_yticks(np.arange(len(order)),order);ax2.set_title('All genres: mean recommendation share across the 20 real neighbors',pad=14)
  for i,row in enumerate(matrix.to_numpy()):
   for j,v in enumerate(row):ax2.text(j,i,f'{v:.1f}',ha='center',va='center',color='white' if v>55 else '#162B3B',fontsize=10)
  ax2.add_patch(Rectangle((-.5,-.5),len(doses),1,fill=False,edgecolor='#111111',linewidth=3,clip_on=False,zorder=10))
  ax2.get_yticklabels()[0].set_fontweight('bold')
  fig.colorbar(im,ax=ax2,pad=.015,label='Share (%)')
  fig.suptitle(f'{model.upper()} | {target} injection: what neighbors are recommended',fontsize=22,y=.98)
  fig.text(.5,.045,'Bars: old + new contributions = final target share. Old/New labels: fraction of that final share.\nNeighbors are real users; books already rated are excluded. Multi-genre books may count in multiple rows.',ha='center',fontsize=11)
  fig.subplots_adjust(top=.83,bottom=.13,left=.1,right=.95)
  fig.savefig(out/'neighborhood_recommendations_up_to_500.png',facecolor='white');plt.close(fig)
  source=r/'result'/model.upper()/'UMAP'/target/'umap_before_after_500.png'
  assert source.is_file(),source
  (out/'UMAP').mkdir(exist_ok=True);shutil.copy2(source,out/'UMAP/umap_before_after_500.png')
  print(model,target,'baseline',total[0],'500',total[-1],'turnover500',turn[-1])
