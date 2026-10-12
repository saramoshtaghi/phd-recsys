from pathlib import Path
import argparse, shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--genres',nargs='+',default=['Drama','Thriller']);a=p.parse_args()
r=a.root
identity=pd.read_csv(r/'tables/controlled/genre_summary.csv')
taste=pd.read_csv(r/'tables/neighborhood_taste/genre_table_old_new.csv')
color_limit=0.0
for (_, _), group in taste[(taste.n_fake<=500)&(taste.genre.str.lower()!='unknown')].groupby(['model','target']):
 values=group.pivot(index='genre',columns='n_fake',values='total_percent').sort_index(axis=1)
 changes=values.diff(axis=1).fillna(0)
 color_limit=max(color_limit,float(changes.abs().to_numpy().max()))
color_limit=max(1.0,float(np.ceil(color_limit/5)*5))
blue='#2563A6';green='#218B58';orange='#D97732'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.titlesize':17,'axes.labelsize':13,'savefig.dpi':200,'axes.spines.top':False,'axes.spines.right':False})
for model in ['nmf','svd']:
 for target in a.genres:
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
  fig=plt.figure(figsize=(20,11.25));gs=fig.add_gridspec(2,1,height_ratios=[1,2.4],hspace=.4)
  ax=fig.add_subplot(gs[0]);old=z.old_contribution_pp.to_numpy();new=z.new_contribution_pp.to_numpy();total=z.total_percent.to_numpy()
  ax.bar(x,old,color=blue,label='Retained real neighbors');ax.bar(x,new,bottom=old,color=green,label='Incoming real neighbors')
  for xx,y,o,n in zip(x,total,z.old_share_of_total_percent,z.new_share_of_total_percent):
   ax.text(xx,y+1.5,f'{y:.1f}%\nOld {o:.0f}%\nNew {n:.0f}%',ha='center',fontsize=10)
  ax.set_ylim(0,min(115,max(total)*1.3+8));ax.set_xticks(x,labels);ax.set_ylabel(f'{target} share (%)');ax.set_title('Target genre in neighbors’ own Top-20 recommendations');ax.legend(loc='upper left',bbox_to_anchor=(0,1.3),ncol=2,frameon=False,fontsize=11);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
  matrix=t.pivot(index='genre',columns='n_fake',values='total_percent').loc[:,doses]
  order=[target]+[g for g in matrix.index if g!=target];matrix=matrix.loc[order]
  mini=gs[1].subgridspec(3,5,hspace=.75,wspace=.4)
  for k,g in enumerate(order):
   ax2=fig.add_subplot(mini[k//5,k%5]);v=matrix.loc[g].to_numpy()
   # Equal axes across genres/models; dose levels are categorical and explicitly labeled.
   ax2.plot(x,v,color=blue if g==target else '#5E6E82',linewidth=2.5,marker='o',markersize=4)
   ax2.set_title(g,fontweight='bold' if g==target else 'normal',fontsize=12)
   ax2.set_xticks(x,labels,rotation=60,ha='right',fontsize=8)
   ax2.set_ylim(0,100);ax2.set_yticks([0,50,100]);ax2.tick_params(axis='y',labelsize=9)
   ax2.grid(axis='y',alpha=.2);ax2.set_axisbelow(True)
   if k%5==0:ax2.set_ylabel('Share (%)',fontsize=10)
   if g==target:
    for spine in ax2.spines.values():spine.set_visible(True);spine.set_color('#111111');spine.set_linewidth(2.5)
   ax2.annotate(f'{v[0]:.1f}%',(x[0],v[0]),xytext=(0,8),textcoords='offset points',fontsize=9)
   ax2.annotate(f'{v[-1]:.1f}%',(x[-1],v[-1]),xytext=(-5,8),ha='right',textcoords='offset points',fontsize=9)
  fig.suptitle(f'{model.upper()} | {target} injection: what neighbors are recommended',fontsize=22,y=.98)
  fig.text(.5,.045,'Bars: old + new contributions = final target share. Old/New labels: fraction of that final share.\nLines: each genre’s actual share across injection levels. Multi-genre books can contribute to more than one genre.',ha='center',fontsize=11)
  fig.subplots_adjust(top=.86,bottom=.13,left=.06,right=.97)
  fig.savefig(out/'neighborhood_recommendations_up_to_500.png',facecolor='white');plt.close(fig)
  source=r/'result'/model.upper()/'UMAP'/target/'umap_before_after_500.png'
  assert source.is_file(),source
  (out/'UMAP').mkdir(exist_ok=True);shutil.copy2(source,out/'UMAP/umap_before_after_500.png')
  print(model,target,'baseline',total[0],'500',total[-1],'turnover500',turn[-1])
