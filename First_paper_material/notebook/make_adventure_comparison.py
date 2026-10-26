#!/usr/bin/env python3
"""One grouped Adventure comparison, matching the Drama/Thriller PowerPoint style."""
from pathlib import Path
import json
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
R=Path('/home/moshtasa/Research/phd-recsys');NB=R/'First_paper_material/notebook';D=R/'Book/current/PerturbRec/svd_nmf/layer1';OUT=D/'result/final_results/comparision - Adventure_top15_top25_top35_powerpoint.png'
COUNTS=[0,2,4,6,25,50,100,200,500,1000,2000,2500,3000,3500];KS=[15,25,35]
COLORS=['#34495E','#4E79A7','#76B7B2','#59A14F','#AF7AA1','#9C755F','#E15759','#F28E2B','#D4A000','#2A9D8F','#A0CBE8','#FF9DA7','#8064A2','#8A9A5B']
if __name__=='__main__':
 d=pd.read_csv(D/'tables/adventure_recommendation_summary.csv');bad={v['n_fake'] for v in json.loads((NB/'adventure_analysis_status.json').read_text())['nonnegative_constraint_failures']}
 assert set(d.n_fake)==set(COUNTS)
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':14,'axes.labelcolor':'#23415D','text.color':'#17324D','xtick.color':'#23415D','ytick.color':'#607286','axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False})
 fig,axs=plt.subplots(1,2,figsize=(16,9),sharey=True);fig.subplots_adjust(left=.085,right=.975,bottom=.20,top=.71,wspace=.12)
 fig.text(.055,.94,'Adventure recommendations after synthetic-user injection',fontsize=25,fontweight='bold',ha='left',color='#17324D')
 fig.text(.055,.89,'Controlled model runs • Baseline and 13 injection sizes • Top-15, Top-25 and Top-35 recommendations',fontsize=13,color='#607286',ha='left')
 handles=[Patch(facecolor=c,label=('Baseline' if not n else f'{n} users')+(' †' if n in bad else '')) for n,c in zip(COUNTS,COLORS)];fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.52,.855),ncol=7,frameon=False,fontsize=11,handlelength=1.2,columnspacing=1.5,handletextpad=.5)
 x=np.arange(3);width=.058
 for ax,model in zip(axs,['nmf','svd']):
  for j,(n,color) in enumerate(zip(COUNTS,COLORS)):
   values=[]
   for k in KS:
    match=d[(d.model==model)&(d.n_fake==n)&(d.k==k)];assert len(match)==1;v=float(match.iloc[0].mean_target_count);assert 0<=v<=k;values.append(v)
   ax.bar(x+(j-(len(COUNTS)-1)/2)*width,values,width=width*.96,color=color,edgecolor='white',linewidth=.5,zorder=3)
  ax.set_title(model.upper(),fontsize=24,fontweight='bold',pad=16);ax.set_xticks(x,[f'Top-{k}' for k in KS],fontsize=16);ax.set_xlabel('Recommendation list size',fontsize=14,labelpad=18);ax.set_ylim(0,35);ax.set_yticks(np.arange(0,36,5));ax.tick_params(axis='both',length=0);ax.grid(axis='y',color='#E1E8EF',linewidth=.8,zorder=0);ax.spines['bottom'].set_color('#CBD5E1');ax.set_xlim(-.55,2.55)
 axs[0].set_ylabel('Mean Adventure books per real user',fontsize=16,labelpad=12)
 fig.text(.055,.105,'Previously rated books excluded • All injection sizes through 3,500 • Raw scores and factors retained',fontsize=12,color='#23415D')
 fig.text(.055,.065,'† NMF factor sign check failed at 1,000, 2,000 and 2,500 injected users; these raw runs are shown.',fontsize=11,color='#9D211C')
 fig.savefig(OUT,dpi=240,facecolor='white');plt.close(fig);print(OUT)
