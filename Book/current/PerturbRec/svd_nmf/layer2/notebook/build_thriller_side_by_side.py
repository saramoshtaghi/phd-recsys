from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);r=p.parse_args().root
d=pd.read_csv(r/'tables/neighborhood_taste/genre_table_old_new.csv')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'savefig.dpi':220,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(1,2,figsize=(20,8),sharey=True)
for ax,model in zip(axs,['nmf','svd']):
 z=d[(d.model==model)&(d.target=='Thriller')&(d.genre=='Thriller')&(d.n_fake<=500)].sort_values('n_fake')
 x=np.arange(len(z));labels=['Baseline' if n==0 else str(n) for n in z.n_fake]
 assert np.allclose(z.old_contribution_pp+z.new_contribution_pp,z.total_percent)
 ax.bar(x,z.old_contribution_pp,color='#2563A6',label='Retained real neighbors')
 ax.bar(x,z.new_contribution_pp,bottom=z.old_contribution_pp,color='#218B58',label='Incoming real neighbors')
 for j,row in enumerate(z.itertuples()):
  ax.text(j,row.total_percent+1.4,f'{row.total_percent:.1f}%\nOld {row.old_share_of_total_percent:.0f}%\nNew {row.new_share_of_total_percent:.0f}%',ha='center',fontsize=10,linespacing=1.2)
 ax.set_xticks(x,labels,rotation=35,ha='right');ax.set_ylim(0,100);ax.set_yticks(range(0,101,20));ax.set_title(model.upper(),fontsize=22,fontweight='bold',pad=15)
 ax.set_xlabel('Injected users');ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
axs[0].set_ylabel('Mean Thriller share in neighbors’ Top-20 recommendations (%)')
fig.suptitle('Thriller injection | NMF versus SVD',fontsize=25,y=.97)
handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.91),ncol=2,frameon=False,fontsize=13)
fig.text(.5,.055,'Bar height = total Thriller share. Blue + green = retained + incoming contributions.\nOld/New labels show their fractions of that final share; incoming neighbors are real users, not injected users.\nEach user has 20 real cosine neighbors. Shares use those neighbors’ own Top-20 unseen-book recommendations.',ha='center',fontsize=12,linespacing=1.5)
fig.subplots_adjust(top=.79,bottom=.25,left=.07,right=.98,wspace=.1)
out=r/'result/final_results/SVD/Thriller/neighborhood_recommendations_up_to_500.png'
fig.savefig(out,facecolor='white');print(out)
