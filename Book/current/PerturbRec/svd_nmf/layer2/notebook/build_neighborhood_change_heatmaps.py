from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);r=p.parse_args().root
all_data=pd.read_csv(r/'tables/neighborhood_taste/genre_table_old_new.csv')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'savefig.dpi':200})
for model in ['nmf','svd']:
 for target in ['Drama','Thriller']:
  t=all_data[(all_data.model==model)&(all_data.target==target)&(all_data.n_fake<=500)&(all_data.genre.str.lower()!='unknown')]
  values=t.pivot(index='genre',columns='n_fake',values='total_percent').sort_index(axis=1)
  values=values.loc[[target]+[g for g in values.index if g!=target]]
  delta=values.diff(axis=1).fillna(0)
  fig=plt.figure(figsize=(16,14));gs=fig.add_gridspec(2,1,height_ratios=[1,4],hspace=.3)
  top=fig.add_subplot(gs[0]);ax=fig.add_subplot(gs[1])
  # A fixed +/-5 pp display range makes small shifts visible; printed changes retain exact magnitude.
  im=ax.imshow(delta.to_numpy(),cmap='RdBu_r',norm=TwoSlopeNorm(vmin=-5,vcenter=0,vmax=5),aspect='auto')
  doses=values.columns.tolist()
  z=t[t.genre==target].set_index('n_fake').loc[doses]
  x=np.arange(len(doses));labels=['Baseline']+[str(n) for n in doses[1:]]
  top.bar(x,z.old_contribution_pp,color='#2563A6',label='Retained real neighbors')
  top.bar(x,z.new_contribution_pp,bottom=z.old_contribution_pp,color='#218B58',label='Incoming real neighbors')
  top.set_ylim(0,min(115,float(z.total_percent.max())*1.3+8));top.set_xticks(x,labels)
  top.set_ylabel(f'{target} share (%)');top.set_title('Target genre in neighbors’ own Top-20 recommendations',fontsize=14)
  top.legend(loc='upper left',bbox_to_anchor=(0,1.3),ncol=2,frameon=False,fontsize=10)
  top.grid(axis='y',alpha=.18);top.set_axisbelow(True);top.spines['top'].set_visible(False);top.spines['right'].set_visible(False)
  ax.set_xticks(range(len(doses)),['Baseline']+[str(n) for n in doses[1:]],fontsize=12)
  ax.set_yticks(range(len(values)),values.index,fontsize=13)
  ax.get_yticklabels()[0].set_fontweight('bold')
  for i in range(len(values)):
   for j in range(len(doses)):
    v=values.iloc[i,j];d=delta.iloc[i,j]
    dark=abs(d)>3
    color='white' if dark else '#172535'
    if j==0:
     ax.text(j,i,f'{v:.1f}%',ha='center',va='center',fontsize=10,color=color,fontweight='bold' if i==0 else 'normal')
    else:
     ax.text(j,i-.14,f'{v:.1f}%',ha='center',va='center',fontsize=10,color=color,fontweight='bold' if i==0 else 'normal')
     ax.text(j,i+.18,f'({d:+.2f})',ha='center',va='center',fontsize=9,color='#D0D0D0' if dark else '#666666')
  ax.set_xticks(np.arange(-.5,len(doses),1),minor=True);ax.set_yticks(np.arange(-.5,len(values),1),minor=True)
  ax.grid(which='minor',color='white',linewidth=1.4);ax.tick_params(which='minor',bottom=False,left=False)
  ax.add_patch(Rectangle((-.5,-.5),len(doses),1,fill=False,edgecolor='#111111',linewidth=3,clip_on=False,zorder=10))
  # Mute all non-target rows while keeping their data and text legible.
  ax.add_patch(Rectangle((-.5,.5),len(doses),len(values)-1,facecolor='#D0D0D0',edgecolor='none',alpha=.62,zorder=2))
  ax.set_xlabel('Injected users',labelpad=12)
  fig.suptitle(f'{model.upper()} | {target}: neighborhood recommendations and genre changes',fontsize=21,y=.97)
  ax.set_title('Warm = increase • Cool = decrease • Each column compared with the preceding column',fontsize=13,pad=18)
  cb=fig.colorbar(im,ax=ax,pad=.02,extend='both',fraction=.035)
  cb.set_label('Change from previous level (percentage points)',fontsize=12)
  fig.text(.5,.045,'Heatmap: actual share (%) and change from the preceding column (gray numbers, in percentage points).\nColors saturate at ±5 percentage points; numbers show full changes. Baseline is neutral.\nBars: retained + incoming contributions = final share. Gray overlay mutes non-target genres. All neighbors are real users.',ha='center',fontsize=11,linespacing=1.5)
  fig.subplots_adjust(top=.87,bottom=.13,left=.12,right=.92)
  out=r/'result/final_results'/model.upper()/target/'neighborhood_recommendation_change_heatmap_up_to_500.png'
  out.parent.mkdir(parents=True,exist_ok=True);fig.savefig(out,facecolor='white');plt.close(fig)
  print(out)
