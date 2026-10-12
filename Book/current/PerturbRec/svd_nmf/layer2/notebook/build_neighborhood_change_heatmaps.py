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
  fig,ax=plt.subplots(figsize=(16,10))
  # A fixed +/-5 pp display range makes small shifts visible; printed changes retain exact magnitude.
  im=ax.imshow(delta.to_numpy(),cmap='RdBu_r',norm=TwoSlopeNorm(vmin=-5,vcenter=0,vmax=5),aspect='auto')
  doses=values.columns.tolist()
  ax.set_xticks(range(len(doses)),['Baseline']+[str(n) for n in doses[1:]],fontsize=12)
  ax.set_yticks(range(len(values)),values.index,fontsize=13)
  ax.get_yticklabels()[0].set_fontweight('bold')
  for i in range(len(values)):
   for j in range(len(doses)):
    v=values.iloc[i,j];d=delta.iloc[i,j]
    label=f'{v:.1f}%' if j==0 else f'{v:.1f}%\n({d:+.2f} pp)'
    ax.text(j,i,label,ha='center',va='center',fontsize=10,color='white' if abs(d)>3 else '#172535',fontweight='bold' if i==0 else 'normal')
  ax.set_xticks(np.arange(-.5,len(doses),1),minor=True);ax.set_yticks(np.arange(-.5,len(values),1),minor=True)
  ax.grid(which='minor',color='white',linewidth=1.4);ax.tick_params(which='minor',bottom=False,left=False)
  ax.add_patch(Rectangle((-.5,-.5),len(doses),1,fill=False,edgecolor='#111111',linewidth=3,clip_on=False,zorder=10))
  ax.set_xlabel('Injected users',labelpad=12)
  fig.suptitle(f'{model.upper()} | {target}: recommendation changes across injection levels',fontsize=21,y=.96)
  ax.set_title('Warm = increase • Cool = decrease • Each column compared with the preceding column',fontsize=13,pad=18)
  cb=fig.colorbar(im,ax=ax,pad=.02,extend='both',fraction=.035)
  cb.set_label('Change from previous level (percentage points)',fontsize=12)
  fig.text(.5,.055,'Each cell: actual genre share (%) and change from the previous column (pp). Baseline has no preceding comparison.\nColors saturate at ±5 pp; printed changes show the full magnitude. Shares average neighbors’ own Top-20 recommendations.\nAll 20 neighbors are real users. Multi-genre books can contribute to multiple rows.',ha='center',fontsize=11,linespacing=1.5)
  fig.subplots_adjust(top=.86,bottom=.17,left=.12,right=.92)
  out=r/'result/final_results'/model.upper()/target/'neighborhood_recommendation_change_heatmap_up_to_500.png'
  out.parent.mkdir(parents=True,exist_ok=True);fig.savefig(out,facecolor='white');plt.close(fig)
  print(out)
