from pathlib import Path
import os
os.environ['MPLBACKEND']='Agg'
import pandas as pd,numpy as np,matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
R=Path('/home/moshtasa/Research/phd-recsys/First_paper_material')
f=pd.read_csv(R/'notebook/three_genre_panel_independent_components.csv')
f=f[(f.genre=='Drama')&(f.view=='real_unseen_prediction')]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
counts=[0,2,4,6,25,50,100,200,500]
fig,ax=plt.subplots(figsize=(12,4.7));centers=np.arange(len(counts))*1.35
for idx,n in enumerate(counts):
 for j,model in enumerate(['svd','nmf']):
  d=f[(f.model==model)&(f.n_fake==n)].set_index('group')
  bi=float(d.loc['target','mean_bi']-d.loc['non_target','mean_bi'])
  dot=float(d.loc['target','mean_interaction']-d.loc['non_target','mean_interaction'])
  total=float(d.loc['target','mean_prediction']-d.loc['non_target','mean_prediction'])
  assert np.isclose(total,bi+dot,atol=1e-10)
  x=centers[idx]+(-.29 if j==0 else .29)
  pos=neg=0
  for value,color in [(bi,'#D55E00'),(dot,'#0072B2')]:
   bottom=pos if value>=0 else neg
   ax.bar(x,value,width=.43,bottom=bottom,color=color)
   if value>=0:pos+=value
   else:neg+=value
  ax.scatter(x,total,s=35,c='#009E73',edgecolors='white',linewidths=1,zorder=5)
  ax.annotate(('0.00' if abs(total)<.005 else f'{total:+.2f}'),(x,total),xytext=(0,10),textcoords='offset points',ha='center',fontsize=6.5,color='#007A59',fontweight='bold')
  ax.text(x,-2.12,model.upper(),ha='center',fontsize=7)
ax.axhline(0,color='#536471',lw=.9)
ax.set_xticks(centers,[str(n) for n in counts]);ax.tick_params(axis='x',length=0,pad=25)
ax.set_xlabel('Injected users',labelpad=10)
ax.set_ylabel('Difference in mean score (Drama − Non-Drama)')
ax.set_title('What builds the score advantage for Drama?',loc='left',fontweight='bold',pad=58)
ax.set_ylim(-2.16,2.35);ax.grid(axis='y',alpha=.16);ax.set_axisbelow(True)
ax.legend(handles=[Patch(color='#D55E00',label=r'Book bias: $\Delta\overline{b_i}$'),Patch(color='#0072B2',label=r'Dot product: $\Delta\overline{p_u^\top q_i}$'),Line2D([0],[0],marker='o',color='none',markerfacecolor='#009E73',markersize=8,label=r'Predicted rating: $\Delta\overline{\hat r_{ui}}$')],loc='lower left',bbox_to_anchor=(0,1.02),frameon=False,ncol=3,fontsize=10)
fig.subplots_adjust(left=.08,right=.99,bottom=.23,top=.76)
fig.text(.5,.015,r'$\Delta$ = mean for Drama books − mean for Non-Drama books; averaged over real users.',ha='center',fontsize=9)
out=R/'figures/first_reviewer/drama/prediction_score_gap_example.png'
fig.savefig(out,dpi=220,facecolor='white');print(out)

