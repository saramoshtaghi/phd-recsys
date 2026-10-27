from pathlib import Path
import os,json
os.environ['MPLBACKEND']='Agg'
import numpy as np,matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator,LogFormatterMathtext
R=Path('/home/moshtasa/Research/phd-recsys/First_paper_material')
stats=json.loads((R/'data/dataset_overview_counts.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False,'axes.labelcolor':'#173243','xtick.color':'#536572','ytick.color':'#536572'})
fig,ax=plt.subplots(figsize=(8,5))
names=['Books','Movies','Mobile apps'];x=np.arange(3)
for j,(key,label,color) in enumerate([('users','Users','#0072B2'),('items','Items','#E69F00'),('observed_ratings','Ratings','#009E73')]):
 vals=[stats[n][key] for n in names]
 ax.bar(x+(j-1)*.24,vals,width=.22,color=color,label=label,zorder=3)
ax.set_yscale('log');ax.set_ylim(500,5e7);ax.set_xticks(x,names);ax.set_ylabel('Count (log scale)',labelpad=10)
ax.yaxis.set_major_locator(LogLocator(base=10,numticks=7));ax.yaxis.set_major_formatter(LogFormatterMathtext())
ax.grid(axis='y',color='#DFE3E6',lw=.7);ax.set_axisbelow(True);ax.tick_params(length=0,pad=8);ax.spines['bottom'].set_color('#AAB4BB')
ax.legend(frameon=False,ncol=3,loc='upper left');ax.set_title('Dataset Scale',loc='left',fontweight='bold',fontsize=16,pad=15)
fig.subplots_adjust(left=.12,right=.98,bottom=.13,top=.88)
out=R/'figures/dataset_scale.png';fig.savefig(out,dpi=400,facecolor='white');plt.close(fig);print(out)

