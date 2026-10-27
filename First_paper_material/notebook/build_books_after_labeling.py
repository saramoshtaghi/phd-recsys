from pathlib import Path
import os,json
os.environ['MPLBACKEND']='Agg'
import numpy as np,matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
R=Path('/home/moshtasa/Research/phd-recsys/First_paper_material')
data=json.loads((R/'data/dataset_category_counts.json').read_text())['books']
counts=data['counts'];labels=list(counts);values=list(counts.values())
assert data['n_unique_items']==9415 and len(labels)==13 and counts['Thriller']==1606
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False,'text.color':'#173243','axes.labelcolor':'#536572','xtick.color':'#536572','ytick.color':'#536572'})
fig,ax=plt.subplots(figsize=(8.6,6.4));y=np.arange(len(labels))
ax.barh(y,values,height=.66,color=['#CF102D' if s=='Thriller' else '#6395B3' for s in labels],zorder=3)
ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_xlim(0,3500)
for pos,val in zip(y,values):ax.text(val+45,pos,f'{val:,}',va='center',fontsize=10.5,color='#173243')
for label in ax.get_yticklabels():
 if label.get_text()=='Thriller':label.set_color('#CF102D');label.set_fontweight('bold')
ax.set_xlabel('Number of unique books',labelpad=10);ax.xaxis.set_major_locator(MaxNLocator(nbins=7,integer=True))
ax.grid(axis='x',color='#DFE3E6',lw=.7);ax.set_axisbelow(True);ax.tick_params(length=0,pad=7);ax.spines['bottom'].set_color('#AAB4BB')
fig.text(.245,.965,'Books After Genre Labeling',fontsize=17,fontweight='bold',ha='left',va='top')
fig.text(.245,.915,'9,415 books | 13 GPT-3.5-assigned genres',fontsize=10.5,color='#536572',ha='left',va='top')
fig.text(.5,.025,'Books may have up to two genre labels; category counts overlap.',ha='center',fontsize=9.5,color='#536572')
fig.subplots_adjust(left=.245,right=.97,bottom=.13,top=.86)
out=R/'figures/books_after_labeling.png';fig.savefig(out,dpi=400,facecolor='white');plt.close(fig);print(out)

