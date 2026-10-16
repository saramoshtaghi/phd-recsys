"""Combined dataset figure, using verified saved counts; no model training."""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator,StrMethodFormatter
p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
o=json.loads((a.data/'dataset_overview_counts.json').read_text());c=json.loads((a.data/'dataset_category_counts.json').read_text())
colors=['#0072B2','#009E73','#8064A2'];names=['Books','Movies','Mobile apps'];navy='#172B3A'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,'axes.titleweight':'bold','axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8.5,'axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False,'axes.linewidth':.5,'figure.facecolor':'white','savefig.facecolor':'white'})
fig=plt.figure(figsize=(10.4,7.5))
# Left: overall dataset statistics. Right: category distributions.
l=fig.add_gridspec(2,1,left=.095,right=.435,top=.94,bottom=.09,hspace=.46)
r=fig.add_gridspec(3,1,left=.69,right=.965,top=.94,bottom=.09,height_ratios=[13,8,9],hspace=.66)
ax=fig.add_subplot(l[0]);y=np.arange(3)
for offset,key,col,label in [(-.18,'users','#425D72','Users'),(.18,'items','#B2C5D2','Items')]:
 vals=[o[n][key] for n in names];ax.barh(y+offset,vals,height=.30,color=col,label=label,zorder=3)
 for yy,v in zip(y+offset,vals):ax.text(v*1.15,yy,f'{v:,}',va='center',fontsize=8)
ax.set_yticks(y,names);ax.invert_yaxis();ax.set_xscale('log');ax.set_xlim(500,4000000);ax.set_xlabel('Number of users / items (log scale)');ax.set_title('(a) Dataset size',loc='left',color=navy,pad=10);ax.legend(frameon=False,loc='lower right',fontsize=8)
ax=fig.add_subplot(l[1]);x=np.arange(1,6)
for j,(n,col) in enumerate(zip(names,colors)):
 vals=[100*o[n]['rating_counts'][str(k)]/o[n]['observed_ratings'] for k in x]
 ax.bar(x+(j-1)*.24,vals,width=.22,color=col,label=n,zorder=3)
ax.set_xticks(x);ax.set_ylim(0,45);ax.set_xlabel('Rating');ax.set_ylabel('Share of ratings (%)');ax.set_title('(b) Rating distribution',loc='left',color=navy,pad=10);ax.legend(frameon=False,fontsize=8,loc='upper left')
for j,(key,title,col,unit) in enumerate(zip(['books','movies','mobile'],['Books: genres','Movies: release decades','Mobile apps: domains'],colors,['Unique books','Unique movies','Unique apps'])):
 ax=fig.add_subplot(r[j]);counts=c[key]['counts'];labels=list(counts);vals=list(counts.values())
 if key=='movies':labels=[v+'s' for v in labels]
 ax.barh(np.arange(len(vals)),vals,height=.68,color=col,zorder=3);ax.set_yticks(np.arange(len(vals)),labels);ax.invert_yaxis();ax.set_xlim(0,max(vals)*1.20)
 for yy,v in enumerate(vals):ax.text(v+max(vals)*.025,yy,f'{v:,}',va='center',fontsize=8)
 ax.set_title(f'({chr(99+j)}) {title}',loc='left',color=navy,pad=9);ax.set_xlabel(unit,labelpad=3);ax.xaxis.set_major_locator(MaxNLocator(nbins=4,integer=True));ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
for ax in fig.axes:
 ax.tick_params(axis='both',length=0,pad=4,colors='#4E5B65');ax.set_axisbelow(True);ax.spines['bottom'].set_color('#AEBAC2');ax.grid(axis='y' if ax==fig.axes[1] else 'x',color='#DFE5E9',lw=.45)
a.output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.output,dpi=600);fig.savefig(a.output.with_suffix('.preview.png'),dpi=120);plt.close(fig)
