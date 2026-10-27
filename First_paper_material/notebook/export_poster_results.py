from pathlib import Path
import os,json
os.environ['MPLBACKEND']='Agg'
import pandas as pd,numpy as np,matplotlib.pyplot as plt
R=Path('/home/moshtasa/Research/phd-recsys/First_paper_material')
manifest=json.loads((R/'notebook/poster_svd_nmf_stage1_validation.json').read_text())
f=pd.DataFrame(manifest['results'])
colors={'SVD':'#0072B2','NMF':'#D55E00'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'text.color':'#173243','axes.labelcolor':'#536572','xtick.color':'#536572','ytick.color':'#536572','axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False})
for domain,title,name in [('Books','Books: Thriller','recommendation_books_thriller.png'),('Mobile apps','Mobile Apps: Shopping','recommendation_mobile_shopping.png'),('Movies','Movies: 1990s Release Decade','recommendation_movies_1990s.png')]:
 d=f[f.dataset==domain];ns=sorted(d.n_fake.unique());x=np.arange(len(ns))
 fig,ax=plt.subplots(figsize=(10,5.7))
 for j,model in enumerate(['SVD','NMF']):
  v=d[d.model==model].set_index('n_fake').loc[ns];assert len(v)==len(ns)
  off=-.19 if j==0 else .19
  for k,row in enumerate(v.itertuples()):
   valid=bool(row.sign_ok)
   ax.bar(k+off,row.mean,width=.36,color=colors[model] if valid else 'white',edgecolor=colors[model],hatch=None if valid else '////',linewidth=.8,label=model if k==0 else None,zorder=3)
   if k==len(ns)-1 or (domain!='Books' and k==len(ns)-2) or (domain=='Books' and ns[k]==500):
    ax.text(k+off,row.mean+.20,f'{row.mean:.2f}'+('†' if not valid else ''),ha='center',va='bottom',fontsize=9.5,color=colors[model],fontweight='bold')
 ax.set_ylim(0,16.7);ax.set_yticks([0,5,10,15]);ax.set_xticks(x,[str(n) for n in ns],rotation=45 if len(ns)>8 else 0,ha='right' if len(ns)>8 else 'center')
 ax.set_xlabel('Injected users (0 = baseline)',labelpad=10);ax.set_ylabel('Mean target-category items in Top-15',labelpad=10)
 ax.set_title(title,loc='left',fontweight='bold',fontsize=17,pad=18)
 ax.grid(axis='y',color='#DFE3E6',lw=.7);ax.set_axisbelow(True);ax.tick_params(length=0,pad=7,labelsize=10.5);ax.spines['bottom'].set_color('#AAB4BB')
 ax.legend(frameon=False,ncol=2,loc='upper left')
 b=d[d.n_fake==0].set_index('model')['mean']
 fig.text(.12,.065,f"Baseline: SVD {b['SVD']:.2f} | NMF {b['NMF']:.2f}. Counts shown as categories.",fontsize=9.5,color='#536572')
 if domain=='Books':
  fig.text(.12,.022,'† Hatched NMF bars: raw outputs failing the factor non-negativity check at 1,000, 2,000, 2,500 and 3,000.',fontsize=8.1,color='#A71930')
 fig.subplots_adjust(left=.12,right=.98,top=.87,bottom=.23 if domain=='Books' else .19)
 path=R/'figures'/name;fig.savefig(path,dpi=400,facecolor='white');plt.close(fig);print(path)

