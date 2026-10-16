"""Rebuild remaining first-paper figures from saved data, without training models."""
from pathlib import Path
from collections import Counter
import json,hashlib
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator,StrMethodFormatter
R=Path('/home/moshtasa/Research/phd-recsys');O=R/'First_paper_material/Book';D=O/'data';D.mkdir(parents=True,exist_ok=True)
NAVY='#172B3A';MUTED='#4E5B65';MODEL={'NMF':'#0072B2','SVD':'#D55E00'};GRAY='#BCC5CD'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,'axes.titleweight':'bold','axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,'axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False,'axes.linewidth':.5,'savefig.facecolor':'white','figure.facecolor':'white'})
SOURCES=[]
def read(p):
 SOURCES.append({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()});return pd.read_csv(p)
def save(fig,name):fig.savefig(O/name,dpi=600);plt.close(fig);print('SAVED',name,flush=True)
def style(ax,ns):
 ax.set_xticks(np.arange(len(ns)),['Base']+[str(n) for n in ns[1:]],rotation=45,ha='right');ax.set_xlabel('Injected users',color=MUTED,labelpad=3)
 ax.tick_params(axis='both',length=0,pad=3,colors=MUTED);ax.grid(axis='y',color='#DFE5E9',lw=.45);ax.set_axisbelow(True);ax.spines['bottom'].set_color('#AEBAC2');ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
configs=[('Books','Thriller',[0,2,4,6,25,50,100,200,500],'genre','books'),('Mobile apps','Shopping',[0,30,60,300],'category','allapps'),('Movies','1990',[0,2,6,10,50],'decade','movies')]

f=pd.read_csv(D/'score_offset_parameters.csv')
def panel(ax,d,model,col,ns,limits):
 target=d.target.iloc[0]
 for cat,g in d[d.category!=target].groupby('category'):
  g=g.sort_values('n_fake');assert g.n_fake.tolist()==ns;ax.plot(range(len(ns)),g[col],color=GRAY,lw=.65,alpha=.9,zorder=2)
 t=d[d.category==target].sort_values('n_fake');ax.plot(range(len(ns)),t[col],color=MODEL[model],lw=1.7,marker='o',ms=2.8,zorder=4)
 if col=='mu_plus_bi':ax.plot(range(len(ns)),t.global_mean_mu,ls='--',color='#505A63',lw=.85,zorder=3)
 else:ax.axhline(0,color='#9DA9B2',lw=.55,ls='--',zorder=1)
 ax.set_ylim(*limits);style(ax,ns);ax.text(.025,.94,model,color=MODEL[model],transform=ax.transAxes,fontsize=8,weight='bold',va='top')
def lim(d,col):
 v=d[col].to_numpy()
 if col=='mu_plus_bi':v=np.r_[v,d.global_mean_mu.to_numpy()]
 lo=float(v.min());hi=float(v.max());pad=max((hi-lo)*.14,.1);return lo-pad,hi+pad
legend=[Line2D([0],[0],color=MODEL['NMF'],lw=1.7,label='Target (NMF)'),Line2D([0],[0],color=MODEL['SVD'],lw=1.7,label='Target (SVD)'),Line2D([0],[0],color=GRAY,lw=.9,label='Other category means')]
# Combined score-offset figure and the individual figures expected by the manuscript.
fig,axs=plt.subplots(2,3,figsize=(7.1,4.25))
for ci,(dataset,target,ns,category,domain) in enumerate(configs):
 d=f[f.dataset==dataset];yl=lim(d,'mu_plus_bi')
 for ri,model in enumerate(['NMF','SVD']):
  ax=axs[ri,ci];panel(ax,d[d.model==model],model,'mu_plus_bi',ns,yl)
  # Keep model labels outside the data area, below the dataset headings.
  for label in list(ax.texts): label.remove()
  ax.text(.025,1.025,model,color=MODEL[model],transform=ax.transAxes,fontsize=8,weight='bold',va='bottom',clip_on=False)
  if ri==0:ax.set_title(f'({chr(97+ci)}) {dataset}: '+('1990s' if target=='1990' else target),loc='left',pad=23,color=NAVY)
  if ri==0:ax.set_xlabel('')
  if ci==0:ax.set_ylabel(r'Mean $\mu+b_i$',color=NAVY)
leg=legend+[Line2D([0],[0],color='#505A63',ls='--',lw=.85,label=r'Global mean ($\mu$)')]
fig.legend(handles=leg,loc='lower center',bbox_to_anchor=(.51,.005),ncol=4,frameon=False,fontsize=6.8,handlelength=1.7,columnspacing=1.2)
fig.subplots_adjust(left=.075,right=.99,top=.84,bottom=.205,wspace=.30,hspace=.70);save(fig,'score_offsets_three_domains.png')
