from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT=Path('/home/moshtasa/Research/phd-recsys/First_paper_material')
NB=ROOT/'notebook'; OUT=ROOT/'figures'
rows=[
 ('Books','Thriller',NB/'three_genre_panel_independent_components.csv','Thriller'),
 ('Books','Drama',NB/'three_genre_panel_independent_components.csv','Drama'),
 ('Books','Adventure',NB/'three_genre_panel_independent_components.csv','Adventure'),
 ('Movies','1990s',NB/'movies_1990_reviewer_components.csv',None),
 ('Movies','1950s',NB/'movies_1950_reviewer_components.csv',None),
 ('Movies','1920s',NB/'movies_1920_reviewer_components.csv',None),
 ('Mobile apps','Shopping',NB/'shopping_reviewer_components.csv',None),
 ('Mobile apps','Casual',NB/'casual_reviewer_components.csv',None),
 ('Mobile apps','Beauty',NB/'beauty_reviewer_components.csv',None),
]
COLORS={'svd':'#0072B2','nmf':'#D55E00'}
GROUPS=[('target','-','o'),('non_target','--','s')]
PANELS=[
 ('mean_prediction',r'(A) Prediction $\hat r_{ui}$','Mean predicted rating'),
 ('mean_bi',r'(B) Book bias $b_i$','Mean score component'),
 ('mean_interaction',r'(C) Dot product $p_u^\top q_i$','Mean score component'),
 ('mean_bu',r'(D) User bias $b_u$','Mean score component'),
]
plt.rcParams.update({
 'font.family':'DejaVu Sans','font.size':6.2,'axes.titlesize':8.4,'axes.titleweight':'bold',
 'axes.labelsize':5.4,'xtick.labelsize':4.7,'ytick.labelsize':4.7,
 'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':0.55,
 'figure.facecolor':'white','savefig.facecolor':'white',
})
def load_row(row):
 domain,label,path,genre=row
 d=pd.read_csv(path)
 if genre is not None:
  d=d[d.genre.astype(str).str.casefold()==genre.casefold()]
 d=d[d.view=='real_unseen_prediction'].copy()
 ns=sorted(d.n_fake.unique())
 return d,ns

fig,axs=plt.subplots(9,4,figsize=(10.4,10.2),squeeze=False)
for ri,row in enumerate(rows):
 domain,label,_,_=row
 d,ns=load_row(row)
 x=np.arange(len(ns))
 for ci,(key,heading,ylabel) in enumerate(PANELS):
  ax=axs[ri,ci]
  for model,color in COLORS.items():
   for group,ls,mk in GROUPS:
    a=d[(d.model==model)&(d.group==group)].set_index('n_fake').loc[ns]
    if key=='mean_prediction':
     assert np.allclose(a.mean_prediction,a.mu+a.mean_bu+a.mean_bi+a.mean_interaction,atol=2e-5)
    ax.plot(x,a[key].to_numpy(dtype=float),color=color,ls=ls,marker=mk,ms=2.3,lw=1.0,solid_capstyle='round')
  if ri>=6:
   # Symmetric log keeps ordinary predictions legible and also displays signed,
   # extremely large MobileRec outputs instead of clipping them off the panel.
   ax.set_yscale('symlog',linthresh=0.5,linscale=0.7,base=10)
   ylo,yhi=ax.get_ylim()
   vals=d[key].to_numpy(dtype=float)
   maxabs=float(np.nanmax(np.abs(vals)))
   maxexp=max(1,int(np.ceil(np.log10(maxabs)))) if maxabs>0 else 1
   exps=list(range(6,maxexp+1,6))
   if maxexp not in exps and maxexp>=2: exps.append(maxexp)
   ticks=[-1.0,0.0,1.0]
   for e in sorted(set(exps)):
    ticks.extend([-10.0**e,10.0**e])
   ticks=[t for t in ticks if ylo<=t<=yhi]
   ax.set_yticks(ticks)
   ticklabels=[]
   for t in ticks:
    if t==0: ticklabels.append('0')
    elif abs(t)==1: ticklabels.append('−1' if t<0 else '1')
    else: ticklabels.append(('−' if t<0 else '')+f'1e{int(np.log10(abs(t)))}')
   ax.set_yticklabels(ticklabels)
   ax.set_ylim(ylo,yhi)
  ax.set_xlim(-.25,len(ns)-.75 if len(ns)>1 else .25)
  ax.set_xticks(x,[str(int(n)) for n in ns])
  ax.tick_params(axis='both',length=2,pad=1)
  ax.grid(axis='y',color='#E1E7EB',lw=.45);ax.set_axisbelow(True)
  if ri==0: ax.set_title(heading,pad=4,loc='left')
  if ci==0:
   ax.set_ylabel(label,rotation=0,ha='right',va='center',labelpad=7,fontweight='bold',fontsize=6.5)
  else: ax.set_ylabel('')
  ax.set_xlabel('Injected users' if ri in (2,5,8) else '',labelpad=1)
  if ri==3 and ci==0:
   ax.text(-0.57,0.5,'MOVIES',transform=ax.transAxes,rotation=90,ha='center',va='center',fontsize=8,fontweight='bold',color='#43505B',clip_on=False)
  if ri==6 and ci==0:
   ax.text(-0.57,0.5,'MOBILE APPS',transform=ax.transAxes,rotation=90,ha='center',va='center',fontsize=8,fontweight='bold',color='#43505B',clip_on=False)
  if ri==0 and ci==0:
   ax.text(-0.57,0.5,'BOOKS',transform=ax.transAxes,rotation=90,ha='center',va='center',fontsize=8,fontweight='bold',color='#43505B',clip_on=False)
handles=[
 Line2D([],[],color=COLORS['svd'],lw=1.6,label='SVD'),
 Line2D([],[],color=COLORS['nmf'],lw=1.6,label='NMF'),
 Line2D([],[],color='#4D5B66',ls='-',marker='o',ms=3,label='Target category'),
 Line2D([],[],color='#4D5B66',ls='--',marker='s',ms=3,label='Other categories')
]
fig.suptitle('Recommendations for Real Users',fontsize=13,fontweight='bold',y=.995)
fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.52,.978),ncol=4,frameon=False,fontsize=6.8,handlelength=1.7,columnspacing=1.2)
fig.text(.5,.012,'Books and movies use linear y-axes. Mobile-app y-axes use symmetric log spacing (linthresh = 0.5) to show large positive and negative outputs without clipping.',ha='center',fontsize=6,color='#43505B')
fig.subplots_adjust(left=.12,right=.995,top=.946,bottom=.065,hspace=.56,wspace=.16)
out=OUT/'prediction_real_users_all_9_categories_mobile_symlog.png'
fig.savefig(out,dpi=500,bbox_inches='tight')
plt.close(fig)
print(out,out.stat().st_size)
