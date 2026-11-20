from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT=Path('/home/moshtasa/Research/phd-recsys/First_paper_material')
NB=ROOT/'notebook'
OUT=ROOT/'figures/first_reviewer'
OUT.mkdir(parents=True,exist_ok=True)

# Nine established representative categories, grouped by dataset.
rows=[
 ('Books','Thriller','thriller',NB/'three_genre_panel_independent_components.csv','Thriller'),
 ('Books','Drama','drama',NB/'three_genre_panel_independent_components.csv','Drama'),
 ('Books','Adventure','advanture',NB/'three_genre_panel_independent_components.csv','Adventure'),
 ('Movies','1990s','1990',NB/'movies_1990_reviewer_components.csv',None),
 ('Movies','1950s','1950',NB/'movies_1950_reviewer_components.csv',None),
 ('Movies','1920s','1920',NB/'movies_1920_reviewer_components.csv',None),
 ('Mobile apps','Shopping','shopping',NB/'shopping_reviewer_components.csv',None),
 ('Mobile apps','Casual','casual',NB/'casual_reviewer_components.csv',None),
 ('Mobile apps','Beauty','beauty',NB/'beauty_reviewer_components.csv',None),
]
COLORS={'svd':'#0072B2','nmf':'#D55E00'}
GROUPS=[('target','-','o'),('non_target','--','s')]
PRED_PANELS=[
 ('mean_prediction',r'(A) Prediction $\hat r_{ui}$','Mean predicted rating'),
 ('mean_bi',r'(B) Book bias $b_i$','Mean score component'),
 ('mean_interaction',r'(C) Dot product $p_u^\top q_i$','Mean score component'),
 ('mean_bu',r'(D) User bias $b_u$','Mean score component'),
]
TRAIN_PANELS=[
 ('mae','(A) Training MAE','Mean absolute error'),
 ('mean_bi',r'(B) Book bias $b_i$','Mean score component'),
 ('mean_interaction',r'(C) Dot product $p_u^\top q_i$','Mean score component'),
 ('mean_bu',r'(D) User bias $b_u$','Mean score component'),
 ('mean_prediction',r'(E) Prediction $\hat r_{ui}$','Mean predicted rating'),
]

plt.rcParams.update({
 'font.family':'DejaVu Sans','font.size':6.2,'axes.titlesize':8.4,'axes.titleweight':'bold',
 'axes.labelsize':5.4,'xtick.labelsize':4.7,'ytick.labelsize':4.7,
 'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':0.55,
 'figure.facecolor':'white','savefig.facecolor':'white',
})

def load_row(row,view):
 domain,label,slug,path,filter_genre=row
 d=pd.read_csv(path)
 if filter_genre is not None:
  d=d[d.genre.astype(str).str.casefold()==filter_genre.casefold()]
 d=d[d.view==view].copy()
 if d.empty: raise ValueError(f'No rows: {label} / {view}')
 # Every category must provide the same model/group/dose grid for each chosen panel.
 ns=sorted(d.n_fake.unique())
 if len(ns)<2: raise ValueError(f'Insufficient dose levels: {label} / {view}')
 for _,g in d.groupby(['model','group']):
  if sorted(g.n_fake.unique()) != ns: raise ValueError(f'Incomplete doses for {label} / {view}')
 return d,ns

def draw(view,panels,filename,title):
 ncols=len(panels)
 fig,axs=plt.subplots(9,ncols,figsize=(16.5 if ncols==5 else 14.2,10.2),squeeze=False)
 for ri,row in enumerate(rows):
  d,ns=load_row(row,view)
  x=np.arange(len(ns))
  for ci,(key,heading,ylabel) in enumerate(panels):
   ax=axs[ri,ci]
   for model,color in COLORS.items():
    for group,ls,mk in GROUPS:
     a=d[(d.model==model)&(d.group==group)].set_index('n_fake').loc[ns]
     if key=='mean_prediction':
      assert np.allclose(a.mean_prediction,a.mu+a.mean_bu+a.mean_bi+a.mean_interaction,atol=2e-5)
     ax.plot(x,a[key].to_numpy(),color=color,ls=ls,marker=mk,ms=2.3,lw=1.0,solid_capstyle='round')
    # Recorded ratings only apply to observed training ratings; retain the original gray reference.
    if key=='mean_prediction' and view!='real_unseen_prediction':
     a=d[(d.model==model)&(d.group=='target')].set_index('n_fake').loc[ns]
     # In the source figures this reference was rendered once (gray) for the two rating groups.
     if model=='svd':
      for group,ls in [('target','-'),('non_target','--')]:
       b=d[(d.model=='svd')&(d.group==group)].set_index('n_fake').loc[ns]
       ax.plot(x,b.mean_recorded_rating.to_numpy(),color='#777777',ls=ls,lw=.75,zorder=1)
   # Preserve extreme outputs without letting a numerical blow-up flatten every
   # ordinary value in the same small panel. The original values remain in CSV.
   values=d[key].to_numpy(dtype=float)
   extreme=np.isfinite(values)&(np.abs(values)>1e6)
   if extreme.any():
    regular=values[np.isfinite(values)&(~extreme)]
    if len(regular):
     lo=float(np.min(regular)); hi=float(np.max(regular)); span=max(hi-lo,0.5)
     ax.set_ylim(lo-.16*span,hi+.16*span)
     ylo,yhi=ax.get_ylim()
     for model,color in COLORS.items():
      for group,ls,mk in GROUPS:
       a=d[(d.model==model)&(d.group==group)].set_index('n_fake').loc[ns]
       arr=a[key].to_numpy(dtype=float)
       for xi,val in zip(x,arr):
        if np.isfinite(val) and abs(val)>1e6:
         edge=yhi if val>0 else ylo
         ax.scatter([xi],[edge],marker='^' if val>0 else 'v',s=15,color=color,edgecolors='white',linewidths=.25,zorder=8,clip_on=False)
   ax.set_xlim(-.25,len(ns)-.75 if len(ns)>1 else .25)
   ax.set_xticks(x,[str(int(n)) for n in ns])
   ax.tick_params(axis='both',length=2,pad=1)
   ax.grid(axis='y',color='#E1E7EB',lw=.45)
   ax.set_axisbelow(True)
   if ri==0:
    ax.set_title(heading,pad=4,loc='left')
   if ci==0:
    ax.set_ylabel(row[1],rotation=0,ha='right',va='center',labelpad=7,fontweight='bold',fontsize=6.5)
   else:
    ax.set_ylabel('')
   # Put x label on the final row in each dataset block so all injected-user scales are legible.
   if ri in (2,5,8):
    ax.set_xlabel('Injected users',labelpad=1)
   else:
    ax.set_xlabel('')
   if ri==3 and ci==0:
    ax.text(-0.57,0.5,'MOVIES',transform=ax.transAxes,rotation=90,ha='center',va='center',fontsize=8,fontweight='bold',color='#43505B',clip_on=False)
   if ri==6 and ci==0:
    ax.text(-0.57,0.5,'MOBILE APPS',transform=ax.transAxes,rotation=90,ha='center',va='center',fontsize=8,fontweight='bold',color='#43505B',clip_on=False)
   if ri==0 and ci==0:
    ax.text(-0.57,0.5,'BOOKS',transform=ax.transAxes,rotation=90,ha='center',va='center',fontsize=8,fontweight='bold',color='#43505B',clip_on=False)
 # A clean full-width title and source-consistent legend.
 fig.suptitle(title,fontsize=13,fontweight='bold',y=.995)
 handles=[Line2D([],[],color=COLORS['svd'],lw=1.6,label='SVD'),Line2D([],[],color=COLORS['nmf'],lw=1.6,label='NMF'),
  Line2D([],[],color='#4D5B66',ls='-',marker='o',ms=3,label='Target category'),
  Line2D([],[],color='#4D5B66',ls='--',marker='s',ms=3,label='Other categories')]
 if view!='real_unseen_prediction':
  handles.append(Line2D([],[],color='#777777',lw=.8,label='Recorded rating'))
 handles.append(Line2D([],[],color='#7A263A',marker='^',ls='none',ms=4,label='Off-scale divergence (>10⁶)'))
 fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.52,.978),ncol=len(handles),frameon=False,fontsize=6.8,handlelength=1.7,columnspacing=1.2)
 fig.text(.5,.012,'Off-scale markers denote extreme values (>10⁶); the plotted values remain in the source data.',ha='center',fontsize=6,color='#7A263A')
 fig.subplots_adjust(left=.12,right=.995,top=.946,bottom=.065,hspace=.56,wspace=.34)
 path=OUT/filename
 fig.savefig(path,dpi=500,bbox_inches='tight')
 plt.close(fig)
 print(path, path.stat().st_size)

# Prediction figures intentionally contain the same four panels as the category source figures.
draw('real_unseen_prediction',PRED_PANELS,'prediction_real_users_all_9_categories.png','Recommendations for Real Users')
draw('all_training',TRAIN_PANELS,'training_all_users_9_categories.png','Training Fit: All Users')
draw('synthetic_training',TRAIN_PANELS,'training_synthetic_only_9_categories.png','Training Fit: Synthetic Users')
