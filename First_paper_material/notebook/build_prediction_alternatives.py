from pathlib import Path
import numpy as np
import pandas as pd
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
 ('mean_prediction',r'(A) Prediction $\hat r_{ui}$'),
 ('mean_bi',r'(B) Book bias $b_i$'),
 ('mean_interaction',r'(C) Dot product $p_u^\top q_i$'),
 ('mean_bu',r'(D) User bias $b_u$'),
]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':5.5,'axes.titlesize':7.1,'axes.titleweight':'bold','xtick.labelsize':4.3,'ytick.labelsize':4.3,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.5,'figure.facecolor':'white','savefig.facecolor':'white'})

def getdata(row):
 domain,label,path,filter_genre=row
 d=pd.read_csv(path)
 if filter_genre: d=d[d.genre.astype(str).str.casefold()==filter_genre.casefold()]
 d=d[d.view=='real_unseen_prediction']
 if d.empty: raise ValueError(label)
 ns=sorted(d.n_fake.unique())
 for _,g in d.groupby(['model','group']):
  if sorted(g.n_fake.unique())!=ns: raise ValueError(f'{label} incomplete')
 return d,ns

def mark_extremes(ax,d,key,ns):
 vals=d[key].to_numpy(float); mask=np.isfinite(vals)&(np.abs(vals)>1e6)
 if not mask.any(): return False
 reg=vals[np.isfinite(vals)&~mask]
 if len(reg):
  lo=float(reg.min());hi=float(reg.max());span=max(hi-lo,.5)
  ax.set_ylim(lo-.16*span,hi+.16*span);yl,yh=ax.get_ylim()
  for model,c in COLORS.items():
   if model not in set(d.model): continue
   for group,_,_ in GROUPS:
    a=d[(d.model==model)&(d.group==group)].set_index('n_fake').loc[ns]
    for x,v in enumerate(a[key].to_numpy(float)):
     if np.isfinite(v) and abs(v)>1e6:
      ax.scatter([x],[yh if v>0 else yl],marker='^' if v>0 else 'v',s=13,color=c,edgecolors='white',linewidths=.25,zorder=8,clip_on=False)
 return True

def format_ax(ax,ns,category,first_col):
 x=np.arange(len(ns));ax.set_xlim(-.25,len(ns)-.75 if len(ns)>1 else .25)
 ax.set_xticks(x,[str(int(v)) for v in ns]);ax.tick_params(axis='both',length=2,pad=1)
 ax.grid(axis='y',color='#E1E7EB',lw=.42);ax.set_axisbelow(True)
 if first_col: ax.set_ylabel(category,rotation=0,ha='right',va='center',labelpad=6,fontweight='bold',fontsize=5.7)
 if category in ('Adventure','1920s','Beauty'): ax.set_xlabel('Injected users',labelpad=1,fontsize=4.8)

# Version 2: separate algorithm columns; each panel has only target and other-category traces.
fig,axs=plt.subplots(9,8,figsize=(17.2,10.2),squeeze=False)
for ri,row in enumerate(rows):
 d,ns=getdata(row)
 for mi,model in enumerate(['svd','nmf']):
  for pi,(key,label) in enumerate(PANELS):
   ci=mi*4+pi;ax=axs[ri,ci];col=COLORS[model]
   for group,ls,mk in GROUPS:
    a=d[(d.model==model)&(d.group==group)].set_index('n_fake').loc[ns]
    if key=='mean_prediction': assert np.allclose(a.mean_prediction,a.mu+a.mean_bu+a.mean_bi+a.mean_interaction,atol=2e-5)
    ax.plot(np.arange(len(ns)),a[key].to_numpy(),color=col,ls=ls,marker=mk,ms=2.0,lw=.9)
   mark_extremes(ax,d[d.model==model],key,ns)
   format_ax(ax,ns,row[1],ci==0)
   if ri==0: ax.set_title(f'{model.upper()}  {label}',loc='left',pad=4)
   if ci==3 or ci==7: ax.spines['right'].set_visible(True);ax.spines['right'].set_color('#AFBAC2')
   if ri==3 and ci==0: ax.text(-.62,.5,'MOVIES',transform=ax.transAxes,rotation=90,ha='center',va='center',fontsize=7.2,fontweight='bold',color='#43505B',clip_on=False)
   if ri==6 and ci==0: ax.text(-.62,.5,'MOBILE APPS',transform=ax.transAxes,rotation=90,ha='center',va='center',fontsize=7.2,fontweight='bold',color='#43505B',clip_on=False)
   if ri==0 and ci==0: ax.text(-.62,.5,'BOOKS',transform=ax.transAxes,rotation=90,ha='center',va='center',fontsize=7.2,fontweight='bold',color='#43505B',clip_on=False)
fig.suptitle('Recommendations for Real Users — SVD and NMF Separated',fontsize=12,fontweight='bold',y=.995)
legend=[Line2D([],[],color='#4D5B66',ls=ls,marker=mk,ms=3,label=grp.replace('_',' ').title()) for grp,ls,mk in GROUPS]
legend.append(Line2D([],[],color='#7A263A',marker='^',ls='none',ms=4,label='Off-scale value (>10⁶)'))
fig.legend(handles=legend,loc='upper center',bbox_to_anchor=(.5,.977),ncol=3,frameon=False,fontsize=6.2)
fig.text(.5,.012,'Off-scale markers denote extreme values (>10⁶); source values are unchanged.',ha='center',fontsize=5.8,color='#7A263A')
fig.subplots_adjust(left=.09,right=.995,top=.938,bottom=.06,hspace=.52,wspace=.19)
p=OUT/'prediction_real_users_all_9_categories_2.png';fig.savefig(p,dpi=500,bbox_inches='tight');plt.close(fig);print(p,p.stat().st_size)

# Version 3: point-only panels; exact dose observations are shown without interpolating lines.
fig,axs=plt.subplots(9,4,figsize=(10.4,10.2),squeeze=False)
for ri,row in enumerate(rows):
 d,ns=getdata(row)
 x=np.arange(len(ns))
 for pi,(key,label) in enumerate(PANELS):
  ax=axs[ri,pi]
  for mi,(model,col) in enumerate(COLORS.items()):
   jitter=(-.075 if model=='svd' else .075)
   for group,ls,mk in GROUPS:
    a=d[(d.model==model)&(d.group==group)].set_index('n_fake').loc[ns]
    if key=='mean_prediction': assert np.allclose(a.mean_prediction,a.mu+a.mean_bu+a.mean_bi+a.mean_interaction,atol=2e-5)
    vals=a[key].to_numpy(float);valid=np.isfinite(vals)&(np.abs(vals)<=1e6)
    ax.scatter(x[valid]+jitter,vals[valid],s=11,marker=mk,color=col,edgecolors='white',linewidths=.25,zorder=3)
   # Draw clipped triangles for numerical-divergence observations.
  ext=mark_extremes(ax,d,key,ns)
  format_ax(ax,ns,row[1],pi==0)
  if ri==0: ax.set_title(label,pad=4,loc='left')
  if ri==3 and pi==0: ax.text(-.57,.5,'MOVIES',transform=ax.transAxes,rotation=90,ha='center',va='center',fontsize=7.2,fontweight='bold',color='#43505B',clip_on=False)
  if ri==6 and pi==0: ax.text(-.57,.5,'MOBILE APPS',transform=ax.transAxes,rotation=90,ha='center',va='center',fontsize=7.2,fontweight='bold',color='#43505B',clip_on=False)
  if ri==0 and pi==0: ax.text(-.57,.5,'BOOKS',transform=ax.transAxes,rotation=90,ha='center',va='center',fontsize=7.2,fontweight='bold',color='#43505B',clip_on=False)
fig.suptitle('Recommendations for Real Users — Individual Dose Observations',fontsize=12,fontweight='bold',y=.995)
legend=[Line2D([],[],color=COLORS[m],marker='o',ls='none',ms=4,label=m.upper()) for m in ['svd','nmf']]
legend += [Line2D([],[],color='#4D5B66',marker=mk,ls='none',ms=4,label=grp.replace('_',' ').title()) for grp,_,mk in GROUPS]
legend.append(Line2D([],[],color='#7A263A',marker='^',ls='none',ms=4,label='Off-scale value (>10⁶)'))
fig.legend(handles=legend,loc='upper center',bbox_to_anchor=(.5,.977),ncol=5,frameon=False,fontsize=6.3)
fig.text(.5,.012,'Points show observed dose levels; no values are interpolated. Off-scale markers denote values >10⁶.',ha='center',fontsize=5.8,color='#7A263A')
fig.subplots_adjust(left=.12,right=.995,top=.938,bottom=.06,hspace=.52,wspace=.19)
p=OUT/'prediction_real_users_all_9_categories_3.png';fig.savefig(p,dpi=500,bbox_inches='tight');plt.close(fig);print(p,p.stat().st_size)
