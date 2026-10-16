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
rows=[]
for dataset,target,counts,category,domain in configs:
 for model in ['NMF','SVD']:
  if dataset=='Books':base=R/f'Book/current/PerturbRec/svd_nmf/layer1/result/{model.lower()}'
  elif dataset=='Mobile apps':base=R/f'Mobile/archive/0405/0405_{model}/result'
  else:base=R/f'Movie /current/0519/{model}_0519/result'
  for n in counts:
   stem='ORIGINAL' if n==0 else f'f_{target}_{n}u_pos5_neg1_{domain}'
   bi=read(base/'bi'/f'{stem}_bi_per_{category}.csv');mu=read(base/'mio'/f'{stem}_mu_per_{category}.csv')
   if dataset=='Books':
    bi=bi[bi.n_fake==n];mu=mu[mu.n_fake==n]
    assert (bi.training_protocol=='controlled_clean_catalog').all()
   bi=bi.copy();mu=mu.copy();bi[category]=bi[category].astype(str);mu[category]=mu[category].astype(str)
   bi=bi[~bi[category].str.lower().isin(['unknown','nan','0','0.0'])]
   mu=mu[~mu[category].str.lower().isin(['unknown','nan','0','0.0'])]
   assert not bi[category].duplicated().any() and not mu[category].duplicated().any()
   assert mu.global_mean_mu.nunique()==1
   merged=bi.merge(mu[[category,'global_mean_mu']],on=category,validate='one_to_one')
   assert len(merged)==len(bi)
   for row in merged.to_dict('records'):
    rows.append({'dataset':dataset,'target':target,'model':model,'n_fake':n,'category':row[category],'avg_bi':row['avg_bi'],'n_items':row['n_items'],'global_mean_mu':row['global_mean_mu'],'mu_plus_bi':row['global_mean_mu']+row['avg_bi']})
f=pd.DataFrame(rows);assert np.isfinite(f[['avg_bi','global_mean_mu','mu_plus_bi']]).all().all()
f.to_csv(D/'score_offset_parameters.csv',index=False)
for dataset,_,ns,_,_ in configs:
 a=f[f.dataset==dataset]
 for n in ns:
  vals=a[a.n_fake==n].groupby('model').global_mean_mu.first().to_numpy();assert len(vals)==2 and np.allclose(vals[0],vals[1],atol=1e-10)
 for model in ['NMF','SVD']:
  d=a[a.model==model]
  assert d[d.category==d.target].n_fake.tolist()==ns
  assert d.groupby('n_fake').category.nunique().nunique()==1

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
# Book item-bias figure, NMF and SVD side by side.
b=f[f.dataset=='Books'];ns=configs[0][2];fig,axs=plt.subplots(1,2,figsize=(7.1,2.8),sharey=True)
for ax,model in zip(axs,['NMF','SVD']):
 panel(ax,b[b.model==model],model,'avg_bi',ns,lim(b,'avg_bi'))
 for label in list(ax.texts): label.remove()
 ax.text(.025,1.025,model,color=MODEL[model],transform=ax.transAxes,fontsize=8,weight='bold',va='bottom',clip_on=False)
axs[0].set_ylabel(r'Mean book bias ($b_i$)',color=NAVY);fig.legend(handles=legend,loc='lower center',ncol=3,frameon=False,fontsize=7,bbox_to_anchor=(.52,.01))
fig.subplots_adjust(left=.085,right=.99,top=.88,bottom=.31,wspace=.15);save(fig,'book_thriller_bi_updated.png')
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
for dataset,target,ns,category,domain in configs:
 d=f[f.dataset==dataset];fig,axs=plt.subplots(1,2,figsize=(7.1,2.8),sharey=True)
 for ax,model in zip(axs,['NMF','SVD']):panel(ax,d[d.model==model],model,'mu_plus_bi',ns,lim(d,'mu_plus_bi'))
 axs[0].set_ylabel(r'Mean $\mu+b_i$',color=NAVY);fig.legend(handles=leg,loc='lower center',bbox_to_anchor=(.52,.005),ncol=4,frameon=False,fontsize=7)
 fig.subplots_adjust(left=.085,right=.99,top=.96,bottom=.31,wspace=.15)
 save(fig,{'Books':'book_thriller_mu_bi_updated.png','Mobile apps':'mobile_shopping_mu_bi.png','Movies':'movie_1990_mu_bi.png'}[dataset])
(D/'score_offset_validation.json').write_text(json.dumps({'no_retraining':True,'source_files':SOURCES,'dose_limits':{'Books':500,'Mobile apps':300,'Movies':50},'global_mean_matches_between_models':True,'category_item_bias_means_used_without_reweighting':True,'unknown_categories_excluded_from_category_curves':True,'original_mobile_categories_used_in_parameter_curves':True,'rows':len(f)},indent=2))
# Dataset overview: count original interactions and distinct IDs, rather than tracing old PNG values.
paths={'Books':(R/'Book/Archive/data/genre_clean.csv','user_id','book_id'),'Movies':(R/'Movie /archive/preprocessing/df_final.csv','user_id','item_id'),'Mobile apps':(R/'Mobile/archive/0316_Single/data/data/mobilerec_final.csv','uid','app_package')}
overview={}
for dataset,(file,u,i) in paths.items():
 print('SCANNING',dataset,flush=True);users=set();items=set();ratings=Counter();N=0;total=0.
 for c in pd.read_csv(file,usecols=[u,i,'rating'],chunksize=400000,dtype={u:'str',i:'str'}):
  assert c[[u,i,'rating']].notna().all().all();users.update(c[u]);items.update(c[i]);rr=pd.to_numeric(c.rating);assert rr.isin([1,2,3,4,5]).all();ratings.update({int(k):int(v) for k,v in rr.value_counts().items()});N+=len(c);total+=float(rr.sum())
 model_mu=f[(f.dataset==dataset)&(f.n_fake==0)].global_mean_mu.iloc[0];assert abs(total/N-model_mu)<1e-10,(dataset,total/N,model_mu)
 overview[dataset]={'source':str(file),'source_size_bytes':file.stat().st_size,'users':len(users),'items':len(items),'observed_ratings':N,'global_mean':total/N,'rating_counts':dict(sorted(ratings.items()))}
 print('COUNTS',dataset,overview[dataset],flush=True)
(D/'dataset_overview_counts.json').write_text(json.dumps(overview,indent=2))
fig,axs=plt.subplots(1,2,figsize=(7.1,2.9));names=['Books','Mobile apps','Movies'];x=np.arange(3);w=.34
for off,col,label,color in [(-w/2,'users','Users','#0072B2'),(w/2,'items','Items','#D55E00')]:
 vals=[overview[n][col] for n in names];axs[0].bar(x+off,vals,width=w,color=color,zorder=3,label=label)
 for xx,v in zip(x+off,vals):axs[0].text(xx,v*1.13,f'{v:,}',ha='center',va='bottom',fontsize=6.4,color=NAVY)
axs[0].set_yscale('log');axs[0].set_ylim(500,2e6);axs[0].set_xticks(x,names);axs[0].set_ylabel('Distinct IDs (log scale)',color=NAVY);axs[0].set_title('(a) Users and items',loc='left',color=NAVY);axs[0].legend(frameon=False,fontsize=7,loc='upper left')
for dataset,color,marker in [('Books','#0072B2','o'),('Mobile apps','#8064A2','s'),('Movies','#009E73','^')]:
 vals=[100*overview[dataset]['rating_counts'].get(k,0)/overview[dataset]['observed_ratings'] for k in range(1,6)]
 axs[1].plot(range(1,6),vals,color=color,marker=marker,ms=3,lw=1.2,label=dataset)
axs[1].set_xticks(range(1,6));axs[1].set_xlabel('Observed rating',color=MUTED);axs[1].set_ylabel('Share of ratings (%)',color=NAVY);axs[1].set_ylim(0,45);axs[1].set_title('(b) Rating distribution',loc='left',color=NAVY);axs[1].legend(frameon=False,fontsize=7,loc='upper left')
for ax in axs:
 ax.grid(axis='y',color='#DFE5E9',lw=.45);ax.set_axisbelow(True);ax.tick_params(axis='both',colors=MUTED,length=0,pad=3);ax.spines['bottom'].set_color('#AEBAC2')
fig.subplots_adjust(left=.085,right=.985,top=.90,bottom=.20,wspace=.30);save(fig,'dataset_overview.png')
# Concise captions live outside the image files.
caption='''# Figure captions\n\n## dataset_overview.png\nOriginal dataset characteristics: (a) distinct user and item IDs on a logarithmic scale, and (b) the empirical distribution of observed ratings. Counts refer to the full rating catalog. The movie catalog includes one item with an unknown release decade, which is omitted from the category-distribution figure.\n\n## book_thriller_bi_updated.png\nMean learned book bias under Thriller injection for NMF and SVD. Coloured lines show the target genre and grey lines show individual non-target genre means. Books with multiple labels can contribute to multiple genre means. Injection counts range from baseline to 500 users.\n\n## score_offsets_three_domains.png\nGlobal mean plus category-mean item bias under injection targeting Thriller books, Shopping applications and movies released in the 1990s. NMF and SVD occupy the top and bottom rows. Coloured lines show the target category, grey lines show individual non-target categories, and dashed lines show the global mean. Book injections extend to 500 users, mobile injections to 300, and movie injections to 50. Mobile parameter curves use the original app categories. The global mean is a common offset within each model and does not directly reorder a user's candidate items.\n'''
(O/'remaining_figures_captions.md').write_text(caption)
print('COMPLETE',flush=True)
