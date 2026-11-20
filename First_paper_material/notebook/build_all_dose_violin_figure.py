import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[k]='1'
from pathlib import Path
import json, gc
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

R=Path('/home/moshtasa/Research/phd-recsys')
M=R/'First_paper_material'
NB=M/'notebook'
OUT=M/'figures'
TARGET=OUT/'prediction_real_users_all_9_categories_violin.png'
STATS_PATH=NB/'prediction_real_users_all_9_categories_violin_stats.csv'
META_PATH=NB/'prediction_real_users_all_9_categories_violin_metadata.json'
BOOK_DOSES=[0,2,4,6,25,50,100,200,500,1000,2000,2500,3000]
MOVIE_DOSES={'1990s':[0,2,6,10,50],'1950s':[0,2,6,10,50],'1920s':[0,1,2,4,6,10]}
APP_DOSES=[0,30,60,300,500,2000,7000]
CATEGORIES={'Books':['Thriller','Drama','Adventure'],
            'Movies':['1990s','1950s','1920s'],
            'Mobile apps':['Shopping','Casual','Beauty']}
COLORS={('svd','target'):'#0072B2',
        ('svd','non_target'):'#8CC5E5',
        ('nmf','target'):'#D55E00',
        ('nmf','non_target'):'#F2AC79'}
LABELS={('svd','target'):'SVD · target',
        ('svd','non_target'):'SVD · other',
        ('nmf','target'):'NMF · target',
        ('nmf','non_target'):'NMF · other'}
YMIN,YMAX=-2.0,7.0
STATS=[]
DISTS={}
N_REFERENCE_VALIDATED=0
N_REFERENCE_MISSING=0

def read_domain(domain):
    if domain=='Books':
        p=R/'Book/Archive/data/genre_clean.csv'
        df=pd.read_csv(p,usecols=['user_id','book_id','rating','genres'],
                       dtype={'user_id':'int64','book_id':'int64','rating':'float32','genres':'string'})
        uc,ic,lc='user_id','book_id','genres'
    elif domain=='Movies':
        p=R/'Movie /archive/preprocessing/df_final.csv'
        df=pd.read_csv(p,usecols=['user_id','item_id','rating','decade'])
        df=df.dropna(subset=['user_id','item_id','rating'])
        df['decade']=pd.to_numeric(df.decade,errors='coerce').fillna(0).astype(int).astype(str)+'s'
        uc,ic,lc='user_id','item_id','decade'
    else:
        p=R/'Mobile/archive/0316_Single/data/data/mobilerec_final.csv'
        df=pd.read_csv(p,usecols=['uid','app_package','rating','app_category'],
                       dtype={'uid':str,'app_package':str})
        df=df.dropna(subset=['uid','app_package','rating'])
        df['rating']=pd.to_numeric(df.rating,errors='coerce').fillna(0).clip(0,5).astype('float32')
        df['app_category']=df.app_category.fillna('').astype(str).str.strip()
        uc,ic,lc='uid','app_package','app_category'
    users=pd.unique(df[uc]); items=pd.unique(df[ic])
    umap={v:i for i,v in enumerate(users)}; imap={v:i for i,v in enumerate(items)}
    rows=np.fromiter((umap[x] for x in df[uc].to_numpy()),dtype=np.int32,count=len(df))
    cols=np.fromiter((imap[x] for x in df[ic].to_numpy()),dtype=np.int32,count=len(df))
    seen=csr_matrix((np.ones(len(df),dtype=np.uint8),(rows,cols)),shape=(len(users),len(items)))
    seen.sum_duplicates(); seen.data[:]=1; seen.eliminate_zeros()
    metadata=df[[ic,lc]].drop_duplicates(ic).set_index(ic)[lc]
    item_labels=np.asarray([str(metadata.get(x,'')) for x in items],dtype=object)
    if domain=='Books':
        label_sets=[set(s.split(', ')) if s!='<NA>' else set() for s in item_labels]
    else:
        label_sets=[{s} if s else set() for s in item_labels]
    del rows,cols,umap,imap,metadata
    return df,uc,ic,users,items,seen,label_sets

def doses_for(domain,category):
    if domain=='Books':
        # Adventure has a saved 3500-user run; Thriller and Drama end at 3000.
        return BOOK_DOSES + ([3500] if category=='Adventure' else [])
    if domain=='Movies': return MOVIE_DOSES[category]
    return APP_DOSES

def factor_path(domain,category,model,n):
    if domain=='Books':
        if n==0:
            return R/f'Book/current/PerturbRec/svd_nmf/layer2/embeddings/pilot/{model}_baseline_a.npz'
        folder='adventure' if category=='Adventure' else category.lower()
        if category=='Adventure':
            return R/f'Book/current/PerturbRec/svd_nmf/layer2/embeddings/controlled/{model}/{model}_adventure_{n}.npz'
        return R/f'Book/current/PerturbRec/svd_nmf/layer2/embeddings/controlled/{model}/{folder}/{model}_{folder}_{n}.npz'
    if domain=='Movies':
        return NB/f'movies_{category[:-1]}_{model}_{n}_factors.npz'
    return NB/f'{category.lower()}_{model}_{n}_factors.npz'

REFS={}
def reference(domain,category):
    if domain=='Books':
        key='books_'+category
        p=NB/'three_genre_panel_independent_components.csv'
    elif domain=='Movies':
        key='movie_'+category
        p=NB/f'movies_{category[:-1]}_reviewer_components.csv'
    else:
        key='app_'+category
        p=NB/f'{category.lower()}_reviewer_components.csv'
    if key not in REFS:
        d=pd.read_csv(p)
        if domain=='Books':
            d=d[d.genre.astype(str).str.casefold()==category.casefold()]
        REFS[key]=d
    return REFS[key]

def score_distribution(domain,category,model,n,context):
    df,uc,ic,raw_users,raw_items,seen,label_sets=context
    path=factor_path(domain,category,model,n)
    if not path.exists(): raise FileNotFoundError(path)
    with np.load(path) as z:
        pu=z['pu']; qi=z['qi']; bu=z['bu']; bi=z['bi']; mu=float(z['mu'])
        if domain=='Books':
            factor_users=z['ids']; factor_items=z['items']
            if not np.array_equal(factor_users[:len(raw_users)],raw_users):
                raise ValueError(f'Book user order mismatch: {path}')
            if not np.array_equal(factor_items,raw_items):
                raise ValueError(f'Book item order mismatch: {path}')
            # Original users/items occupy the same first-seen order in every checkpoint.
            real_idx=np.arange(len(raw_users),dtype=np.int64)
        else:
            if pu.shape[0]<len(raw_users) or qi.shape[0]!=len(raw_items):
                raise ValueError(f'Factor shape mismatch {path}: {pu.shape}, {qi.shape}, data={len(raw_users),len(raw_items)}')
            real_idx=np.arange(len(raw_users),dtype=np.int64)
        if not (np.isfinite(pu).all() and np.isfinite(qi).all() and np.isfinite(bu).all() and np.isfinite(bi).all() and np.isfinite(mu)):
            raise ValueError(f'Non-finite factor in {path}')
        target=np.asarray([category in labels for labels in label_sets],dtype=bool)
        for group,mask in [('target',target),('non_target',~target)]:
            idx=np.flatnonzero(mask)
            obs=seen[:,idx]
            counts=np.asarray(obs.sum(axis=1)).ravel().astype(np.int64)
            avail=len(idx)-counts
            qsum=qi[idx].sum(axis=0)-obs@qi[idx]
            bisum=float(bi[idx].sum())-np.asarray(obs@bi[idx]).ravel()
            valid=avail>0
            scores=mu+bu[real_idx]+bisum/np.maximum(avail,1)+np.einsum('ij,ij->i',pu[real_idx],qsum)/np.maximum(avail,1)
            scores=scores[valid].astype(np.float64)
            if not np.isfinite(scores).all():
                raise ValueError(f'Non-finite user predictions: {domain}/{category}/{model}/n={n}/{group}')
            key=(domain,category,n,model,group)
            DISTS[key]=scores
            ref=reference(domain,category)
            match=ref[(ref.model==model)&(ref.n_fake==n)&(ref.view=='real_unseen_prediction')&(ref.group==group)]
            computed=float(scores.mean())
            expected=np.nan
            global N_REFERENCE_VALIDATED,N_REFERENCE_MISSING
            if len(match)==1:
                expected=float(match.iloc[0].mean_prediction)
                if not np.isclose(computed,expected,rtol=1e-5,atol=1e-5):
                    raise ValueError(f'Mean mismatch {key}: computed {computed}, source {expected}')
                N_REFERENCE_VALIDATED+=1
            elif len(match)==0:
                N_REFERENCE_MISSING+=1
            else:
                raise ValueError(f'Multiple reference rows for {key}: {len(match)}')
            STATS.append({
                'domain':domain,'category':category,'n_fake':n,'model':model,'group':group,
                'n_users':len(scores),'mean':computed,'median':float(np.median(scores)),
                'q05':float(np.quantile(scores,.05)),'q95':float(np.quantile(scores,.95)),
                'min':float(scores.min()),'max':float(scores.max()),
                'n_below_display':int((scores<YMIN).sum()),'n_above_display':int((scores>YMAX).sum()),
                'reference_mean_match':expected,'factor_file':str(path)
            })

for domain,cats in CATEGORIES.items():
    print('LOAD',domain,flush=True)
    ctx=read_domain(domain)
    for category in cats:
        doses=doses_for(domain,category)
        print('CATEGORY',domain,category,'doses',doses,flush=True)
        for n in doses:
            for model in ['svd','nmf']:
                print('FIT-SCORES',domain,category,n,model,flush=True)
                score_distribution(domain,category,model,n,ctx)
    del ctx
    gc.collect()

pd.DataFrame(STATS).to_csv(STATS_PATH,index=False)

# 3 x 3 panels. Every dose position contains four distributions:
# SVD/NMF, each for target and non-target unrated items.
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7.5,'axes.titlesize':9.4,'axes.labelsize':7.2})
fig,axs=plt.subplots(3,3,figsize=(19.2,13.0),sharey='row')
order=[('svd','target'),('svd','non_target'),('nmf','target'),('nmf','non_target')]
offsets=[-.30,-.10,.10,.30]
for ri,(domain,cats) in enumerate(CATEGORIES.items()):
    for ci,category in enumerate(cats):
        ax=axs[ri,ci]
        doses=doses_for(domain,category)
        positions=[]; samples=[]; colors=[]; plotted_keys=[]
        for si,n in enumerate(doses):
            for offset,key in zip(offsets,order):
                dist=DISTS[(domain,category,n,key[0],key[1])]
                inside=dist[(dist>=YMIN)&(dist<=YMAX)]
                if len(inside)>2500:
                    rng=np.random.default_rng(20261119+ri*1000+ci*100+si*10+order.index(key))
                    sample=inside[rng.choice(len(inside),2500,replace=False)]
                else:
                    sample=inside
                xpos=si+offset
                if len(sample)>=2 and np.std(sample)>1e-9:
                    positions.append(xpos);samples.append(sample);colors.append(COLORS[key]);plotted_keys.append((xpos,dist,key,n))
                elif len(sample):
                    # Show near-constant series as a narrow synthetic cloud for KDE only.
                    eps=max(float(np.std(sample)),0.01)
                    sample=np.r_[sample, sample[0]+eps*np.array([-1.,1.])]
                    positions.append(xpos);samples.append(sample);colors.append(COLORS[key]);plotted_keys.append((xpos,dist,key,n))
        if samples:
            vp=ax.violinplot(samples,positions=positions,widths=.17,showmeans=False,showmedians=True,showextrema=False,bw_method='scott')
            for body,color in zip(vp['bodies'],colors):
                body.set_facecolor(color);body.set_edgecolor('#35424C');body.set_linewidth(.35);body.set_alpha(.74)
            vp['cmedians'].set_color('#25313A');vp['cmedians'].set_linewidth(.6)
            for xpos,dist,key,n in plotted_keys:
                col=COLORS[key]
                if np.any(dist>YMAX):
                    ax.scatter([xpos],[YMAX-.10],marker='^',s=7,color=col,edgecolors='none',zorder=4,clip_on=False)
                if np.any(dist<YMIN):
                    ax.scatter([xpos],[YMIN+.10],marker='v',s=7,color=col,edgecolors='none',zorder=4,clip_on=False)
        ax.set_ylim(YMIN,YMAX)
        ax.set_xlim(-.55,len(doses)-.45)
        ax.set_xticks(np.arange(len(doses)),[str(x) for x in doses],rotation=45,ha='right',fontsize=6.2)
        ax.grid(axis='y',color='#E1E7EB',lw=.5);ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
        ax.set_title(f'{domain} · {category}',loc='left',fontweight='bold',pad=3)
        ax.set_xlabel('Synthetic users added',fontsize=6.8,labelpad=1)
        if ci==0: ax.set_ylabel('Per-user mean prediction')
        else: ax.tick_params(labelleft=True)
        ax.tick_params(axis='y',labelsize=6.4,pad=1)
handles=[Patch(facecolor=COLORS[k],edgecolor='#35424C',alpha=.74,label=LABELS[k]) for k in order]
fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.997),ncol=4,frameon=False,fontsize=8.2)
fig.suptitle('Per-User Prediction Distributions Across Injection Doses',fontsize=14,fontweight='bold',y=1.012)
fig.text(.5,.006,'Each violin is the distribution across original users of each user’s mean prediction for unrated target / other items. Median line shown. Density uses in-range scores only (−2 to 7); triangles mark clipped tails. Full range and clipped counts are in the companion CSV.',ha='center',fontsize=7.1,color='#4D5B66')
fig.subplots_adjust(left=.055,right=.995,top=.96,bottom=.055,wspace=.16,hspace=.31)
fig.savefig(TARGET,dpi=450,bbox_inches='tight')
plt.close(fig)
meta={
 'figure':str(TARGET),'stats_csv':str(STATS_PATH),
 'categories':CATEGORIES,
 'dose_steps':{f'{d}/{c}':doses_for(d,c) for d,cats in CATEGORIES.items() for c in cats},
 'series_per_dose':['SVD target','SVD non-target','NMF target','NMF non-target'],
 'view':'real_unseen_prediction','per_user_score':'mean predicted score over items in the category group that the user has not rated',
 'mean_validated_against_existing_component_tables':'where aggregate reference rows exist','reference_rows_validated':N_REFERENCE_VALIDATED,'reference_rows_unavailable':N_REFERENCE_MISSING,
 'density_sample_max_per_violin':2500,'density_range':[YMIN,YMAX],
 'outside_values_are_clipped_in_png_but_reported_in_csv':True,
 'books_baseline_checkpoint':'pilot/{model}_baseline_a.npz'
}
META_PATH.write_text(json.dumps(meta,indent=2))
print('SAVED',TARGET,flush=True)
print('VALIDATED GROUPS',len(STATS),flush=True)
for x in STATS:
    print(x['domain'],x['category'],x['n_fake'],x['model'],x['group'],
          'mean',round(x['mean'],4),'median',round(x['median'],4),
          'range',round(x['min'],3),round(x['max'],3),
          'out',x['n_below_display'],x['n_above_display'],flush=True)
