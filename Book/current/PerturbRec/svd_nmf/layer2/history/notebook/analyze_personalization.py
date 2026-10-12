import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
from pathlib import Path
import sys,json,time,argparse
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
ROOT=Path('/home/moshtasa/Research/phd-recsys/Book/current/PerturbRec/svd_nmf/layer2')
sys.path.insert(0,str(ROOT/'notebook'))
from analyze_recommendation_effect import load_data,l
from analyze_cross_genre import top20
H=ROOT/'history';COUNTS=[2,4,6,25,50,100,200,500];SEED=20261012
for sub in ['notebook','tables','data','result','log','reports']:(H/sub).mkdir(parents=True,exist_ok=True)
def recommendations(model,genre,n,base,seen):
 file=H/'data'/model/('baseline_top20.npz' if n==0 else f'{genre.lower()}_{n}_top20.npz');file.parent.mkdir(parents=True,exist_ok=True)
 if file.exists():
  z=np.load(file);assert np.array_equal(z['ids'],base['ids']);assert np.array_equal(z['items'],base['items']);return z['top']
 if n==0:pu,qi,bi=base['pu'],base['qi'],base['bi']
 else:
  z=np.load(ROOT/'embeddings/controlled'/model/genre.lower()/f'{model}_{genre.lower()}_{n}.npz')
  ui=np.argsort(z['ids']);ui=ui[l.real_rows(z['ids'][ui],base['ids'],expected_fake=n)];ii=np.argsort(z['items'])
  assert np.array_equal(z['items'][ii],base['items']);pu,qi,bi=z['pu'][ui],z['qi'][ii],z['bi'][ii]
  assert np.array_equal(z['ids'][ui],base['ids'])
  if model=='nmf':assert pu.min()>=-1e-12 and qi.min()>=-1e-12,'NMF nonnegative constraint failed'
 rows=[]
 for start in range(0,len(pu),256):
  sl=slice(start,min(start+256,len(pu)));mask=seen[sl].toarray();score=pu[sl]@qi.T+bi
  assert np.isfinite(score).all();ix=top20(score,mask);assert not np.take_along_axis(mask,ix,axis=1).any();rows.append(ix.astype(np.int32))
 top=np.concatenate(rows);assert all(len(np.unique(row))==20 for row in top)
 np.savez_compressed(file,ids=base['ids'],items=base['items'],top=top)
 print('TOP20',model,genre,n,flush=True);return top

def pairs(n,count=100000):
 rng=np.random.default_rng(SEED);codes=np.array([],dtype=np.int64)
 while len(codes)<count:
  a=rng.integers(n,size=count*2);b=rng.integers(n,size=count*2);lo=np.minimum(a,b);hi=np.maximum(a,b);codes=np.unique(np.r_[codes,lo[lo!=hi]*n+hi[lo!=hi]])
 codes=rng.choice(codes,count,replace=False);return codes//n,codes%n

def intersection(top,a,b):
 out=np.zeros(len(a),dtype=np.int16)
 for s in range(0,len(a),5000):
  u=slice(s,min(s+5000,len(a)));out[u]=(top[a[u],:,None]==top[b[u],None,:]).sum(axis=(1,2))
 return out

def exposures(top,nitems,target):
 freq=np.bincount(top.ravel(),minlength=nitems);total=int(freq.sum());prob=freq/total
 tfreq=freq[target];tp=tfreq/tfreq.sum();nonzero=prob[prob>0];tnz=tp[tp>0]
 N=len(top)
 return dict(n_users=N,catalog_size=nitems,unique_books=int((freq>0).sum()),catalog_coverage_percent=float((freq>0).mean()*100),target_unique_books=int((tfreq>0).sum()),target_catalog_size=int(target.sum()),target_coverage_percent=float((tfreq>0).mean()*100),effective_catalog=float(1/(prob@prob)),target_effective_catalog=float(1/(tp@tp)),target_entropy_effective_catalog=float(np.exp(-(tnz*np.log(tnz)).sum())),top10_exposure_percent=float(np.sort(freq)[-10:].sum()/total*100),target_top10_exposure_percent=float(np.sort(tfreq)[-10:].sum()/tfreq.sum()*100),top1_user_reach_percent=float(freq.max()/N*100),target_top1_user_reach_percent=float(tfreq.max()/N*100),mean_target_count=float(tfreq.sum()/N),exact_mean_pair_shared_books=float((freq.astype(float)*(freq-1)).sum()/(N*(N-1))))

def fixed_target(top,target,m=2):
 mask=target[top];out=np.full((len(top),m),-1,dtype=np.int32);eligible=mask.sum(axis=1)>=m
 validrows=np.flatnonzero(eligible)
 for u in validrows:out[u]=top[u,mask[u]][:m]
 return out,eligible

def worker(model):
 base,seen,genres=load_data(model);ids=base['ids'];items=base['items'];allgenres=sorted({g for gs in genres.values() for g in gs})
 membership=np.array([[g in genres[int(i)] for g in allgenres] for i in items],dtype=float)
 fractional=membership/membership.sum(axis=1)[:,None]
 hist=np.asarray(seen@fractional);norm=np.linalg.norm(hist,axis=1);assert (norm>0).all();hist/=norm[:,None]
 a,b=pairs(len(ids));cos=(hist[a]*hist[b]).sum(axis=1);q1,q3=np.quantile(cos,[.25,.75]);strata={'all_pairs':np.ones(len(a),bool),'different_histories':cos<=q1,'similar_histories':cos>=q3}
 pd.DataFrame({'user_a':ids[a],'user_b':ids[b],'history_cosine':cos}).to_csv(H/'tables'/f'{model}_fixed_user_pairs.csv.gz',index=False)
 before=recommendations(model,'baseline',0,base,seen);before_inter=intersection(before,a,b);summary=[];pairrows=[];controls=[]
 for genre in ['Drama','Thriller']:
  target=membership[:,allgenres.index(genre)].astype(bool);tops={0:before}
  for n in COUNTS:tops[n]=recommendations(model,genre,n,base,seen)
  # Count matching: the same users have >=2 target recommendations in EVERY evaluated condition.
  fixed={};common=np.ones(len(ids),bool)
  for n,top in tops.items():fixed[n],eligible=fixed_target(top,target);common &= eligible
  joint=common[a]&common[b]
  for n,top in tops.items():
   rec=dict(model=model,genre=genre,n_fake=n,**exposures(top,len(items),target));summary.append(rec)
   if n:
    cache=np.load(ROOT/'tables/cross_genre'/model/genre/f'{n}_users.npz');assert np.array_equal(cache['ids'],ids)
    gi=list(cache['genres']).index(genre)
    assert np.array_equal(target[top].sum(axis=1),cache['inclusive_after'][:,gi])
    assert np.array_equal(target[before].sum(axis=1),cache['inclusive_before'][:,gi])
   shared=intersection(top,a,b);jaccard=shared/(40-shared)
   for label,keep in strata.items():
    pairrows.append(dict(model=model,genre=genre,n_fake=n,stratum=label,n_pairs=int(keep.sum()),history_cosine_mean=float(cos[keep].mean()),mean_shared_books=float(shared[keep].mean()),mean_jaccard_percent=float(jaccard[keep].mean()*100),mean_shared_change_from_baseline=float((shared[keep]-before_inter[keep]).mean()),fraction_pairs_overlap_increased_percent=float((shared[keep]>before_inter[keep]).mean()*100)))
   for label,keep in strata.items():
    mask=keep&joint;assert mask.sum()>0
    aa,bb=a[mask],b[mask];ss=intersection(fixed[n],aa,bb)
    # Same pairs, same users, exactly TWO target books per user at every dose.
    controls.append(dict(model=model,genre=genre,n_fake=n,stratum=label,n_eligible_users=int(common.sum()),eligible_users_percent=float(common.mean()*100),n_pairs=int(mask.sum()),fixed_target_books_per_user=2,mean_shared_target_books=float(ss.mean()),mean_target_jaccard_percent=float((ss/(4-ss)).mean()*100)))
   print('METRICS',model,genre,n,'shared',round(rec['exact_mean_pair_shared_books'],3),'coverage',rec['unique_books'],'target_eff',round(rec['target_effective_catalog'],1),flush=True)
 pd.DataFrame(summary).to_csv(H/'tables'/f'{model}_catalog_metrics.csv',index=False)
 pd.DataFrame(pairrows).to_csv(H/'tables'/f'{model}_pair_metrics.csv',index=False)
 pd.DataFrame(controls).to_csv(H/'tables'/f'{model}_fixed_target_control.csv',index=False)
 return model

def plot():
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':200})
 catalog=pd.concat([pd.read_csv(H/'tables'/f'{m}_catalog_metrics.csv') for m in ['nmf','svd']]);pair=pd.concat([pd.read_csv(H/'tables'/f'{m}_pair_metrics.csv') for m in ['nmf','svd']]);control=pd.concat([pd.read_csv(H/'tables'/f'{m}_fixed_target_control.csv') for m in ['nmf','svd']])
 for name,df in [('catalog_metrics',catalog),('pair_metrics',pair),('fixed_target_control',control)]:df.to_csv(H/'tables'/f'{name}.csv',index=False)
 matched=H/'tables/cross_model_fixed_target_control.csv'
 if matched.exists():control=pd.read_csv(matched)
 for genre in ['Drama','Thriller']:
  fig,axs=plt.subplots(2,3,figsize=(20,11.25));x=np.arange(9);labels=['Base']+list(map(str,COUNTS))
  for model,color in [('nmf','#218B58'),('svd','#2563A6')]:
   c=catalog[(catalog.model==model)&(catalog.genre==genre)].sort_values('n_fake');p=pair[(pair.model==model)&(pair.genre==genre)&(pair.stratum=='different_histories')].sort_values('n_fake');q=control[(control.model==model)&(control.genre==genre)&(control.stratum=='different_histories')].sort_values('n_fake')
   specs=[(c,'exact_mean_pair_shared_books','Shared book titles across all user pairs','Mean shared titles out of 20'),(p,'mean_jaccard_percent','Overlap between different-history users','Mean Jaccard (%)'),(c,'unique_books','Catalog coverage','Distinct recommended titles'),(c,'target_effective_catalog','Within-target exposure concentration','Effective number of target titles'),(q,'mean_target_jaccard_percent','Count-controlled target overlap','Jaccard of top TWO target books (%)'),(c,'target_top10_exposure_percent','Target exposure captured by 10 titles','Share of target recommendations (%)')]
   for ax,(df,key,title,ylabel) in zip(axs.flat,specs):ax.plot(x,df[key],'-o',color=color,label=model.upper(),lw=2.5,ms=5);ax.set_title(title,fontsize=14);ax.set_ylabel(ylabel);ax.set_xticks(x,labels,rotation=40,ha='right');ax.grid(alpha=.18);ax.legend(frameon=False)
  fig.suptitle(f'{genre} injection: are recommendations becoming more alike?',fontsize=23)
  fig.text(.5,.025,'Same real users and unseen-book catalog; Top-20 raw ranking scores. Different histories = lowest baseline history-cosine quartile.\nTarget control: same eligible users and pairs across both models and every dose, exactly 2 target titles per user. One controlled training seed; sampled pair metrics are descriptive.',ha='center',fontsize=11)
  fig.subplots_adjust(top=.88,bottom=.15,left=.07,right=.98,hspace=.45,wspace=.3)
  folder=H/'result'/genre;folder.mkdir(parents=True,exist_ok=True);fig.savefig(folder/'personalization_6panel.png',facecolor='white');plt.close(fig)
 lines=['# Personalization and concentration after injection','', 'This analysis is about recommended titles, not observed changes in human taste. No retraining was performed.','', '## Results at 500 synthetic users','', '|Model|Target|Shared titles before → after (out of 20)|Distinct recommended titles before → after|Effective target catalog before → after|Different-history fixed-two target Jaccard before → after|','|---|---|---|---|---|---|']
 for m in ['nmf','svd']:
  for g in ['Drama','Thriller']:
   z=catalog[(catalog.model==m)&(catalog.genre==g)].set_index('n_fake');q=control[(control.model==m)&(control.genre==g)&(control.stratum=='different_histories')].set_index('n_fake')
   lines.append(f"|{m.upper()}|{g}|{z.loc[0,'exact_mean_pair_shared_books']:.3f} → {z.loc[500,'exact_mean_pair_shared_books']:.3f}|{z.loc[0,'unique_books']:.0f} → {z.loc[500,'unique_books']:.0f}|{z.loc[0,'target_effective_catalog']:.1f} → {z.loc[500,'target_effective_catalog']:.1f}|{q.loc[0,'mean_target_jaccard_percent']:.2f}% → {q.loc[500,'mean_target_jaccard_percent']:.2f}%|")
 lines+=['','## Method and limits','', '- Catalog/exposure metrics cover every real user. Mean pair intersection is exact for all distinct user pairs, using item exposure frequencies. Jaccard metrics use the same 100,000 randomly selected distinct user pairs at every condition (sampling seed 20261012).','- History similarity uses cosine of original per-user genre frequencies, with each rated book allocating one unit fractionally across its tags. Lowest/highest quartiles define different/similar history groups; this is a history proxy, not measured human preferences.','- Raw ranking score is pu @ qi.T + bi. Global mean and user bias are omitted because they are constant across candidates for a user. Previously rated books are excluded; ties favor smaller book IDs. Recommendation target counts are validated exactly against the earlier cross-genre arrays.','- Effective target catalog = 1 / sum(p_i²), where p_i is a title’s fraction of all target recommendations. This normalizes exposure volume but is still affected by which users receive target recommendations.','- The fixed-two control uses the top-ranked TWO target books within each Top-20 list, only users who have at least two target books in every condition. It fixes users, pair identities, and target list length, but selects a subset and cannot establish an effect for excluded users. It does not test a separately trained genre-share-matched intervention.','- Increased title overlap is evidence of output homogenization; it is not by itself proof of poorer relevance, utility, or human satisfaction. No held-out accuracy or behavioral outcomes are measured.','- This is one fixed-initialization training seed and one dense injection protocol (target 5, other books 1). Pair sample sizes do not constitute independent training repetitions; no training-level significance is claimed. Analysis is restricted to <=500.','- There is no expectation that every model or genre must move in the same direction. Report declines and null/mixed findings as well as increases.']
 (H/'reports/findings.md').write_text('\n'.join(lines))

def main():
 manifest={'status':'running','counts':[0]+COUNTS,'models':['nmf','svd'],'genres':['Drama','Thriller'],'k':20,'pair_sample':100000,'pair_seed':SEED,'trained_models_reused':True,'training_seed':42,'created':time.strftime('%Y-%m-%dT%H:%M:%S')}
 (H/'log/manifest.json').write_text(json.dumps(manifest,indent=2))
 with ProcessPoolExecutor(max_workers=2) as pool:list(pool.map(worker,['nmf','svd']))
 plot();manifest['status']='complete';manifest['validation']='finite scores; user/item IDs matched; unseen books; no duplicate recommendations; NMF nonnegative <=500; every saved target-count array matched previous cross-genre output exactly';(H/'log/manifest.json').write_text(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
