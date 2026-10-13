import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import ast,json,hashlib,time
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from concurrent.futures import ProcessPoolExecutor
R=Path('/home/moshtasa/Research/phd-recsys/Book/current/PerturbRec/svd_nmf/layer2');H=R/'history';OUT=H/'tables/title_mechanism';OUT.mkdir(parents=True,exist_ok=True)
DATA=H/'data/title_mechanism';DATA.mkdir(parents=True,exist_ok=True)
SRC=Path('/home/moshtasa/Research/phd-recsys/Book/Archive/data/genre_clean.csv');CONDITIONS=['baseline','embedding_only','item_bias_only','full_after']

def genres(v):
 try:
  x=ast.literal_eval(str(v));return [str(g).strip() for g in x if str(g).strip().lower()!='unknown']
 except (ValueError,SyntaxError):return [g.strip() for g in str(v).replace('|',',').split(',') if g.strip().lower()!='unknown']

def load(path,ids=None,items=None):
 z=np.load(path);u=np.argsort(z['ids']);i=np.argsort(z['items'])
 if ids is not None:
  pos=np.searchsorted(z['ids'][u],ids);assert (pos<len(u)).all();u=u[pos];assert np.array_equal(z['ids'][u],ids)
 if items is not None:assert np.array_equal(z['items'][i],items)
 return dict(ids=z['ids'][u],items=z['items'][i],pu=z['pu'][u],qi=z['qi'][i],bi=z['bi'][i])

def rank(score,mask,k=20):
 s=score.copy();s[mask]=-np.inf;ix=np.argpartition(-s,k-1,axis=1)[:,:k];cut=np.take_along_axis(s,ix,axis=1).min(axis=1);out=np.empty_like(ix)
 for row in range(len(s)):
  c=np.flatnonzero(s[row]>=cut[row]);out[row]=c[np.lexsort((c,-s[row,c]))[:k]]
 assert np.isfinite(np.take_along_axis(s,out,axis=1)).all();return out.astype(np.int32)

def shared(top,a,b):
 v=[]
 for start in range(0,len(a),5000):
  sl=slice(start,min(start+5000,len(a)));v.append((top[a[sl],:,None]==top[b[sl],None,:]).sum(axis=(1,2)))
 return np.concatenate(v)

def metrics(top,target):
 freq=np.bincount(top.ravel(),minlength=len(target));N=len(top);f=freq[target];p=f/f.sum()
 return dict(exact_mean_shared_titles=float((freq.astype(float)*(freq-1)).sum()/(N*(N-1))),mean_target_count=float(f.sum()/N),target_effective_catalog=float(1/(p@p)),target_top10_exposure_percent=float(np.sort(f)[-10:].sum()/f.sum()*100),catalog_coverage=int((freq>0).sum()))

def heterogeneity(p,q,bi,target):
 pp=p-p.mean(axis=0);qq=q[target]-q[target].mean(axis=0);covp=pp.T@pp/len(pp);covq=qq.T@qq/len(qq)
 residual=float(np.sqrt(max(0,np.trace(covp@covq))))
 common=float(np.std(p.mean(axis=0)@q[target].T+bi[target]))
 return dict(target_personalized_score_rms=residual,target_common_score_std=common,common_to_personalized_ratio=common/residual)

def job(model):
 basepath=R/'embeddings/pilot'/f'{model}_baseline_a.npz';afterpath=R/'embeddings/controlled'/model/'thriller'/f'{model}_thriller_500.npz'
 b=load(basepath);ids=b['ids'];items=b['items'];f=load(afterpath,ids,items)
 for z in [b,f]:assert all(np.isfinite(z[k]).all() for k in ['pu','qi','bi'])
 if model=='nmf':assert min(b['pu'].min(),b['qi'].min(),f['pu'].min(),f['qi'].min())>=-1e-12
 df=pd.read_csv(SRC,usecols=['user_id','book_id','original_title','genres']);meta=df.drop_duplicates('book_id').set_index('book_id').loc[items]
 ur=np.searchsorted(ids,df.user_id);ir=np.searchsorted(items,df.book_id);assert np.array_equal(ids[ur],df.user_id) and np.array_equal(items[ir],df.book_id)
 seen=csr_matrix((np.ones(len(df),bool),(ur,ir)),shape=(len(ids),len(items)));target=np.array(['Thriller' in genres(g) for g in meta.genres])
 n_rating_rows=len(df)
 del df
 cache0=np.load(H/'data'/model/'baseline_top20.npz');cache1=np.load(H/'data'/model/'thriller_500_top20.npz');assert np.array_equal(cache0['ids'],ids) and np.array_equal(cache1['ids'],ids)
 pairs=pd.read_csv(H/'tables/cross_model_thriller_fixed_pairs.csv.gz');a=np.searchsorted(ids,pairs.user_a);v=np.searchsorted(ids,pairs.user_b)
 threshold=pd.read_csv(H/'tables/nmf_fixed_user_pairs.csv.gz').history_cosine.quantile(.25);keep=pairs.history_cosine.to_numpy()<=threshold;a=a[keep];v=v[keep]
 # The original selected user pairs are fixed, even if hybrid Top-20 genre counts differ.
 tops={c:[] for c in CONDITIONS};target2={c:[] for c in CONDITIONS};den=np.zeros(len(items));g0sum=np.zeros(len(items));g1sum=np.zeros(len(items));rel0=np.zeros(len(items));rel1=np.zeros(len(items));relg=np.zeros(len(items));relbi=np.zeros(len(items));g0sq=np.zeros(len(items));g1sq=np.zeros(len(items))
 for start in range(0,len(ids),256):
  sl=slice(start,min(start+256,len(ids)));mask=seen[sl].toarray();valid=~mask;e0=b['pu'][sl]@b['qi'].T;e1=f['pu'][sl]@f['qi'].T
  scores=[e0+b['bi'],e1+b['bi'],e0+f['bi'],e1+f['bi']]
  for condition,s in zip(CONDITIONS,scores):
   rec=rank(s,mask);tops[condition].append(rec);target2[condition].append(rank(s,mask|~target,2))
   if condition=='baseline':assert np.array_equal(rec,cache0['top'][sl])
   if condition=='full_after':assert np.array_equal(rec,cache1['top'][sl])
  count=valid.sum(axis=1);means0=(scores[0]*valid).sum(axis=1)/count;means1=(scores[3]*valid).sum(axis=1)/count;eg=( (e1-e0)*valid).sum(axis=1)/count;bg=(valid@(f['bi']-b['bi']))/count
  den+=valid.sum(axis=0);g0sum+=(e0*valid).sum(axis=0);g1sum+=(e1*valid).sum(axis=0);g0sq+=(e0*e0*valid).sum(axis=0);g1sq+=(e1*e1*valid).sum(axis=0)
  rel0+=((scores[0]-means0[:,None])*valid).sum(axis=0);rel1+=((scores[3]-means1[:,None])*valid).sum(axis=0);relg+=((e1-e0-eg[:,None])*valid).sum(axis=0);relbi+=((f['bi']-b['bi']-bg[:,None])*valid).sum(axis=0)
 records=[];freq={};hier={}
 for condition in CONDITIONS:
  top=np.concatenate(tops[condition]);two=np.concatenate(target2[condition]);freq[condition]=np.bincount(top.ravel(),minlength=len(items));ss=shared(two,a,v);m=metrics(top,target)
  records.append(dict(model=model,condition=condition,n_fake=500 if condition!='baseline' else 0,fixed_target_pair_count=len(a),fixed_target_jaccard_percent=float((ss/(4-ss)).mean()*100),**m))
  if condition in ['baseline','full_after']:
   # Verify top-two target recommendations agree with the original within-Top-20 control on selected users.
   users=np.unique(np.r_[a,v]);actual=np.array([row[target[row]][:2] for row in top[users]]);assert np.array_equal(actual,two[users])
  np.savez_compressed(DATA/f'{model}_{condition}_recommendations.npz',ids=ids,items=items,top20=top,top2_unseen_thriller=two)
 pd.DataFrame(records).to_csv(OUT/f'{model}_conditions.csv',index=False)
 records=pd.DataFrame(records).set_index('condition');attrib=[]
 for key in ['exact_mean_shared_titles','fixed_target_jaccard_percent','mean_target_count']:
  y=records[key];embedding=.5*((y.embedding_only-y.baseline)+(y.full_after-y.item_bias_only));bias=.5*((y.item_bias_only-y.baseline)+(y.full_after-y.embedding_only));assert np.isclose(embedding+bias,y.full_after-y.baseline)
  attrib.append(dict(model=model,metric=key,embedding_allocation=float(embedding),item_bias_allocation=float(bias),total_change=float(y.full_after-y.baseline)))
 pd.DataFrame(attrib).to_csv(OUT/f'{model}_attribution.csv',index=False)
 # Catalog-wide item diagnostics: centering removes common score shifts irrelevant to within-user ranking.
 mean0=g0sum/den;mean1=g1sum/den;gdelta=mean1-mean0;bidelta=f['bi']-b['bi'];assert np.allclose(relg/den+relbi/den,(rel1-rel0)/den,atol=1e-9)
 it=pd.DataFrame(dict(model=model,book_id=items,title=meta.original_title.to_numpy(),genres=meta.genres.to_numpy(),target=target,eligible_real_users=den.astype(int),baseline_reach_percent=freq['baseline']/len(ids)*100,after_reach_percent=freq['full_after']/len(ids)*100,embedding_only_reach_percent=freq['embedding_only']/len(ids)*100,item_bias_only_reach_percent=freq['item_bias_only']/len(ids)*100,bias_before=b['bi'],bias_after=f['bi'],bias_delta=bidelta,mean_dot_before=mean0,mean_dot_after=mean1,mean_dot_delta=gdelta,dot_std_before=np.sqrt(np.maximum(0,g0sq/den-mean0**2)),dot_std_after=np.sqrt(np.maximum(0,g1sq/den-mean1**2)),relative_score_before=rel0/den,relative_score_after=rel1/den,relative_score_delta=(rel1-rel0)/den,relative_embedding_delta=relg/den,relative_bias_delta=relbi/den))
 it['n_original_raters']=len(ids)-it.eligible_real_users
 it['fake_fraction_of_raters_percent']=500/(it.n_original_raters+500)*100
 it['reach_delta_pp']=it.after_reach_percent-it.baseline_reach_percent;it.to_csv(OUT/f'{model}_book_scores.csv',index=False)
 topbooks=it[it.target].nlargest(10,'after_reach_percent');topbooks.to_csv(OUT/f'{model}_leading_thriller_titles.csv',index=False)
 h=[]
 for condition,p,q,bi in [('baseline',b['pu'],b['qi'],b['bi']),('full_after',f['pu'],f['qi'],f['bi'])]:h.append(dict(model=model,condition=condition,**heterogeneity(p,q,bi,target)))
 pd.DataFrame(h).to_csv(OUT/f'{model}_score_heterogeneity.csv',index=False)
 proof={'model':model,'n_real_users':len(ids),'n_books':len(items),'n_original_rating_rows':n_rating_rows,'n_injected_rating_rows_protocol':500*len(items),'injected_to_original_rating_rows_percent':500*len(items)/n_rating_rows*100,'original_top20_exactly_matched':True,'original_count_control_exact_match':True,'relative_score_decomposition_max_error':float(np.max(np.abs(it.relative_embedding_delta+it.relative_bias_delta-it.relative_score_delta))),'artifacts':{str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in [basepath,afterpath]}}
 (OUT/f'{model}_validation.json').write_text(json.dumps(proof,indent=2))
 print(model,records[['exact_mean_shared_titles','fixed_target_jaccard_percent']].to_string(),flush=True)
 return model

def table(df):
 columns=list(df.columns)
 rows=['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']
 for row in df.itertuples(index=False,name=None):rows.append('| '+' | '.join(str(v).replace('|','/') for v in row)+' |')
 return '\n'.join(rows)

def finish():
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'savefig.dpi':200,'axes.spines.top':False,'axes.spines.right':False})
 conditions=pd.concat([pd.read_csv(OUT/f'{m}_conditions.csv') for m in ['nmf','svd']]);alloc=pd.concat([pd.read_csv(OUT/f'{m}_attribution.csv') for m in ['nmf','svd']]);hetero=pd.concat([pd.read_csv(OUT/f'{m}_score_heterogeneity.csv') for m in ['nmf','svd']])
 for name,df in [('conditions',conditions),('attribution',alloc),('score_heterogeneity',hetero)]:df.to_csv(OUT/f'{name}.csv',index=False)
 labels=['Baseline','Updated\nembedding only','Updated\nitem bias only','Full after'];colors=['#8C98A6','#218B58','#E77817','#2563A6']
 for model in ['nmf','svd']:
  z=conditions[conditions.model==model].set_index('condition').loc[CONDITIONS];books=pd.read_csv(OUT/f'{model}_leading_thriller_titles.csv').head(6)
  fig,axs=plt.subplots(2,2,figsize=(20,11.25))
  for ax,key,title,ylabel in [(axs[0,0],'exact_mean_shared_titles','Shared titles across all real-user pairs','Mean shared titles out of 20'),(axs[0,1],'fixed_target_jaccard_percent','Within-Thriller ordering (including outside Top-20)','Mean Jaccard (%)')]:
   ax.bar(range(4),z[key],color=colors);ax.set_xticks(range(4),labels);ax.set_title(title);ax.set_ylabel(ylabel);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True);ax.set_ylim(0,max(z[key])*1.2)
   for j,value in enumerate(z[key]):ax.text(j,value+max(z[key])*.02,f'{value:.2f}',ha='center')
  x=np.arange(len(books));ax=axs[1,0];ax.bar(x-.18,books.baseline_reach_percent,width=.36,label='Baseline',color=colors[0]);ax.bar(x+.18,books.after_reach_percent,width=.36,label='Full after',color=colors[3]);ax.set_ylabel('Real users receiving title (%)');ax.set_title('Leading target books after injection');ax.set_xticks(x,books.book_id.astype(str));ax.set_xlabel('Book ID (full titles in accompanying table)');ax.legend(frameon=False);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
  ax=axs[1,1];ax.bar(x-.18,books.relative_embedding_delta,width=.36,color=colors[1],label='Embedding');ax.bar(x+.18,books.relative_bias_delta,width=.36,color=colors[2],label='Item bias');ax.axhline(0,color='#444',lw=.8);ax.set_xticks(x,books.book_id.astype(str));ax.set_title('Change in score relative to other unseen books');ax.set_ylabel('Mean relative score change');ax.set_xlabel('Book ID');ax.legend(frameon=False)
  fig.suptitle(f'{model.upper()} | Thriller, 500 injected users: what drives shared titles?',fontsize=23)
  fig.text(.5,.035,'Hybrid scores are algebraic diagnostics, not separately trained causal interventions. NMF embedding-only recommends ZERO Thriller titles in its Top-20.\nIts high two-title Jaccard measures within-Thriller ordering outside the actual Top-20; baseline/full reproduce the earlier selected-cohort control.',ha='center',fontsize=11)
  fig.subplots_adjust(top=.87,bottom=.16,left=.07,right=.98,hspace=.48,wspace=.22)
  dest=H/'result/Thriller/title_mechanism'/model.upper();dest.mkdir(parents=True,exist_ok=True);fig.savefig(dest/'mechanism_4panel.png',facecolor='white');plt.close(fig)
 fig,axs=plt.subplots(1,2,figsize=(16,9));x=np.arange(2);model_labels=['NMF','SVD']
 for condition,color,shift in [('baseline','#8C98A6',-.18),('full_after','#E77817',.18)]:
  d=hetero[hetero.condition==condition].set_index('model').loc[['nmf','svd']]
  for ax,key in zip(axs,['target_personalized_score_rms','common_to_personalized_ratio']):
   vals=d[key].to_numpy();ax.bar(x+shift,vals,width=.36,color=color,label='Baseline' if condition=='baseline' else '500 injected users')
   for xx,value in zip(x+shift,vals):ax.text(xx,value+max(hetero[key])*.025,f'{value:.3f}',ha='center',fontsize=12)
 for ax,key in zip(axs,['target_personalized_score_rms','common_to_personalized_ratio']):ax.set_xticks(x,model_labels);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True);ax.legend(frameon=False);ax.set_ylim(0,max(hetero[key])*1.2)
 axs[0].set_title('User-specific variation in Thriller scores',fontsize=16);axs[0].set_ylabel('Double-centered interaction RMS')
 axs[1].set_title('Common book-order signal / user-specific variation',fontsize=16);axs[1].set_ylabel('Ratio')
 fig.suptitle('Thriller injection: does scoring distinguish users less?',fontsize=22)
 fig.text(.5,.045,'Scores are computed from every real user and every target-tagged book. The interaction RMS removes user-constant and item-common terms.\nThese are score diagnostics; they do not measure human taste, recommendation accuracy, or a causal training effect.',ha='center',fontsize=11)
 fig.subplots_adjust(top=.85,bottom=.17,left=.08,right=.97,wspace=.25)
 dest=H/'result/Thriller/title_mechanism';dest.mkdir(parents=True,exist_ok=True);fig.savefig(dest/'score_differentiation_2panel.png',facecolor='white');plt.close(fig)
 report=['# What drives shared Thriller titles at 500 injected users?','', '## Four score combinations','', 'Baseline = G0 + bi0; embedding only = G1 + bi0; item bias only = G0 + bi1; full after = G1 + bi1. G = pu @ qi.T. The global mean and user bias cancel within each user’s raw-score ranking. Previously rated books are excluded.', '', table(conditions), '', '## Symmetric allocation of metric changes','', table(alloc), '', '## Personalized score heterogeneity within Thriller','', table(hetero), '', 'Within-target personalized score RMS is the root mean square of the double-centered interaction matrix: sqrt(trace(Cov(pu) Cov(qi_target))). It removes user-constant and item-common score terms and is invariant to equivalent factor reparameterization. Common-score spread is std_i(mean_u G_ui + bi_i) within target books. These are diagnostics of score variation, not measured preference or accuracy.', '', '## Limits', '', '- Hybrid scores are not retrained models. Symmetric allocations sum exactly to the observed output-metric change but are not estimates of causal effects of changing the training algorithm.', '- Bias-only and embedding-only combinations are descriptive substitutions at the saved fitted parameters. They can be implausible standalone training outcomes.', '- The target control fixes original pairs and exactly two target titles. In hybrid conditions those titles are selected from all unseen Thriller books, even if fewer than two appear in a hybrid Top-20; the baseline/full conditions exactly reproduce the original within-Top-20 control.', '- Relative book-score decomposition centers each user’s score by their mean score over all unseen candidates. Raw score increases alone do not establish rank improvements; observed recommendation reach and the hybrid rerankings provide separate evidence.', '- One training seed and one dense target=5 / other=1 protocol; only Thriller at 500 evaluated here. No additional training was run.']
 (H/'reports/title_mechanism.md').write_text('\n'.join(report))
 validation={'status':'complete','models':['nmf','svd'],'target':'Thriller','n_fake':500,'original_ranking_exact_match':True,'relative_score_identity':True,'symmetric_metric_allocations_sum_to_observed_change':True,'retrained':False}
 (H/'log/title_mechanism_validation.json').write_text(json.dumps(validation,indent=2))

if __name__=='__main__':
 with ProcessPoolExecutor(max_workers=2) as pool:list(pool.map(job,['nmf','svd']))
 finish()
