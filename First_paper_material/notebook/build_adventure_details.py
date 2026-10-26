#!/usr/bin/env python3
"""Legacy-compatible Adventure bi/bu/mio/merged tables and detail plots from raw checkpoints."""
import os
for key in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[key]='1'
from pathlib import Path
import json,time,hashlib
import numpy as np,pandas as pd
from scipy.sparse import csr_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/home/moshtasa/Research/phd-recsys');ROOT=R/'Book/current/PerturbRec/svd_nmf';L=ROOT/'layer1';E=ROOT/'layer2/embeddings';NB=R/'First_paper_material/notebook';COUNTS=[0,2,4,6,25,50,100,200,500,1000,2000,2500,3000,3500];GENRE='Adventure';PROTO='controlled_clean_catalog';FILES=[];IMAGES=[]
COLS={'bi':['n_fake','injected_genre','genre','avg_bi','std_bi','n_items','is_target','training_protocol'],'bu':['n_fake','injected_genre','genre','user_type','avg_bu','std_bu','n_users','is_target','training_protocol'],'mio':['n_fake','injected_genre','genre','avg_rating','std_rating','n_ratings','global_mean_mu','is_target','training_protocol'],'merged':['n_fake','injected_genre','genre','avg_bi','std_bi','n_items','is_target','user_type','avg_bu','std_bu','n_users','avg_rating','std_rating','n_ratings','global_mean_mu','training_protocol']};SUFFIX={'bi':'bi_per_genre','bu':'bu_per_genre_biased','mio':'mu_per_genre','merged':'merged'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':13,'axes.spines.top':False,'axes.spines.right':False})

def checkpoint(m,n):return E/'pilot'/f'{m}_baseline_a.npz' if n==0 else E/'controlled'/m/f'{m}_adventure_{n}.npz'
def load(m,n):
 with np.load(checkpoint(m,n)) as z:
  oi=np.argsort(z['items']);ou=np.argsort(z['ids']);real=ou[np.searchsorted(z['ids'][ou],USERS)];fake=np.flatnonzero(~np.isin(z['ids'],USERS));assert np.array_equal(z['items'][oi],ITEMS) and np.array_equal(z['ids'][real],USERS) and len(fake)==n
  return z['bi'][oi],z['bu'][real],z['bu'][fake],float(z['mu'])
def sample_stats(a):return float(np.mean(a)),float(np.std(a,ddof=1)),len(a)
def stats(m,n):
 bi,bu,fbu,mu=load(m,n);rows={k:[] for k in ['bi','bu','mio']}
 expected_mu=(REAL_SUM+n*SYNTH.sum())/(len(RATINGS)+n*len(ITEMS));assert np.isclose(expected_mu,mu,atol=1e-12,rtol=0)
 for j,g in enumerate(GENRES):
  mask=MASK[:,j];common=dict(n_fake=n,injected_genre=GENRE if n else 'Baseline',genre=g,is_target=('target' if g==GENRE else 'non_target') if n else 'ORIGINAL',training_protocol=PROTO)
  mean,std,count=sample_stats(bi[mask]);rows['bi'].append(dict(common,avg_bi=mean,std_bi=std,n_items=count))
  for name,values in [('non_biasing',bu[HISTORY[:,j]]),('biasing',fbu)]:
   if len(values):
    mean,std,count=sample_stats(values);rows['bu'].append(dict(common,user_type=name,avg_bu=mean,std_bu=std,n_users=count))
  count=int(ORIG_N[j]+n*mask.sum());total=ORIG_SUM[j]+n*SYNTH[mask].sum();square=ORIG_SQ[j]+n*(SYNTH[mask]**2).sum();avg=total/count;variance=(square-total**2/count)/(count-1);assert variance>=0
  rows['mio'].append(dict(common,avg_rating=avg,std_rating=float(np.sqrt(variance)),n_ratings=count,global_mean_mu=mu))
 result={k:pd.DataFrame(v)[COLS[k]] for k,v in rows.items()}
 keys=['n_fake','injected_genre','genre','is_target','training_protocol'];merged=result['bi'].merge(result['bu'],on=keys,validate='one_to_many').merge(result['mio'],on=keys,validate='many_to_one');result['merged']=merged[COLS['merged']]
 for k,d in result.items():
  assert np.isfinite(d.select_dtypes('number')).all().all();assert d.n_fake.nunique()==1 and d.n_fake.iloc[0]==n
 if not n:
  for key,new in result.items():
   old=pd.read_csv(L/'result'/m/key/f'ORIGINAL_{SUFFIX[key]}.csv');sort=['genre']+(['user_type'] if key in ['bu','merged'] else []);old=old.sort_values(sort).reset_index(drop=True);new=new.sort_values(sort).reset_index(drop=True);assert old.shape==new.shape,(m,key,old.shape,new.shape)
   for col in new.select_dtypes('number').columns:assert np.allclose(old[col],new[col],rtol=1e-10,atol=1e-10),(m,key,col)
 print('STATS',m,n,flush=True);return result

def figure_base(model,title,ylabel):
 fig,ax=plt.subplots(figsize=(16,9));fig.subplots_adjust(left=.09,right=.96,bottom=.31,top=.86);ax.set_title(f'{model.upper()} · Adventure injection · {title}',fontsize=21,fontweight='bold',pad=18);ax.set_ylabel(ylabel,fontsize=15);ax.set_xticks(range(len(COUNTS)),['Baseline']+list(map(str,COUNTS[1:])),rotation=45,ha='right',fontsize=10);ax.set_xlabel('Number of injected users',fontsize=13);ax.grid(axis='y',color='#DFE5EB',lw=.8);ax.set_axisbelow(True);ax.spines['left'].set_color('#444444');ax.spines['bottom'].set_color('#444444');fig.text(.09,.185,'Sizes are equally spaced categories; multi-genre books and user-history groups overlap.',fontsize=10,color='#607286')
 if model=='nmf':fig.text(.09,.155,'Raw results retained: NMF factor sign check failed at n = 1000, 2000 and 2500.',fontsize=10,color='#9D211C')
 return fig,ax

def draw(model,data):
 out=L/'result'/model/f'{model.upper()}_details_figures';assert out.is_dir();x=np.arange(len(COUNTS));bi=data['bi'];bu=data['bu'];mu=data['mio'];palette=plt.get_cmap('tab20').colors
 for name,title,ylabel,key in [('bi_only','Book bias',r'$b_i$','avg_bi'),('mu_plus_bi','Global mean + book bias',r'$\mu+b_i$','sum'),('bu_only_non_biasing','Real-user bias by history genre',r'$b_u$','avg_bu')]:
  fig,ax=figure_base(model,title,ylabel)
  for j,g in enumerate([v for v in GENRES if v!=GENRE]+[GENRE]):
   if key=='avg_bu':s=bu[(bu.genre==g)&(bu.user_type=='non_biasing')].sort_values('n_fake');values=s.avg_bu.to_numpy()
   else:
    s=bi[bi.genre==g].sort_values('n_fake');values=s.avg_bi.to_numpy()
    if key=='sum':values=values+mu[mu.genre==g].sort_values('n_fake').global_mean_mu.to_numpy()
   assert s.n_fake.tolist()==COUNTS;target=g==GENRE;ax.plot(x,values,'o-',color='#B8392F' if target else palette[j],lw=3 if target else 1.6,ms=4,label=g+(' (target)' if target else ''))
  fig.legend(*ax.get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.51,.015),ncol=4,frameon=False,fontsize=10);p=out/f'Adventure_{name}.png';fig.savefig(p,dpi=240,facecolor='white');plt.close(fig);IMAGES.append(str(p))
 fig,ax=figure_base(model,'Global mean rating',r'$\mu$');s=mu[mu.genre==GENRE].sort_values('n_fake');ax.plot(x,s.global_mean_mu,'o-',color='#B8392F',lw=3,ms=4,label='Global mean (all genres)');fig.legend(*ax.get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.51,.04),frameon=False,fontsize=11);p=out/'Adventure_mu_only.png';fig.savefig(p,dpi=240,facecolor='white');plt.close(fig);IMAGES.append(str(p))
 fig,ax=figure_base(model,'Real and synthetic user biases',r'$b_u$')
 for user_type,group,color,ls,label in [('non_biasing','target','#1F77B4','-','Real users: target-history group'),('non_biasing','other','#80B7DA','--','Real users: other genre groups'),('biasing','target','#159155','-','Synthetic users: target group'),('biasing','other','#74C29C','--','Synthetic users: other genre groups')]:
  vals=[]
  for n in COUNTS:
   s=bu[(bu.n_fake==n)&(bu.user_type==user_type)&((bu.genre==GENRE) if group=='target' else (bu.genre!=GENRE))];vals.append(float(s.avg_bu.mean()) if len(s) else np.nan)
  ax.plot(x,vals,marker='o',color=color,ls=ls,lw=2.5,ms=5,label=label)
 fig.legend(*ax.get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.51,.015),ncol=2,frameon=False,fontsize=10.5);p=out/'Adventure_bu_biased_vs_unbiased_target_vs_nontarget.png';fig.savefig(p,dpi=240,facecolor='white');plt.close(fig);IMAGES.append(str(p))

if __name__=='__main__':
 tick=time.time();df=pd.read_csv(R/'Book/Archive/data/genre_clean.csv',usecols=['user_id','book_id','rating','genres']);USERS=np.sort(df.user_id.unique());ITEMS=np.sort(df.book_id.unique());ui=np.searchsorted(USERS,df.user_id.to_numpy());ii=np.searchsorted(ITEMS,df.book_id.to_numpy());RATINGS=df.rating.to_numpy(float);REAL_SUM=float(RATINGS.sum());meta=df.drop_duplicates('book_id').set_index('book_id').loc[ITEMS];tags=[{v.strip() for v in str(s).split(',') if v.strip() and v.lower()!='unknown'} for s in meta.genres];GENRES=sorted(set.union(*tags));assert len(GENRES)==13;MASK=np.array([[g in t for g in GENRES] for t in tags]);seen=csr_matrix((np.ones(len(df),bool),(ui,ii)),shape=(len(USERS),len(ITEMS)));HISTORY=np.asarray((seen.astype(np.int32)@MASK.astype(np.int32))>0);SYNTH=np.where(MASK[:,GENRES.index(GENRE)],5,1);bn=np.bincount(ii,minlength=len(ITEMS));bs=np.bincount(ii,weights=RATINGS,minlength=len(ITEMS));bq=np.bincount(ii,weights=RATINGS**2,minlength=len(ITEMS));ORIG_N=bn@MASK;ORIG_SUM=bs@MASK;ORIG_SQ=bq@MASK;assert len(USERS)==53424 and len(ITEMS)==9415 and len(df)==5815569
 for model in ['nmf','svd']:
  result=[stats(model,n) for n in COUNTS];data={k:pd.concat([v[k] for v in result],ignore_index=True) for k in COLS}
  for key,d in data.items():
   for bucket in [2,4,6,25,50,100,200,500,1000,2000]:
    selected=d[d.n_fake.isin([2000,2500,3000,3500]) if bucket==2000 else d.n_fake.eq(bucket)];expected=(52 if key in ['bi','mio'] else 104) if bucket==2000 else (13 if key in ['bi','mio'] else 26);assert len(selected)==expected
    path=L/'result'/model/key/f'f_Adventure_{bucket}u_pos5_neg1_books_{SUFFIX[key]}.csv';assert path.parent.is_dir();selected[COLS[key]].to_csv(path,index=False);back=pd.read_csv(path);assert back.shape==selected.shape;assert np.allclose(back.select_dtypes('number'),selected.select_dtypes('number'),rtol=1e-12,atol=1e-12);FILES.append(str(path))
  # Cross-check against the independently generated score/bias analysis.
  old=pd.read_csv(L/'tables/adventure_genre_biases.csv');own=data['bi'];check=own.merge(old[old.model==model],on=['n_fake','genre']);assert len(check)==14*13 and np.allclose(check.avg_bi,check.mean_bi,rtol=1e-12,atol=1e-12)
  draw(model,data)
 manifest=dict(status='complete',genre=GENRE,models=['nmf','svd'],counts=COUNTS,csv_files=FILES,images=IMAGES,baseline_files_reused_and_numerically_verified=True,original_files_unchanged=True,new_directories_created=False,raw_values_preserved=True,extended_doses_bucket=2000,extended_doses=[2000,2500,3000,3500],std_convention='sample standard deviation (ddof=1)',user_genre_groups='Unique real users who rated any book with that genre, irrespective of rating; groups overlap. Every dense synthetic profile belongs to all genre groups.',other_group_plot='Unweighted mean across non-target genre-group mean user biases; not a disjoint pooled user group.',nonnegative_constraint_failures=[1000,2000,2500],elapsed_seconds=time.time()-tick,finished=time.time());(NB/'adventure_details_manifest.json').write_text(json.dumps(manifest,indent=2));print('COMPLETE',len(FILES),'CSV files;',len(IMAGES),'images;',round(time.time()-tick),'seconds',flush=True)
