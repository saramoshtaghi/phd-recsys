"""Identity turnover vs baseline and Top-15 genre shares of current 20 real neighbors."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import sys,json,time,hashlib
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap,Normalize
from matplotlib.patches import Rectangle
from scipy.sparse import csr_matrix
ROOT=Path('/home/moshtasa/Research/phd-recsys/Book/current/PerturbRec/svd_nmf/layer2')
sys.path.insert(0,str(ROOT/'materials/notebook'))
from embedding_utils import neighbors,real_rows,parse_genres
MODEL='nmf';TARGET='Thriller';DOSES=[0,2,4,6,25,50,100,200,500]
OUT=ROOT/'final_results/NMF/Thriller';DATA=ROOT/'materials/tables/neighborhood_top15/nmf/Thriller';DATA.mkdir(parents=True,exist_ok=True)
SRC=Path('/home/moshtasa/Research/phd-recsys/Book/Archive/data/genre_clean.csv')
base=np.load(ROOT/'embeddings/pilot/nmf_baseline_a.npz');uorder=np.argsort(base['ids']);iorder=np.argsort(base['items']);ids=base['ids'][uorder];items=base['items'][iorder]
df=pd.read_csv(SRC,usecols=['user_id','book_id','genres']);ur=np.searchsorted(ids,df.user_id.to_numpy());ir=np.searchsorted(items,df.book_id.to_numpy());assert np.array_equal(ids[ur],df.user_id) and np.array_equal(items[ir],df.book_id)
seen=csr_matrix((np.ones(len(df),bool),(ur,ir)),shape=(len(ids),len(items)))
booklabels={int(i):[g for g in parse_genres(v) if g.lower()!='unknown'] for i,v in df[['book_id','genres']].drop_duplicates('book_id').itertuples(index=False,name=None)}
genres=sorted({g for gs in booklabels.values() for g in gs});genres=[TARGET]+[g for g in genres if g!=TARGET];M=np.array([[g in booklabels[int(i)] for g in genres] for i in items],dtype=np.int16)
old=pd.read_csv(ROOT/'materials/tables/neighborhood_taste/genre_table_old_new.csv');old=old[(old.model==MODEL)&(old.target==TARGET)&(old.genre==TARGET)].set_index('n_fake')
result=[];turnovers=[];checks=[];baseline_nn=None
for n in DOSES:
 z=base if n==0 else np.load(ROOT/f'embeddings/controlled/{MODEL}/{TARGET.lower()}/{MODEL}_{TARGET.lower()}_{n}.npz')
 ui=np.argsort(z['ids']);ui=ui[real_rows(z['ids'][ui],ids,expected_fake=n)];ii=np.argsort(z['items']);assert np.array_equal(z['items'][ii],items)
 pu=z['pu'][ui];qi=z['qi'][ii];bi=z['bi'][ii];assert np.isfinite(pu).all()
 nbfile=DATA/f'{n}_neighbors.npz'
 if nbfile.exists():
  q=np.load(nbfile);assert np.array_equal(q['ids'],ids);nn=q['neighbor_indices'];ties=int(q['boundary_ties'])
 else:
  print('KNN',n,flush=True);nn,ties=neighbors(pu,ids,ids,k=20,metric='cosine',workers=4);np.savez_compressed(nbfile,ids=ids,neighbor_indices=nn,boundary_ties=ties)
 assert nn.shape==(len(ids),20) and not (ids[nn]==ids[:,None]).any();assert np.all(np.sort(nn,axis=1)[:,1:]!=np.sort(nn,axis=1)[:,:-1])
 if n==0:baseline_nn=nn.copy()
 retained=(nn[:,:,None]==baseline_nn[:,None,:]).any(axis=2);turnover=100*(~retained).mean();turnovers.append(turnover)
 assert np.isclose(turnover,old.loc[n,'neighbor_turnover_percent'],atol=1e-8),(n,turnover,old.loc[n,'neighbor_turnover_percent'])
 path=ROOT/'history/data/nmf'/('baseline_top20.npz' if n==0 else f'thriller_{n}_top20.npz');cache=np.load(path);assert np.array_equal(cache['ids'],ids) and np.array_equal(cache['items'],items);top=cache['top'][:,:15]
 assert top.shape==(len(ids),15)
 # Confirm saved ordered Top-20 prefix is the exact Top-15 score ranking for fixed users.
 sample=np.random.default_rng(42).choice(len(ids),20,replace=False);mask=seen[sample].toarray();scores=pu[sample]@qi.T+bi;scores[mask]=-np.inf;expected=np.argsort(-scores,axis=1,kind='stable')[:,:15];assert np.array_equal(top[sample],expected)
 selected=seen[np.arange(len(ids))[:,None],top]
 selected=selected.toarray() if hasattr(selected,'toarray') else np.asarray(selected)
 assert not selected.any()
 counts=M[top].sum(axis=1);weights=np.bincount(nn.ravel(),minlength=len(ids));shares=weights@counts.astype(float)/(len(ids)*20*15)*100
 old_top20_share=weights@M[cache['top']].sum(axis=1).astype(float)/(len(ids)*20*20)*100
 assert np.isclose(old_top20_share[0],old.loc[n,'total_percent'],atol=1e-8)
 # Independent direct nested averaging for a sample of egos.
 direct=(counts[nn[sample]].astype(float)/15*100).mean(axis=(0,1));weighted=np.bincount(nn[sample].ravel(),minlength=len(ids))@counts.astype(float)/(len(sample)*20*15)*100;assert np.allclose(direct,weighted)
 for g,v in zip(genres,shares):result.append(dict(model=MODEL,injected_genre=TARGET,n_fake=n,genre=g,n_egos=len(ids),n_neighbors=20,recommendation_k=15,mean_genre_share_percent=float(v)))
 checks.append(dict(n_fake=n,turnover_percent=float(turnover),mean_replaced_neighbors=float(turnover/5),knn_boundary_ties=ties,top15_exact_ranking_sample_checks=len(sample),top20_source=str(path)))
 print('DONE',n,'turnover',round(turnover,5),'Thriller',round(shares[0],5),flush=True)
table=pd.DataFrame(result);table.to_csv(DATA/'genre_shares_top15.csv',index=False);pd.DataFrame(checks).to_csv(DATA/'neighbor_turnover.csv',index=False)
values=table.pivot(index='genre',columns='n_fake',values='mean_genre_share_percent').loc[genres,DOSES]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','savefig.facecolor':'white'})
fig=plt.figure(figsize=(20,11.25));gs=fig.add_gridspec(2,1,height_ratios=[1,3.25],hspace=.40);ax=fig.add_subplot(gs[0]);hm=fig.add_subplot(gs[1]);x=np.arange(len(DOSES));labels=['Baseline']+[str(n) for n in DOSES[1:]]
ax.bar(x,turnovers,color='#0072B2',width=.68)
for j,v in enumerate(turnovers):
 if j:ax.text(j,v+1.7,f'{v:.1f}%',ha='center',va='bottom',fontsize=12,fontweight='bold',color='#17364A')
ax.set_xticks(x,labels);ax.set_ylim(0,115);ax.set_yticks([0,25,50,75,100]);ax.set_ylabel('Neighbors replaced (%)');ax.set_title('Replacement among the 20 nearest real neighbors — compared with baseline',loc='left',fontsize=15,pad=14);ax.grid(axis='y',alpha=.16);ax.set_axisbelow(True)
cmap=LinearSegmentedColormap.from_list('share',['#F4F8FA','#B5D9E2','#5CA9BB','#12627B']);norm=Normalize(vmin=0,vmax=100);im=hm.imshow(values.to_numpy(),aspect='auto',cmap=cmap,norm=norm)
hm.set_yticks(range(len(genres)),genres,fontsize=12);hm.get_yticklabels()[0].set_fontweight('bold');hm.get_yticklabels()[0].set_fontsize(15);hm.set_xticks(x,labels);hm.set_xlabel('Injected users',labelpad=12);hm.set_title('Mean genre share in the current 20 neighbors’ own Top-15 recommendations',loc='left',fontsize=15,pad=16)
for i in range(len(genres)):
 for j in range(len(DOSES)):
  v=values.iloc[i,j];hm.text(j,i,f'{v:.1f}%',ha='center',va='center',fontsize=12 if i==0 else 11,fontweight='bold' if i==0 else 'normal',color='white' if v>65 else '#163342')
hm.set_xticks(np.arange(-.5,len(DOSES),1),minor=True);hm.set_yticks(np.arange(-.5,len(genres),1),minor=True);hm.grid(which='minor',color='white',lw=1.3);hm.tick_params(which='minor',bottom=False,left=False);hm.add_patch(Rectangle((-.5,-.5),len(DOSES),1,fill=False,edgecolor='#172B3A',linewidth=2.2,clip_on=False))
# Dedicated colorbar axis preserves identical column positions in both panels.
fig.subplots_adjust(left=.12,right=.925,top=.89,bottom=.075);pos=hm.get_position();cax=fig.add_axes([.94,pos.y0,.012,pos.height]);fig.colorbar(im,cax=cax,label='Genre share (%)');fig.suptitle('NMF | Real-user neighborhoods',x=.12,ha='left',fontsize=23,fontweight='bold',y=.975)
out=OUT/'neighborhood_recommendation_change_heatmap_up_to_500.png';fig.savefig(out,dpi=200);fig.savefig(DATA/'preview.png',dpi=100);plt.close(fig)
report={'status':'complete','model':MODEL,'injected_genre':TARGET,'doses':DOSES,'n_real_users':len(ids),'neighbor_k':20,'recommendation_k':15,'neighbor_selection':'exact cosine in original saved user-factor space; self and synthetic users excluded; smaller raw user ID breaks ties','turnover_reference':'baseline identities at every dose, not previous injection level','bottom_metric':'mean over real egos of mean over their current 20 real neighbors of 100*genre count in own Top15/15','history_and_real_ids_unchanged':True,'recommendations':'previously rated books excluded; raw pu@qi.T+bi; ties by raw book ID; first 15 of saved exact ordered Top20 validated','genre_counting':'inclusive multi-label; genre percentages need not sum to 100','no_observed_human_taste_change_claim':True,'top_panel_baseline_annotation':None,'heatmap_color':'absolute share 0 to100; no delta numbers','in_figure_caption_removed':True,'no_retraining':True,'checks':checks,'output':str(out)}
(DATA/'validation.json').write_text(json.dumps(report,indent=2));print('SAVED',out,flush=True)
