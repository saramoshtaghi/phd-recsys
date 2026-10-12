#!/usr/bin/env python3
"""Post-training test of the reviewer's sign/bias hypothesis; no retraining.

Outputs separate algebraic identities, descriptive evidence, and conditional
score-component swaps. Swaps are not causal retraining interventions.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(key,'1')
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
DEFAULT_ROOT=Path('/home/moshtasa/Research/phd-recsys/Book/current/PerturbRec/svd_nmf')
DEFAULT_DATA=Path('/home/moshtasa/Research/phd-recsys/Book/Archive/data/genre_clean.csv')
COUNTS=(2,4,6,25,50,100,200,500,1000,2000)

def mean(x):return float(np.mean(x)) if np.size(x) else None

def top_counts(score,seen,target,ks):
    score=np.array(score,copy=True);score[seen]=-np.inf
    assert np.all((~seen).sum(axis=1)>=max(ks))
    # Ascending raw item IDs resolve ties (columns are sorted beforehand).
    counts={}
    for k in ks:
        threshold=np.partition(score,-k,axis=1)[:,-k]
        higher=score>threshold[:,None];ties=score==threshold[:,None]
        remaining=k-higher.sum(1)
        chosen=higher|(ties&(ties.cumsum(1)<=remaining[:,None]))
        assert np.all(chosen.sum(1)==k)
        counts[k]=(chosen&target).sum(axis=1)
    return counts

def align(z,users,items):
    ids=z['ids'];rawitems=z['items'];uorder=np.argsort(ids)
    idx=np.searchsorted(ids[uorder],users)
    assert np.all(idx<len(ids)) and np.array_equal(ids[uorder[idx]],users)
    ui=uorder[idx];ii=np.argsort(rawitems)
    assert np.array_equal(rawitems[ii],items)
    fake=np.flatnonzero(~np.isin(ids,users))
    return {'p':z['pu'][ui].astype(float),'q':z['qi'][ii].astype(float),
            'bu':z['bu'][ui].astype(float),'bi':z['bi'][ii].astype(float),
            'mu':float(z['mu']),'fp':z['pu'][fake].astype(float),
            'fbu':z['bu'][fake].astype(float)}

def observed_components(m,u,i):
    interaction=np.einsum('ij,ij->i',m['p'][u],m['q'][i])
    return interaction,m['bu'][u],m['bi'][i]

def synthetic_stats(m,target,eps=1e-10):
    """Exact moments of dense fake-user pairs in manageable blocks."""
    out=[]
    for label,mask,rating in [('target',target,5.),('non_target',~target,1.)]:
        total=0;sums=np.zeros(8);neg_interact=0;need_neg=0;impossible=0;mininteraction=np.inf
        b=m['bi'][mask];q=m['q'][mask]
        for a in range(0,len(m['fp']),64):
            p=m['fp'][a:a+64];bu=m['fbu'][a:a+64,None]
            interaction=p@q.T;offset=m['mu']+bu+b
            pred=offset+interaction;resid=pred-rating
            total+=interaction.size
            sums+=np.array([interaction.sum(),offset.sum(),pred.sum(),resid.sum(),
                            np.abs(resid).sum(),(resid**2).sum(),
                            np.broadcast_to(b,interaction.shape).sum(),
                            np.broadcast_to(bu,interaction.shape).sum()])
            neg_interact+=np.count_nonzero(interaction < -eps)
            # If offset exceeds rating, reaching the rating requires a negative
            # interaction contribution at that fixed offset.
            need_neg+=np.count_nonzero(offset>rating+eps)
            impossible+=np.count_nonzero((offset>rating+eps)&(interaction>=-eps))
            mininteraction=min(mininteraction,float(interaction.min()))
        if not total:continue
        vals=sums/total
        out.append(dict(pair_group=label,rating=rating,n_pairs=total,
                        mean_interaction=vals[0],mean_offset=vals[1],mean_prediction=vals[2],
                        mean_signed_residual=vals[3],mae=vals[4],rmse=float(np.sqrt(vals[5])),
                        mean_bi=vals[6],mean_bu=vals[7],mu=m['mu'],
                        negative_interaction_percent=100*neg_interact/total,
                        offset_above_rating_percent=100*need_neg/total,
                        above_rating_offset_with_nonnegative_interaction_percent=100*impossible/total,
                        min_interaction=mininteraction))
    return out

def analyze_case(base,after,target,u,i,rating,seen,ks=(15,25,35),block=256):
    gi0,bu0,bi0=observed_components(base,u,i)
    gi1,bu1,bi1=observed_components(after,u,i)
    dmu=after['mu']-base['mu'];dgi=gi1-gi0;dbu=bu1-bu0;dbi=bi1-bi0
    before=base['mu']+bu0+bi0+gi0;final=after['mu']+bu1+bi1+gi1
    assert np.allclose(final-before,dmu+dbu+dbi+dgi,rtol=1e-10,atol=1e-10)
    groups=[]
    for label,mask in [('target_observed',target[i]),('non_target_observed',~target[i]),
                       ('non_target_low_rating_observed',(~target[i])&(rating<=2))]:
        if not mask.any():continue
        groups.append(dict(pair_group=label,n_pairs=int(mask.sum()),
            mean_prediction_before=mean(before[mask]),mean_prediction_after=mean(final[mask]),
            mean_delta_prediction=mean((final-before)[mask]),delta_mu=float(dmu),
            mean_delta_bu=mean(dbu[mask]),mean_delta_bi=mean(dbi[mask]),
            mean_delta_interaction=mean(dgi[mask]),
            bi_decrease_percent=100*mean(dbi[mask]<-1e-10),
            interaction_decrease_percent=100*mean(dgi[mask]<-1e-10),
            mae_before=mean(np.abs(before[mask]-rating[mask])),
            mae_after=mean(np.abs(final[mask]-rating[mask]))))
    # Mean target-vs-other candidate margins, equal weight for each eligible user.
    margin_total={name:0. for name in ('before_interaction','after_interaction','before_bi','after_bi')}
    valid=0;totals={name:{k:0 for k in ks} for name in ('baseline','after','embedding_only','item_bias_only')}
    for a in range(0,len(base['p']),block):
        b=min(a+block,len(base['p']));mask=seen[a:b].toarray().astype(bool)
        g0=base['p'][a:b]@base['q'].T;g1=after['p'][a:b]@after['q'].T
        eligible=~mask;tm=eligible&target;nm=eligible&~target
        nt=tm.sum(1);nn=nm.sum(1);ok=(nt>0)&(nn>0);valid+=int(ok.sum())
        for name,matrix in [('before_interaction',g0),('after_interaction',g1),
                            ('before_bi',np.broadcast_to(base['bi'],g0.shape)),
                            ('after_bi',np.broadcast_to(after['bi'],g0.shape))]:
            delta=(matrix[ok]*tm[ok]).sum(1)/nt[ok]-(matrix[ok]*nm[ok]).sum(1)/nn[ok]
            margin_total[name]+=float(delta.sum())
        for name,score in [('baseline',g0+base['bi']),('after',g1+after['bi']),
                           ('embedding_only',g1+base['bi']),('item_bias_only',g0+after['bi'])]:
            counts=top_counts(score,mask,target,ks)
            for k in ks:totals[name][k]+=int(counts[k].sum())
    margin={key:value/valid for key,value in margin_total.items()} if valid else {}
    if valid:
        margin['delta_interaction_margin']=margin['after_interaction']-margin['before_interaction']
        margin['delta_bi_margin']=margin['after_bi']-margin['before_bi']
        margin['delta_total_margin']=margin['delta_interaction_margin']+margin['delta_bi_margin']
    rec=[]
    for k in ks:
        means={key:value[k]/len(base['p']) for key,value in totals.items()}
        # Symmetric two-component allocation handles ranking nonlinearity.
        # This is descriptive post-training Shapley allocation, not causal.
        emb=.5*((means['embedding_only']-means['baseline'])+(means['after']-means['item_bias_only']))
        bias=.5*((means['item_bias_only']-means['baseline'])+(means['after']-means['embedding_only']))
        assert np.isclose(emb+bias,means['after']-means['baseline'])
        rec.append(dict(k=k,n_users=len(base['p']),**{key+'_count':value for key,value in means.items()},
                        delta_count=means['after']-means['baseline'],
                        embedding_swap_allocation=emb,item_bias_swap_allocation=bias))
    return groups,margin,rec

def run(args):
    root=args.root;output=args.output or root/'layer1/result/reviewer_hypothesis'
    df=pd.read_csv(args.data,usecols=['user_id','book_id','rating','genres'])
    users=np.sort(df.user_id.unique());items=np.sort(df.book_id.unique())
    u=np.searchsorted(users,df.user_id.to_numpy());i=np.searchsorted(items,df.book_id.to_numpy())
    rating=df.rating.to_numpy(dtype=float)
    tags=dict(df[['book_id','genres']].drop_duplicates('book_id').itertuples(index=False,name=None))
    tags={key:{g.strip() for g in str(value).split(',') if g.strip() and g.strip().lower()!='unknown'} for key,value in tags.items()}
    assert all(tags[x] for x in items),'Unknown/unlabelled item remains in clean catalog'
    seen=csr_matrix((np.ones(len(df),bool),(u,i)),shape=(len(users),len(items)))
    all_obs=[];all_fake=[];all_margin=[];all_rec=[];sources=[]
    for model in ('nmf','svd'):
        basepath=root/'layer2/embeddings/pilot'/f'{model}_baseline_a.npz'
        with np.load(basepath) as z:base=align(z,users,items)
        assert np.isclose(base['mu'],rating.mean(),rtol=0,atol=1e-10), 'Baseline mean does not match real data'
        repeat=root/'layer2/embeddings/pilot'/f'{model}_baseline_b.npz'
        if repeat.exists():
            with np.load(basepath) as za, np.load(repeat) as zb:
                assert set(za.files)==set(zb.files)
                assert all(np.array_equal(za[key],zb[key]) for key in za.files), 'Baseline repeat differs'
        for genre in ('Drama','Thriller'):
            target=np.array([genre in tags[x] for x in items]);assert target.any() and (~target).any()
            for n in args.counts:
                path=root/'layer2/embeddings/controlled'/model/genre.lower()/f'{model}_{genre.lower()}_{n}.npz'
                with np.load(path) as z:after=align(z,users,items)
                assert len(after['fp'])==n
                expected_mu=(rating.sum()+n*np.where(target,5.,1.).sum())/(len(rating)+n*len(items))
                assert np.isclose(after['mu'],expected_mu,rtol=0,atol=1e-10), 'Augmented mean does not match dense injection'
                if model=='nmf':
                    assert min(after['p'].min(),after['q'].min(),after['fp'].min())>=-1e-12
                metadata={'model':model,'genre':genre,'n_fake':n}
                observed,margin,recs=analyze_case(base,after,target,u,i,rating,seen,ks=tuple(args.k),block=args.block)
                all_obs.extend([{**metadata,**row} for row in observed])
                all_fake.extend([{**metadata,**row} for row in synthetic_stats(after,target)])
                all_margin.append({**metadata,**margin})
                all_rec.extend([{**metadata,**row} for row in recs])
                sources.append({'baseline':str(basepath),'after':str(path),**metadata})
                dest=output/model.upper()/genre;dest.mkdir(parents=True,exist_ok=True)
                pd.DataFrame([x for x in all_obs if x['model']==model and x['genre']==genre]).to_csv(dest/'real_observed_score_components.csv',index=False)
                pd.DataFrame([x for x in all_fake if x['model']==model and x['genre']==genre]).to_csv(dest/'synthetic_score_components.csv',index=False)
                pd.DataFrame([x for x in all_margin if x['model']==model and x['genre']==genre]).to_csv(dest/'candidate_margins.csv',index=False)
                pd.DataFrame([x for x in all_rec if x['model']==model and x['genre']==genre]).to_csv(dest/'recommendation_component_swaps.csv',index=False)
                print('DONE',model,genre,n,flush=True)
    tables={'observed':pd.DataFrame(all_obs),'synthetic':pd.DataFrame(all_fake),
            'margins':pd.DataFrame(all_margin),'recommendations':pd.DataFrame(all_rec)}
    plot(tables,output)
    provenance={'status':'complete','data':str(args.data),'data_sha256':hashlib.sha256(args.data.read_bytes()).hexdigest(),
                'users':len(users),'items':len(items),'real_ratings':len(df),'cases':sources,
                'counts':args.counts,'k':args.k,'no_retraining':True,
                'interpretation':'Component swaps are conditional fitted-score decompositions, not causal retraining interventions.'}
    (output/'validation.json').write_text(json.dumps(provenance,indent=2))
    (output/'README.md').write_text(METHOD)
    print('COMPLETE',output,flush=True)

METHOD='''# Reviewer hypothesis: sign constraint and bias compensation

Saved controlled SVD/NMF checkpoints are analyzed without retraining. Original
real users/items are aligned by raw IDs; fake users are identified as IDs absent
from the original data. Unknown labels are excluded by requiring the clean catalog.

1. Dense synthetic target-5 and other-1 pairs: signed interaction scores, biases,
   offset, prediction, residual, and fraction with a negative interaction.
2. Original observed pairs: exact before/after prediction decomposition into
   global mean, user bias, item bias, and interaction. Low ratings (<=2) on
   non-target items are separately reported, without treating them as a new test set.
3. Unseen candidate pairs: each user's mean target-vs-non-target raw score margin
   is split into item-bias and interaction parts. Mean/user weighting is used.
4. Top-15/25/35 after excluding previously rated items: baseline, fully augmented,
   augmented embeddings with baseline item bias, and baseline embeddings with
   augmented item bias. Sorted raw book IDs provide stable tie resolution.
   Symmetric two-component allocation sums to the total exposure change, including
   nonlinearity; it is not a causal proof because components were jointly trained.

NMF's nonnegative interaction is an algebraic constraint, not empirical confirmation
of the reviewer's full explanation. Evidence for bias compensation requires actual
bias/offset changes alongside low-rating fit. A decrease in a positive interaction
can also lower predictions. No component percentage is reported when opposite signs
or a near-zero net change would make that percentage misleading. Larger SVD negative
interaction usage and stronger NMF bias compensation would support, but not prove,
the proposed mechanism. A trained-model constraint/optimizer ablation is needed for
exclusive causal attribution. Raw predictions are not clipped before decomposition.

Charts use linear axes and PNG output only; counts stop at 2,000.
'''

def plot(tables,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    for model in ('nmf','svd'):
      for genre in ('Drama','Thriller'):
        fig,axes=plt.subplots(2,2,figsize=(13,8))
        fake=tables['synthetic'];fake=fake[(fake.model==model)&(fake.genre==genre)&(fake.pair_group=='non_target')].sort_values('n_fake')
        x=np.arange(len(fake));ticks=fake.n_fake.astype(str)
        axes[0,0].plot(x,fake.negative_interaction_percent,'o-',color='#3066A5')
        axes[0,0].set_title('Fake non-target pairs: negative interaction');axes[0,0].set_ylabel('Percent of pairs');axes[0,0].set_ylim(0,100)
        axes[0,1].plot(x,fake.mean_interaction,'o-',label='Interaction',color='#3066A5')
        axes[0,1].plot(x,fake.mean_offset,'o-',label='Global mean + user bias + item bias',color='#B34032')
        axes[0,1].plot(x,fake.mean_prediction,'o-',label='Full prediction',color='#238650')
        axes[0,1].axhline(1,color='#333',ls=':',label='Observed rating = 1')
        axes[0,1].set_title('How fake non-target ratings are fitted');axes[0,1].set_ylabel('Mean raw score contribution');axes[0,1].legend(fontsize=8)
        margin=tables['margins'];margin=margin[(margin.model==model)&(margin.genre==genre)].sort_values('n_fake')
        for name,label,color in [('delta_interaction_margin','Interaction','#3066A5'),('delta_bi_margin','Item bias','#B34032')]:axes[1,0].plot(x,margin[name],'o-',label=label,color=color)
        axes[1,0].axhline(0,color='#777',lw=.7);axes[1,0].set_title('Change in target-versus-other candidate margin');axes[1,0].set_ylabel('Mean raw score change');axes[1,0].legend()
        rec=tables['recommendations'];rec=rec[(rec.model==model)&(rec.genre==genre)&(rec.k==tables['recommendations'].k.min())].sort_values('n_fake')
        for name,label,color in [('embedding_swap_allocation','Embedding swap','#3066A5'),('item_bias_swap_allocation','Item-bias swap','#B34032'),('delta_count','Total change','#238650')]:axes[1,1].plot(x,rec[name],'o-',label=label,color=color)
        axes[1,1].axhline(0,color='#777',lw=.7);axes[1,1].set_title(f'Top-{int(rec.k.iloc[0])}: conditional component allocation');axes[1,1].set_ylabel('Change in mean target-book count');axes[1,1].legend(fontsize=9)
        for ax in axes.flat:ax.set_xticks(x,ticks,rotation=45);ax.set_xlabel('Injected users');ax.grid(axis='y',alpha=.2)
        fig.suptitle(f'{model.upper()} · {genre} · Reviewer hypothesis',weight='bold',fontsize=17)
        fig.text(.04,.018,'Saved-model diagnostics; component swaps do not establish a causal training mechanism.',fontsize=10)
        fig.tight_layout(rect=(0,.04,1,.94));fig.savefig(output/model.upper()/genre/'reviewer_hypothesis_4panel.png',dpi=240);plt.close(fig)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    parser.add_argument('--data',type=Path,default=DEFAULT_DATA)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--counts',nargs='+',type=int,default=list(COUNTS))
    parser.add_argument('--k',nargs='+',type=int,default=[15,25,35])
    parser.add_argument('--block',type=int,default=256)
    args=parser.parse_args();run(args)
