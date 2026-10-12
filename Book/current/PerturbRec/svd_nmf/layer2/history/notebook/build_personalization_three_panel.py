from pathlib import Path
import sys,json,textwrap,re,argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
R=Path('/home/moshtasa/Research/phd-recsys/Book/current/PerturbRec/svd_nmf/layer2');H=R/'history'
sys.path.insert(0,str(H/'notebook'))
from analyze_personalization import intersection,l
p=argparse.ArgumentParser();p.add_argument('--model',choices=['nmf','svd'],default='nmf');model=p.parse_args().model
base=np.load(H/'data'/model/'baseline_top20.npz');after=np.load(H/'data'/model/'thriller_500_top20.npz');ids=base['ids'];items=base['items']
pairs=pd.read_csv(H/'tables'/f'{model}_fixed_user_pairs.csv.gz');a=np.searchsorted(ids,pairs.user_a);b=np.searchsorted(ids,pairs.user_b)
c0=intersection(base['top'],a,b);c1=intersection(after['top'],a,b);threshold=pairs.history_cosine.quantile(.25);valid=pairs.history_cosine.to_numpy()<=threshold
# Choose the first pair closest to the population mean shared counts, not a maximal-change case.
metrics=pd.read_csv(H/'tables/catalog_metrics.csv');m=metrics[(metrics.model==model)&(metrics.genre=='Thriller')].set_index('n_fake')
quality=np.abs(c0-m.loc[0,'exact_mean_pair_shared_books'])+np.abs(c1-m.loc[500,'exact_mean_pair_shared_books']);quality[~valid]=np.inf;k=int(np.argmin(quality));u,v=int(a[k]),int(b[k])
book_metadata=pd.read_csv('/home/moshtasa/Research/phd-recsys/Book/Archive/data/genre_clean.csv',usecols=['book_id','original_title','genres']).drop_duplicates('book_id').set_index('book_id')
catalog_titles=book_metadata.original_title.to_dict()
catalog_genres={int(book):[g for g in l.parse_genres(value) if g.lower()!='unknown'] for book,value in book_metadata.genres.items()}
shared0=set(base['top'][u])&set(base['top'][v]);shared1=set(after['top'][u])&set(after['top'][v]);shared=sorted(shared0|shared1)
palette=plt.get_cmap('tab20');colors={book:tuple(np.r_[np.array(palette(j%20)[:3])*.7+.3,1.0]) for j,book in enumerate(shared)}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'savefig.dpi':220,'axes.spines.top':False,'axes.spines.right':False})
fig=plt.figure(figsize=(24,13.5));gs=fig.add_gridspec(1,3,width_ratios=[1.12,1,1],wspace=.25);ax=fig.add_subplot(gs[0]);ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
ax.set_title('A  A real-user example',loc='left',fontweight='bold',fontsize=17,pad=20)
ax.text(0,.965,f'Real {model.upper()} example · users {ids[u]} and {ids[v]}',fontsize=12)
def title_for(book):
 title=str(catalog_titles[int(items[book])])
 # Use the catalog's own bracketed romanization for this multilingual title.
 if '[' in title and any('\u0600'<=ch<='\u06ff' for ch in title):title=re.search(r'\[([^]]+)\]',title).group(1)
 return title
for phase,top,y0,common in [('Baseline',base['top'],.855,shared0),('500 injected users',after['top'],.625,shared1)]:
 ax.text(0,y0+.045,f'{phase}  |  {len(common)} shared titles out of 20',fontsize=13,fontweight='bold')
 ordered=[book for book in top[u] if book in common]
 y=y0
 for book in ordered:
  title=title_for(book);lines=textwrap.wrap(title,width=46,break_long_words=False,break_on_hyphens=False);tags=' · '.join(catalog_genres[int(items[book])]);taglines=textwrap.wrap(tags,width=56,break_long_words=False)
  height=.025*len(lines)+.019*len(taglines)+.011
  ax.add_patch(FancyBboxPatch((0,y-height),.96,height-.005,boxstyle='round,pad=.002,rounding_size=.005',facecolor=colors[book],edgecolor='none'))
  ax.text(.025,y-.008,'\n'.join(lines),fontsize=10.5,va='top',color='#142536',linespacing=1.15)
  ax.text(.025,y-.008-.025*len(lines),'\n'.join(taglines),fontsize=8.5,va='top',color='#34404C',linespacing=1.1)
  y-=height+.004
 ax.text(0,y-.012,f'+ {20-len(common)} non-shared titles in each user’s list',fontsize=10,color='#65717B',va='top')
ax.text(0,.015,'Displayed: shared titles and their catalog genre tags.\nBoth full Top-20 lists are used in the analysis.\nThis is one real example; population results are at right.',fontsize=10,linespacing=1.4,va='top')
catalog=pd.read_csv(H/'tables/catalog_metrics.csv');control=pd.read_csv(H/'tables/cross_model_fixed_target_control.csv');x=np.arange(9);ticks=['Base','2','4','6','25','50','100','200','500']
axes=[fig.add_subplot(gs[1]),fig.add_subplot(gs[2])]
for chart_model,color in [('nmf','#218B58'),('svd','#2563A6')]:
 data=catalog[(catalog.model==chart_model)&(catalog.genre=='Thriller')].sort_values('n_fake')
 test=control[(control.model==chart_model)&(control.genre=='Thriller')&(control.stratum=='different_histories')].sort_values('n_fake')
 for aa,vals in [(axes[0],data.exact_mean_pair_shared_books.to_numpy()),(axes[1],test.mean_target_jaccard_percent.to_numpy())]:
  aa.plot(x,vals,'-o',color=color,lw=3,ms=6,label=chart_model.upper());aa.annotate(f'{vals[0]:.2f}',(0,vals[0]),xytext=(0,12),textcoords='offset points',fontsize=12,color=color);aa.annotate(f'{vals[-1]:.2f}',(8,vals[-1]),xytext=(-5,12),textcoords='offset points',ha='right',fontsize=12,color=color)
for aa in axes:
 aa.set_xticks(x,ticks,rotation=35,ha='right');aa.set_xlabel('Injected users',labelpad=10);aa.grid(axis='y',alpha=.2);aa.set_axisbelow(True);aa.legend(frameon=False,fontsize=13)
axes[0].set_title('B  Full Top-20 lists',loc='left',fontweight='bold',fontsize=17,pad=20);axes[0].set_ylabel('Mean shared book titles across all user pairs');axes[0].set_ylim(0,10)
axes[1].set_title('C  Exactly two Thriller titles per user',loc='left',fontweight='bold',fontsize=17,pad=20);axes[1].set_ylabel('Mean title-set Jaccard (%)');axes[1].set_ylim(0,50)
fig.suptitle('More Target-Genre Recommendations—or More of the Same Books?',fontsize=25,y=.96)
fig.text(.5,.075,'B: exact average over all distinct real-user pairs. C: same 3,815 eligible real users in both models at every dose;\n26,148 sampled pairs with relatively different original genre histories. Only the two highest-ranked Thriller titles in each Top-20 list are compared.\nRecommended-title similarity is an output property, not a measured change in human taste or proof of poorer accuracy. One controlled training seed.',ha='center',fontsize=12,linespacing=1.5)
fig.subplots_adjust(top=.82,bottom=.23,left=.04,right=.98)
out=R/'result/final_results/Thriller_personalization_3panel.png' if model=='nmf' else R/'result/final_results/SVD/Thriller/Thriller_personalization_3panel.png'
out.parent.mkdir(parents=True,exist_ok=True);fig.savefig(out,facecolor='white')
meta={'figure':str(out),'illustrative_model':model.upper(),'illustrative_real_user_ids':[int(ids[u]),int(ids[v])],'history_cosine':float(pairs.history_cosine.iloc[k]),'selection':'first lowest-quartile-history pair minimizing distance to aggregate baseline/500 mean intersections','shared_titles_before':len(shared0),'shared_titles_after':len(shared1),'before_book_ids':[[int(items[t]) for t in base['top'][user]] for user in [u,v]],'after_book_ids':[[int(items[t]) for t in after['top'][user]] for user in [u,v]]}
meta['before_book_titles']=[[str(catalog_titles[book]) for book in row] for row in meta['before_book_ids']]
meta['before_book_genres']=[[catalog_genres[book] for book in row] for row in meta['before_book_ids']]
meta['after_book_genres']=[[catalog_genres[book] for book in row] for row in meta['after_book_ids']]
meta['after_book_titles']=[[str(catalog_titles[book]) for book in row] for row in meta['after_book_ids']]
(H/'reports'/f'{model}_three_panel_example.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta,indent=2))
