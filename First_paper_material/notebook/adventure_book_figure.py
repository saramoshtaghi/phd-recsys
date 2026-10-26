#!/usr/bin/env python3
"""Show actual Adventure book titles and original popularity for final repeated items."""
from pathlib import Path
import time,json,textwrap
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/home/moshtasa/Research/phd-recsys');NB=R/'First_paper_material/notebook';L=R/'Book/current/PerturbRec/svd_nmf/layer2';F=L/'final_results'
if __name__=='__main__':
 for m in ['svd','nmf']:
  while not (NB/f'adventure_{m}_3500_analysis.json').exists():time.sleep(15)
 while json.loads((NB/'adventure_analysis_status.json').read_text()).get('status')!='complete':time.sleep(15)
 df=pd.read_csv(R/'Book/Archive/data/genre_clean.csv',usecols=['book_id','genres','original_title','rating']);meta=df.drop_duplicates('book_id').set_index('book_id');hist=df.groupby('book_id').rating.agg(['size','mean'])
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
 fig,axs=plt.subplots(2,1,figsize=(7.1,5.1));rows=[]
 for ax,m in zip(axs,['svd','nmf']):
  final=np.load(L/'embeddings/controlled'/m/f'{m}_adventure_3500_analysis.npz');base=np.load(L/'embeddings/controlled'/m/f'{m}_adventure_0_analysis.npz');items=final['items'];target=np.array(['Adventure' in [x.strip() for x in str(meta.loc[int(i),'genres']).split(',')] for i in items]);f=np.bincount(final['top35'][:,:15].ravel(),minlength=len(items));b=np.bincount(base['top35'][:,:15].ravel(),minlength=len(items));choices=np.flatnonzero(target);chosen=choices[np.argsort(-f[choices],kind='stable')[:6]][::-1];y=np.arange(len(chosen));labels=[]
  for j in chosen:
   book=int(items[j]);title=str(meta.loc[book,'original_title']);genres=str(meta.loc[book,'genres']);labels.append('\n'.join(textwrap.wrap(title,width=45))+'\n['+genres+']');rows.append(dict(model=m,book_id=book,title=title,genres=genres,baseline_reach_percent=100*int(b[j])/53424,reach_percent_3500=100*int(f[j])/53424,original_rating_count=int(hist.loc[book,'size']),original_mean_rating=float(hist.loc[book,'mean'])))
  ax.barh(y-.17,100*b[chosen]/53424,height=.3,color='#AAB0B6',label='Baseline');ax.barh(y+.17,100*f[chosen]/53424,height=.3,color='#D55E00' if m=='svd' else '#0072B2',label='3,500 injected users');ax.set_yticks(y,labels,fontsize=6.5);ax.set_xlabel('Original users receiving this book in Top-15 (%)',fontsize=7);ax.set_title(m.upper()+' | Most frequently recommended Adventure books at 3,500',fontsize=8);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True);ax.legend(frameon=False,fontsize=6.5,ncol=2,loc='lower right')
 fig.tight_layout(h_pad=1.8)
 bad=json.loads((NB/'adventure_analysis_status.json').read_text()).get('nonnegative_constraint_failures',[])
 if bad:
  fig.canvas.draw();box=fig.get_tightbbox(fig.canvas.get_renderer());fig.text(.5,box.y0/fig.get_size_inches()[1]-.03,'NMF factor sign failures: '+', '.join(str(x['n_fake']) for x in bad)+'. Raw outputs retained.',ha='center',va='top',fontsize=6,color='#9D211C')
 for ext in ['png','pdf']:fig.savefig(F/f'Adventure_most_repeated_books_with_genres.{ext}',dpi=400,bbox_inches='tight',facecolor='white')
 plt.close(fig);pd.DataFrame(rows).to_csv(NB/'adventure_repeated_books_original_popularity.csv',index=False);(NB/'adventure_book_figure_status.json').write_text(json.dumps(dict(status='complete',books_per_model=6,n_fake=3500),indent=2));print('Adventure book title figure complete',flush=True)
