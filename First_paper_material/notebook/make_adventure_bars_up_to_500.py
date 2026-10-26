#!/usr/bin/env python3
"""Adventure Top-15 through 500: exact rendering style of the Thriller reference; no retraining."""
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
ROOT=Path('/home/moshtasa/Research/phd-recsys/Book/current/PerturbRec/svd_nmf/layer1')
OUT=ROOT/'result/final_results'
GENRE='Adventure'
COUNTS=[0,2,4,6,25,50,100,200,500]
TICKS=['Baseline']+list(map(str,COUNTS[1:]))
NAVY='#142E49';MUTED='#50657A';TARGET='#C62828'
DOSE_COLORS=['#586979','#0072B2','#E69F00','#009E73','#7B4AB5','#D55E00','#009DBA','#C75B9B','#233B75']
GENRE_COLORS={'Adult':'#0072B2','Adventure':'#E69F00',"Children's":'#009E73','Classics':'#6F4C9B','Drama':'#A65E2E','Fantasy':'#00A6A6','Historical':'#CC79A7','Horror':'#5C7D14','Mystery':'#303030','Nonfiction':'#D55E00','Science Fiction':'#3A60CB','Romance':'#B27A00','Thriller':'#006D77'}
MARKERS=['o','s','^','D','v','P','X','<','>','h','p','*']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':15,'axes.spines.top':False,'axes.spines.right':False})

def style(ax):
 ax.set_axisbelow(True);ax.grid(axis='y',color='#DFE5EB',lw=.8)
 ax.spines['left'].set_color('#BAC6D2');ax.spines['bottom'].set_color('#BAC6D2')
 ax.tick_params(axis='both',colors=MUTED,length=0)
 ax.set_xticks(range(len(COUNTS)),TICKS,rotation=40,ha='right',fontsize=12)
 ax.set_xlabel('Number of injected users',fontsize=15,color=NAVY,labelpad=12)

def save(fig,name):
 folder=ROOT.parent/'layer2/final_results' if '_bi_only_' in name else OUT
 assert folder.is_dir();p=folder/name
 fig.savefig(p,dpi=240,facecolor='white');plt.close(fig);print('SAVED',p,flush=True)

def bars():
 df=pd.read_csv(ROOT/'tables/adventure_recommendation_summary.csv')
 df=df[(df.genre==GENRE)&(df.k==15)&df.n_fake.isin(COUNTS)]
 assert len(df)==18 and (df.n_users==53424).all()
 assert df.loc[df.model=='nmf','nmf_nonnegative_constraint_ok'].all()
 fig,axes=plt.subplots(1,2,figsize=(16,9),sharey=True)
 for ax,model in zip(axes,['nmf','svd']):
  d=df[df.model==model].sort_values('n_fake');assert d.n_fake.tolist()==COUNTS
  assert d.mean_target_count.between(0,15).all()
  vals=d.mean_target_count.to_numpy()
  ax.bar(range(len(COUNTS)),vals,width=.76,color=DOSE_COLORS,edgecolor='white',linewidth=1.2,zorder=3)
  for x,y in enumerate(vals):ax.text(x,y+.20,f'{y:.2f}',ha='center',fontsize=12,weight='bold',color=NAVY)
  style(ax);ax.set_ylim(0,15);ax.yaxis.set_major_locator(MultipleLocator(3))
  ax.set_title(model.upper(),fontsize=24,weight='bold',color=NAVY,pad=22)
 axes[0].set_ylabel(f'Mean {GENRE} books in Top-15',fontsize=17,color=NAVY,labelpad=15)
 fig.subplots_adjust(left=.09,right=.97,bottom=.23,top=.70,wspace=.12)
 fig.text(.055,.94,f'{GENRE} recommendations after synthetic-user injection',fontsize=25,weight='bold',color=NAVY)
 fig.text(.055,.885,'NMF and SVD · Top-15 · Baseline to 500 injected users',fontsize=15,color=MUTED)
 fig.text(.055,.825,'SVD can lower scores through negative user–book interactions; NMF with nonnegative factors cannot.',fontsize=13.5,color=NAVY)
 fig.text(.055,.790,'Signed bias offsets can lower NMF scores, while changes in positive interactions can also contribute.',fontsize=13.5,color=MUTED)
 fig.text(.055,.09,'Mean over 53,424 real users; previously rated books excluded. Matching colors represent the same injection size in both models.',fontsize=12.5,color=NAVY)
 fig.text(.055,.045,f'Synthetic profiles: {GENRE} books rated 5, other books rated 1. Existing controlled runs; no retraining.',fontsize=12,color=MUTED)
 save(fig,f'{GENRE}_bars_top15_up_to_500.png')

if __name__=="__main__":bars()
