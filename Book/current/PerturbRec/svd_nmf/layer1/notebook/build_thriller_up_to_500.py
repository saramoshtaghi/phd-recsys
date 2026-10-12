#!/usr/bin/env python3
"""PNG presentation figures for Thriller/Drama from existing controlled tables; no retraining."""
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
ROOT=Path('/home/moshtasa/Research/phd-recsys/Book/current/PerturbRec/svd_nmf/layer1')
OUT=ROOT/'result/final_results'
GENRE='Thriller'
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
 OUT.mkdir(parents=True,exist_ok=True);p=OUT/name
 fig.savefig(p,dpi=240,facecolor='white');plt.close(fig);print('SAVED',p,flush=True)

def bars():
 df=pd.read_csv(ROOT/f'tables/controlled_{GENRE.lower()}_top15.csv')
 df=df[(df.genre==GENRE)&(df.k==15)&df.n_fake.isin(COUNTS)]
 assert len(df)==18 and (df.n_users==53424).all()
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
 fig.subplots_adjust(left=.09,right=.97,bottom=.23,top=.77,wspace=.12)
 fig.text(.055,.94,f'{GENRE} recommendations after synthetic-user injection',fontsize=25,weight='bold',color=NAVY)
 fig.text(.055,.885,'NMF and SVD · Top-15 · Baseline to 500 injected users',fontsize=15,color=MUTED)
 fig.text(.055,.09,'Mean over 53,424 real users; previously rated books excluded. Matching colors represent the same injection size in both models.',fontsize=12.5,color=NAVY)
 fig.text(.055,.045,f'Synthetic profiles: {GENRE} books rated 5, other books rated 1. Existing controlled runs; no retraining.',fontsize=12,color=MUTED)
 save(fig,f'{GENRE}_bars_top15_up_to_500.png')

def bias():
 sets={}
 for model in ['nmf','svd']:
  rows=[]
  for n in COUNTS:
   name='ORIGINAL_bi_per_genre.csv' if n==0 else f'f_{GENRE}_{n}u_pos5_neg1_books_bi_per_genre.csv'
   d=pd.read_csv(ROOT/'result'/model/'bi'/name)
   d=d[(d.n_fake==n)&(d.genre.str.lower()!='unknown')]
   assert not d.genre.duplicated().any() and (d.training_protocol=='controlled_clean_catalog').all()
   rows.append(d)
  sets[model]=pd.concat(rows,ignore_index=True)
 genres=sorted(sets['nmf'].genre.unique());assert len(genres)==13 and set(genres)==set(sets['svd'].genre)
 others=[g for g in genres if g!=GENRE];assert set(genres)==set(GENRE_COLORS)
 fig,axes=plt.subplots(1,2,figsize=(16,9),sharey=True)
 allvalues=pd.concat(list(sets.values())).avg_bi
 lo=float(allvalues.min());hi=float(allvalues.max());pad=.12*(hi-lo)
 for ax,model in zip(axes,['nmf','svd']):
  data=sets[model]
  for j,genre in enumerate(others):
   d=data[data.genre==genre].sort_values('n_fake');assert d.n_fake.tolist()==COUNTS
   ax.plot(range(len(COUNTS)),d.avg_bi,marker=MARKERS[j],ls='-' if j%2==0 else '--',lw=2.1,ms=5.3,color=GENRE_COLORS[genre],alpha=1,label=genre)
  d=data[data.genre==GENRE].sort_values('n_fake');assert d.n_fake.tolist()==COUNTS
  ax.plot(range(len(COUNTS)),d.avg_bi,'D-',lw=3.8,ms=7,color=TARGET,label=f'{GENRE} (target)',zorder=10)
  ax.annotate(f'{d.avg_bi.iloc[-1]:.2f}',(8,d.avg_bi.iloc[-1]),xytext=(5,6),textcoords='offset points',color=TARGET,weight='bold',fontsize=13)
  ax.axhline(0,color='#738496',lw=.8,ls='--');style(ax)
  ax.set_ylim(lo-pad,hi+pad);ax.set_title(model.upper(),fontsize=24,weight='bold',color=NAVY,pad=22)
 axes[0].set_ylabel(r'Mean learned book bias ($b_i$)',fontsize=17,color=NAVY,labelpad=15)
 fig.subplots_adjust(left=.09,right=.97,bottom=.34,top=.77,wspace=.12)
 fig.text(.055,.94,f'Learned book biases under {GENRE} injection',fontsize=25,weight='bold',color=NAVY)
 fig.text(.055,.885,'NMF and SVD · Baseline to 500 injected users · Shared linear scale',fontsize=15,color=MUTED)
 handles,labels=axes[0].get_legend_handles_labels();handles=[handles[-1]]+handles[:-1];labels=[labels[-1]]+labels[:-1]
 fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.51,.035),ncol=5,frameon=False,fontsize=11.5,columnspacing=1.3,handlelength=1.8)
 fig.text(.055,.185,'Each curve averages book biases for one genre; multi-genre books can contribute to multiple curves.',fontsize=12,color=MUTED)
 save(fig,f'{GENRE}_bi_only_up_to_500.png')

if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('--genre',choices=['Thriller','Drama','both'],default='both');args=parser.parse_args()
 for GENRE in (['Thriller','Drama'] if args.genre=='both' else [args.genre]):bars();bias()
