#!/usr/bin/env python3
"""Render Drama/Thriller score-component panels for both models, using existing tables."""
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path('/home/moshtasa/Research/phd-recsys/Book/current/PerturbRec/svd_nmf/layer1')
SRC=ROOT/'result/reviewer_hypothesis';OUT=ROOT.parent/'layer2/final_results'
GENRE='Drama'
COUNTS=[2,4,6,25,50,100,200,500]
COLORS={'interaction':'#0072B2','bias':'#D55E00','prediction':'#009E73'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.spines.top':False,'axes.spines.right':False})

def main():
 data={}
 for model in ('NMF','SVD'):
  p=SRC/model/GENRE
  fake=pd.read_csv(p/'synthetic_score_components.csv');fake=fake[(fake.pair_group=='non_target')&fake.n_fake.isin(COUNTS)].sort_values('n_fake')
  margin=pd.read_csv(p/'candidate_margins.csv');margin=margin[margin.n_fake.isin(COUNTS)].sort_values('n_fake')
  rec=pd.read_csv(p/'recommendation_component_swaps.csv');rec=rec[(rec.k==15)&rec.n_fake.isin(COUNTS)].sort_values('n_fake')
  assert all(d.n_fake.tolist()==COUNTS for d in (fake,margin,rec))
  data[model]=(fake,margin,rec)
 def limits(columns,part,include=()):
  v=np.concatenate([data[m][part][columns].to_numpy().ravel() for m in data]+[np.array(include,dtype=float)])
  lo,hi=float(v.min()),float(v.max());pad=max(.15*(hi-lo),.1);return lo-pad,hi+pad
 scorelim=limits(['mean_interaction','mean_offset','mean_prediction'],0,include=[1])
 marginlim=limits(['delta_interaction_margin','delta_bi_margin'],1,include=[0])
 reclim=limits(['embedding_swap_allocation','item_bias_swap_allocation','delta_count'],2,include=[0])
 x=np.arange(len(COUNTS));OUT.mkdir(parents=True,exist_ok=True)
 for model,(fake,margin,rec) in data.items():
  fig,axes=plt.subplots(2,2,figsize=(16,9))
  ax=axes[0,0];ax.plot(x,fake.negative_interaction_percent,'o-',color=COLORS['interaction'],lw=2.5)
  ax.set_title('Non-target books rated 1 by synthetic users',fontsize=15,pad=10)
  ax.set_ylabel('Pairs with negative interaction (%)');ax.set_ylim(-3,103)
  ax=axes[0,1]
  for col,label,color in [('mean_interaction','User–book interaction',COLORS['interaction']),('mean_offset','Global mean + user bias + item bias',COLORS['bias']),('mean_prediction','Full prediction',COLORS['prediction'])]:ax.plot(x,fake[col],'o-',lw=2.4,label=label,color=color)
  ax.axhline(1,color='#404040',ls=':',lw=1.5,label='Observed rating = 1')
  ax.set_title('How the model fits those rating-1 pairs',fontsize=15,pad=10);ax.set_ylabel('Mean raw score contribution');ax.set_ylim(*scorelim);ax.legend(fontsize=10,loc='best',framealpha=.95)
  ax=axes[1,0]
  for col,label,color in [('delta_interaction_margin','Interaction',COLORS['interaction']),('delta_bi_margin','Item bias',COLORS['bias'])]:ax.plot(x,margin[col],'o-',lw=2.4,label=label,color=color)
  ax.axhline(0,color='#737D87',lw=.8);ax.set_title('Target-versus-other candidate score margin',fontsize=15,pad=10)
  ax.set_ylabel('Change from baseline');ax.set_ylim(*marginlim);ax.legend(fontsize=11)
  ax=axes[1,1]
  for col,label,color in [('embedding_swap_allocation','Embedding component',COLORS['interaction']),('item_bias_swap_allocation','Item-bias component',COLORS['bias']),('delta_count','Total exposure change',COLORS['prediction'])]:ax.plot(x,rec[col],'o-',lw=2.4,label=label,color=color)
  ax.axhline(0,color='#737D87',lw=.8);ax.set_title('Top-15: conditional component allocation',fontsize=15,pad=10)
  ax.set_ylabel(f'Change in mean {GENRE}-book count');ax.set_ylim(*reclim);ax.legend(fontsize=10,loc='upper left')
  for ax in axes.flat:
   ax.set_xticks(x,[str(n) for n in COUNTS],rotation=35,ha='right',fontsize=11)
   ax.set_xlabel('Number of injected users',fontsize=12)
   ax.grid(axis='y',color='#E0E5EA',lw=.8);ax.set_axisbelow(True)
  fig.suptitle(f'{model} · {GENRE} · Score components and recommendation changes',fontsize=23,weight='bold',color='#142E49',y=.97)
  fig.text(.06,.916,'2–500 injected users · Existing controlled runs · Linear scales matched between models',fontsize=13,color='#50657A')
  fig.subplots_adjust(left=.075,right=.98,bottom=.19,top=.82,hspace=.65,wspace=.27)
  fig.text(.06,.065,'The lower panels compare each injection with baseline; component allocation uses fitted-model swaps.',fontsize=11,color='#50657A')
  fig.text(.06,.028,'These swaps describe score-component contributions and do not establish a causal training mechanism.',fontsize=11,color='#50657A')
  path=OUT/f'{model}_{GENRE}_reviewer_hypothesis_4panel.png';fig.savefig(path,dpi=240,facecolor='white');plt.close(fig);print('SAVED',path,flush=True)
if __name__=='__main__':
 for GENRE in ('Drama','Thriller'):main()
