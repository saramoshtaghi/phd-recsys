"""Readable reviewer-analysis figures from existing numeric tables, without retraining."""
from pathlib import Path
import subprocess,io,json,hashlib
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
REPO=Path('/home/moshtasa/Research/phd-recsys');L=REPO/'Book/current/PerturbRec/svd_nmf/layer1';OUT=REPO/'First_paper_material/figures/first_reviewer';QA=L/'log/reviewer_redraw';QA.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
COUNTS=[2,4,6,25,50,100,200,500];BLUE='#0072B2';ORANGE='#D55E00';GREEN='#009E73';NAVY='#163248';GRAY='#536471';sources=[]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':15,'axes.labelsize':15,'axes.titlesize':18,'axes.titleweight':'bold','xtick.labelsize':14,'ytick.labelsize':14,'axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False,'axes.edgecolor':'#AEBCC6','axes.labelcolor':NAVY,'text.color':NAVY,'xtick.color':GRAY,'ytick.color':GRAY,'figure.facecolor':'white','savefig.facecolor':'white'})
def load(model,genre,name):
 p=L/'result/reviewer_hypothesis'/model/genre/(name+'.csv')
 if p.exists():raw=p.read_bytes();origin=str(p)
 else:
  relative=str(p.relative_to(REPO));raw=subprocess.check_output(['git','show','HEAD:'+relative],cwd=REPO);origin='git HEAD:'+relative
 sources.append({'source':origin,'sha256':hashlib.sha256(raw).hexdigest()});return pd.read_csv(io.BytesIO(raw))
def getdata(genre):
 d={}
 for model in ['NMF','SVD']:
  a=load(model,genre,'synthetic_score_components');a=a[(a.pair_group=='non_target')&a.n_fake.isin(COUNTS)].sort_values('n_fake')
  b=load(model,genre,'candidate_margins');b=b[b.n_fake.isin(COUNTS)].sort_values('n_fake')
  c=load(model,genre,'recommendation_component_swaps');c=c[(c.k==15)&c.n_fake.isin(COUNTS)].sort_values('n_fake')
  assert all(z.n_fake.tolist()==COUNTS for z in [a,b,c]);assert np.allclose(c.embedding_swap_allocation+c.item_bias_swap_allocation,c.delta_count)
  d[model]=(a,b,c)
 return d
def limits(data):
 specifications=[(0,['negative_interaction_percent'],[0,100]),(0,['mean_interaction','mean_offset','mean_prediction'],[1]),(1,['delta_interaction_margin','delta_bi_margin'],[0]),(2,['embedding_swap_allocation','item_bias_swap_allocation','delta_count'],[0])]
 out=[]
 for j,(part,cols,extra) in enumerate(specifications):
  v=np.r_[np.concatenate([data[m][part][cols].to_numpy().ravel() for m in data]),extra];lo=float(v.min());hi=float(v.max());pad=max((hi-lo)*.16,.15);out.append((-5,105) if j==0 else (lo-pad,hi+pad))
 return out

def draw(ax,metric,data,genre,yl,letter):
 a,b,c=data;x=np.arange(len(COUNTS))
 if metric==0:
  ax.plot(x,a.negative_interaction_percent,color=BLUE,marker='o',lw=3.1,ms=7)
  title='Negative user–book dot products';ylabel='Synthetic rating-1 pairs (%)';ax.set_yticks([0,25,50,75,100])
 elif metric==1:
  for col,label,color,marker in [('mean_interaction','User–book dot product',BLUE,'o'),('mean_offset',r'Bias offset: $\mu+b_u+b_i$',ORANGE,'s'),('mean_prediction','Final predicted rating',GREEN,'^')]:ax.plot(x,a[col],label=label,color=color,marker=marker,lw=3,ms=6)
  ax.axhline(1,color='#46515C',lw=1.5,ls=':',label='Recorded rating = 1');title='Fitting the injected ratings of 1';ylabel='Mean score'
 elif metric==2:
  for col,label,color,marker in [('delta_interaction_margin','Dot-product contribution',BLUE,'o'),('delta_bi_margin','Book-bias contribution',ORANGE,'s')]:ax.plot(x,b[col],label=label,color=color,marker=marker,lw=3,ms=6)
  ax.axhline(0,color='#96A4AE',lw=.9);title=f'Change in {genre} score advantage';ylabel='Score-margin change vs. baseline'
 else:
  for col,label,color,marker in [('embedding_swap_allocation','Embedding component',BLUE,'o'),('item_bias_swap_allocation','Book-bias component',ORANGE,'s'),('delta_count','Total recommendation change',GREEN,'^')]:ax.plot(x,c[col],label=label,color=color,marker=marker,lw=3,ms=6)
  ax.axhline(0,color='#96A4AE',lw=.9);title=f'Contributions to the Top-15 change';ylabel=f'Change in mean {genre} books'
 ax.set_title(f'({letter}) {title}',loc='left',pad=16,fontsize=18);ax.set_ylabel(ylabel,labelpad=10);ax.set_ylim(*yl);ax.set_xlim(-.25,7.25);ax.set_xticks(x,[str(n) for n in COUNTS]);ax.set_xlabel('Injected users',labelpad=10);ax.tick_params(axis='both',length=0,pad=7);ax.grid(axis='y',color='#DEE5EA',lw=.75);ax.set_axisbelow(True)
 if metric:ax.yaxis.set_major_locator(MaxNLocator(nbins=5));ax.legend(loc='best',fontsize=12.5,frameon=True,facecolor='white',edgecolor='#E0E6EB',framealpha=.96,labelspacing=.5,handlelength=2)

def save(fig,name):
 target=OUT/name;fig.savefig(target,dpi=600);fig.savefig(QA/(target.stem+'_preview.png'),dpi=110);plt.close(fig);print('SAVED',target,flush=True)
all_data={}
for genre in ['Thriller','Drama']:
 d=getdata(genre);all_data[genre]=d;yl=limits(d)
 for model in ['NMF','SVD']:
  fig,axes=plt.subplots(2,2,figsize=(16,10.5))
  for j,ax in enumerate(axes.flat):draw(ax,j,d[model],genre,yl[j],chr(97+j))
  fig.suptitle(f'{model} | {genre} injection',x=.075,ha='left',y=.98,fontsize=28,fontweight='bold')
  fig.text(.075,.925,f'Rating-1 pairs: synthetic users × books without the {genre} label',fontsize=15,color=GRAY)
  fig.subplots_adjust(left=.085,right=.975,top=.825,bottom=.11,wspace=.26,hspace=.63)
  fig.text(.085,.025,'Lower panels compare with baseline. Component swaps describe fitted models, not a causal training experiment.',fontsize=12,color=GRAY)
  save(fig,f'{model}_{genre}_reviewer_hypothesis_4panel.png')
# Full comparison: large panels arranged as 4 metric rows × 2 model columns.
genre='Thriller';d=all_data[genre];yl=limits(d);fig,axes=plt.subplots(4,2,figsize=(16,19.5))
for ci,model in enumerate(['NMF','SVD']):
 for j in range(4):draw(axes[j,ci],j,d[model],genre,yl[j],chr(97+j))
 axes[0,ci].text(.5,1.38,model,transform=axes[0,ci].transAxes,fontsize=27,weight='bold',ha='center')
fig.suptitle('Thriller injection | NMF and SVD',x=.075,ha='left',y=.98,fontsize=29,fontweight='bold')
fig.text(.075,.948,'Same metric and linear scale across each row. Synthetic rating-1 pairs exclude Thriller books.',fontsize=14,color=GRAY)
fig.subplots_adjust(left=.085,right=.975,top=.865,bottom=.055,wspace=.27,hspace=.68)
fig.text(.085,.015,'Score and recommendation changes are relative to baseline. Component swaps are descriptive, not causal.',fontsize=12,color=GRAY)
save(fig,'Thriller_NMF_SVD_components_powerpoint.png')
(QA/'validation.json').write_text(json.dumps({'unchanged_source_data':True,'no_retraining':True,'counts':COUNTS,'models':['NMF','SVD'],'genres':['Drama','Thriller'],'same_scales_between_models':True,'component_allocations_sum_to_total_change':True,'source_files':sources,'dpi':600,'four_panel_size_inches':[16,10.5],'combined_size_inches':[16,19.5],'output_count':5},indent=2))
