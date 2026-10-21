"""Separate fitting-of-rating-5 figure from existing synthetic target-pair diagnostics."""
from pathlib import Path
import io,json,hashlib,subprocess
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
REPO=Path('/home/moshtasa/Research/phd-recsys');L=REPO/'Book/current/PerturbRec/svd_nmf/layer1';OUT=REPO/'First_paper_material/figures/first_reviewer';QA=L/'log/reviewer_redraw';QA.mkdir(parents=True,exist_ok=True)
COUNTS=[2,4,6,25,50,100,200,500];BLUE='#0072B2';ORANGE='#D55E00';GREEN='#009E73';NAVY='#163248';GRAY='#536471';sources=[];allrows=[]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':16,'axes.titlesize':21,'axes.titleweight':'bold','axes.labelsize':16,'xtick.labelsize':15,'ytick.labelsize':15,'axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False,'axes.edgecolor':'#AEBCC6','text.color':NAVY,'axes.labelcolor':NAVY,'xtick.color':GRAY,'ytick.color':GRAY,'figure.facecolor':'white','savefig.facecolor':'white'})
fig,axes=plt.subplots(2,2,figsize=(16,10.5),sharey=True)
for ri,g in enumerate(['Thriller','Drama']):
 for ci,m in enumerate(['NMF','SVD']):
  source=L/'result/reviewer_hypothesis'/m/g/'synthetic_score_components.csv'
  if source.exists():raw=source.read_bytes();origin=str(source)
  else:relative=str(source.relative_to(REPO));raw=subprocess.check_output(['git','show','HEAD:'+relative],cwd=REPO);origin='git HEAD:'+relative
  t=pd.read_csv(io.BytesIO(raw));d=t[(t.pair_group=='target')&t.n_fake.isin(COUNTS)].sort_values('n_fake').copy()
  assert d.n_fake.tolist()==COUNTS and d.rating.eq(5).all()
  assert np.isfinite(d[['mean_interaction','mean_offset','mean_prediction']]).all().all()
  assert np.allclose(d.mean_interaction+d.mean_offset,d.mean_prediction,atol=1e-10)
  assert np.allclose(d.mu+d.mean_bu+d.mean_bi,d.mean_offset,atol=1e-10)
  sources.append({'model':m,'genre':g,'source':origin,'sha256':hashlib.sha256(raw).hexdigest()});allrows.append(d)
  ax=axes[ri,ci];x=np.arange(len(COUNTS))
  for key,color,marker in [('mean_interaction',BLUE,'o'),('mean_offset',ORANGE,'s'),('mean_prediction',GREEN,'^')]:ax.plot(x,d[key],color=color,marker=marker,lw=3.1,ms=7,zorder=3)
  ax.axhline(5,color='#46515C',lw=1.7,ls=':',zorder=2)
  ax.annotate(f'{d.mean_prediction.iloc[-1]:.2f}',(7,d.mean_prediction.iloc[-1]),xytext=(0,12),textcoords='offset points',ha='center',va='bottom',fontsize=14,color=GREEN,fontweight='bold')
  ax.set_title(f'({chr(97+ri*2+ci)}) {m} | {g}',loc='left',pad=18)
  ax.set_ylim(-.25,5.85);ax.set_yticks(range(6));ax.set_xlim(-.25,7.25);ax.set_xticks(x,[str(n) for n in COUNTS]);ax.set_xlabel('Injected users',labelpad=12);ax.set_ylabel('Mean raw score contribution',labelpad=12)
  ax.tick_params(axis='both',length=0,pad=8,labelleft=True);ax.grid(axis='y',color='#DEE5EA',lw=.75);ax.set_axisbelow(True)
fig.suptitle('Fitting the injected ratings of 5',x=.075,ha='left',y=.98,fontsize=29,fontweight='bold')
handles=[Line2D([0],[0],color=BLUE,marker='o',lw=3,label='User–book dot product'),Line2D([0],[0],color=ORANGE,marker='s',lw=3,label=r'Bias offset: $\mu+b_u+b_i$'),Line2D([0],[0],color=GREEN,marker='^',lw=3,label='Final predicted rating'),Line2D([0],[0],color='#46515C',lw=1.7,ls=':',label='Recorded rating = 5')]
fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.52,.925),ncol=4,frameon=False,fontsize=13,columnspacing=1.7,handlelength=2)
fig.subplots_adjust(left=.085,right=.975,top=.79,bottom=.095,wspace=.27,hspace=.58)
name='injected_rating5_fit_NMF_SVD_Drama_Thriller.png';fig.savefig(OUT/name,dpi=600);fig.savefig(QA/'injected_rating5_fit_preview.png',dpi=110);plt.close(fig)
pd.concat(allrows,ignore_index=True).to_csv(QA/'injected_rating5_components.csv',index=False)
(QA/'injected_rating5_validation.json').write_text(json.dumps({'sources':sources,'rating':5,'pair_group':'target','counts':COUNTS,'models':['NMF','SVD'],'genres':['Thriller','Drama'],'n_panels':4,'mean_prediction_equals_mean_offset_plus_mean_dot_product':True,'mean_offset_equals_mu_plus_mean_bu_plus_mean_bi':True,'data_unchanged':True,'existing_figures_unchanged':True,'no_retraining':True,'dpi':600,'image_size_pixels':[9600,6300],'output':str(OUT/name)},indent=2))
print('SAVED',OUT/name,flush=True)
