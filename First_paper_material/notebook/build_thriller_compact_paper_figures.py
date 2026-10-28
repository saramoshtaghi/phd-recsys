"""Compact five-panel Thriller figures from the audited source tables."""
from pathlib import Path
import os,json,hashlib,sys
os.environ['MPLBACKEND']='Agg'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from PIL import Image
ROOT=Path('/home/moshtasa/Research/phd-recsys')
NB=ROOT/'First_paper_material/notebook'
OUT=ROOT/'First_paper_material/figures/first_reviewer/thriller/paper_figures'
OUT.mkdir(exist_ok=True)
DATA=NB/'three_genre_panel_independent_components.csv'
REC=ROOT/'Book/current/PerturbRec/svd_nmf/layer1/tables/topk_target_union/summary.csv'
source=pd.read_csv(DATA);recs=pd.read_csv(REC)
COLORS={'svd':'#0072B2','nmf':'#D55E00'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7.6,'axes.titlesize':8.3,'axes.titleweight':'bold','axes.labelsize':7.3,'xtick.labelsize':6.8,'ytick.labelsize':7,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.55,'text.color':'#182E3C','axes.labelcolor':'#263F4D','xtick.color':'#334755','ytick.color':'#334755','figure.facecolor':'white','savefig.facecolor':'white','pdf.fonttype':42})
validation=[]
for view,name in [
 ('all_training','training_thriller_all_users_5panel_paper.png'),
 ('synthetic_training','training_thriller_synthetic_only_5panel_paper.png'),
 ('real_unseen_prediction','prediction_real_users_thriller_5panel_paper.png')]:
 if '--prediction-only' in sys.argv and view!='real_unseen_prediction':continue
 ns=([2,4,6,25,50,100,200,500] if view=='synthetic_training' else [0,2,4,6,25,50,100,200,500])
 x=np.arange(len(ns));f=source[(source.genre=='Thriller')&(source.view==view)]
 fig=plt.figure(figsize=(7.2,3.75))
 grid=fig.add_gridspec(2,6)
 axs=[fig.add_subplot(grid[0,0:2]),fig.add_subplot(grid[0,2:4]),fig.add_subplot(grid[0,4:6]),fig.add_subplot(grid[1,1:3]),fig.add_subplot(grid[1,3:5])]
 panels=[('mae','(A) Training MAE','Mean absolute error'),('mean_bi',r'(B) Book bias $b_i$','Mean component'),('mean_interaction',r'(C) Dot product $p_u^\top q_i$','Mean component'),('mean_bu',r'(D) User bias $b_u$','Mean component'),('mean_prediction',r'(E) Prediction $\hat r_{ui}$','Mean predicted rating')]
 rows=[]
 for pi,(ax,(key,title,ylabel)) in enumerate(zip(axs,panels)):
  if view=='real_unseen_prediction' and pi==0:
   key='mean_prediction'
   for model in ['svd','nmf']:
    for group,style,marker in [('target','-','o'),('non_target','--','s')]:
     d=f[(f.model==model)&(f.group==group)].set_index('n_fake').loc[ns]
     assert len(d)==len(ns) and np.isfinite(d[key]).all()
     assert np.allclose(d.mean_prediction,d.mu+d.mean_bu+d.mean_bi+d.mean_interaction,atol=1e-10)
     vals=d[key].to_numpy()
     ax.plot(x,vals,color=COLORS[model],ls=style,marker=marker,ms=2.3,lw=1.65,zorder=4)
     rows.append(dict(panel='A',model=model,group=group,quantity=key,values=vals.tolist()))
   title=r'(A) Mean predicted rating';ylabel='Mean predicted rating'
   ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
  else:
   for model in ['svd','nmf']:
    for group,style,marker in [('target','-','o'),('non_target','--','s')]:
     d=f[(f.model==model)&(f.group==group)].set_index('n_fake').loc[ns]
     assert len(d)==len(ns) and np.isfinite(d[key]).all()
     assert np.allclose(d.mean_prediction,d.mu+d.mean_bu+d.mean_bi+d.mean_interaction,atol=1e-10)
     vals=d[key].to_numpy();ax.plot(x,vals,color=COLORS[model],ls=style,marker=marker,ms=2.3,lw=1.65 if key=='mean_prediction' else 1.25,zorder=4)
     rows.append(dict(panel=chr(65+pi),model=model,group=group,quantity=key,values=vals.tolist()))
   if key=='mean_prediction' and view!='real_unseen_prediction':
    for group,style in [('target','-'),('non_target','--')]:
     d=f[(f.model=='svd')&(f.group==group)].set_index('n_fake').loc[ns]
     other=f[(f.model=='nmf')&(f.group==group)].set_index('n_fake').loc[ns]
     assert np.allclose(d.mean_recorded_rating,other.mean_recorded_rating,atol=1e-12)
     vals=d.mean_recorded_rating.to_numpy();ax.plot(x,vals,color='#777777',ls=style,lw=.9,zorder=2)
     rows.append(dict(panel='E',model='observed',group=group,quantity='mean_recorded_rating',values=vals.tolist()))
   if key in ['mean_bi','mean_interaction','mean_bu']:ax.axhline(0,color='#9DAAB3',lw=.5,zorder=1)
   if key=='mae':ax.set_ylim(bottom=0)
   ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
  ax.set_title(title,loc='left',pad=4)
  ax.set_ylabel(ylabel,labelpad=3)
  ax.set_xlabel('Injected users',labelpad=3)
  ax.set_xticks(x,[str(n) for n in ns]);ax.tick_params(length=2.5,width=.5,pad=2)
  ax.set_axisbelow(True);ax.grid(axis='y',color='#E1E7EB',lw=.45,zorder=0);ax.margins(x=.04)
 handles=[Line2D([0],[0],color=COLORS['svd'],lw=1.6,label='SVD'),Line2D([0],[0],color=COLORS['nmf'],lw=1.6,label='NMF'),Line2D([0],[0],color='#536471',lw=1.1,marker='o',ms=2.5,label='Thriller'),Line2D([0],[0],color='#536471',lw=1.1,ls='--',marker='s',ms=2.5,label='Non-Thriller')]
 if view!='real_unseen_prediction':handles.append(Line2D([0],[0],color='#777777',lw=.9,label='Recorded rating (E)'))
 fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.995),ncol=len(handles),frameon=False,fontsize=7.5,columnspacing=1.3,handlelength=2.0,handletextpad=.5)
 fig.subplots_adjust(left=.078,right=.99,bottom=.105,top=.85,hspace=.95,wspace=1.05)
 # No figure title or explanatory footer: manuscript captions carry context.
 fig.canvas.draw();renderer=fig.canvas.get_renderer()
 for ax in axs:
  box=ax.get_tightbbox(renderer)
  assert box.x0>=-1 and box.y0>=-1 and box.x1<=fig.bbox.width+1 and box.y1<=fig.bbox.height+1,(name,box.bounds)
 path=OUT/name;fig.savefig(path,dpi=600);plt.close(fig)
 im=Image.open(path);assert im.size==(4320,2250)
 validation.append(dict(file=str(path),view=view,pixel_size=im.size,figure_inches=[7.2,3.75],panels=5,n_fake=ns,main_title=False,footer_caption=False,plotted_series=rows,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
 print(path,flush=True)
validation_path=NB/'thriller_compact_paper_figures_validation.json'
if '--prediction-only' in sys.argv and validation_path.exists():
 previous=json.loads(validation_path.read_text())['figures']
 validation=[v for v in previous if v['view']!='real_unseen_prediction']+validation
validation_path.write_text(json.dumps(dict(source_table=str(DATA),source_sha256=hashlib.sha256(DATA.read_bytes()).hexdigest(),recommendation_table=str(REC),figures=validation),indent=2))
