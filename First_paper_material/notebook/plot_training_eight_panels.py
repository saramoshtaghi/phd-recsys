"""Paper-size eight-panel score decomposition, from existing fitted checkpoints."""
from pathlib import Path
import hashlib,json
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
R=Path('/home/moshtasa/Research/phd-recsys/First_paper_material');D=R/'notebook';O=R/'figures/first_reviewer'
assert D.is_dir() and O.is_dir()
p=D/'training_score_components.csv';ep=D/'training_all_users_mae.csv';f=pd.read_csv(p);err=pd.read_csv(ep)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7.5,'axes.titlesize':8.1,'axes.titleweight':'bold','axes.labelsize':7.2,'xtick.labelsize':6.5,'ytick.labelsize':6.5,'axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False,'axes.linewidth':.5,'axes.edgecolor':'#A8B1B9','text.color':'#163248','axes.labelcolor':'#344755','xtick.color':'#536471','ytick.color':'#536471','figure.facecolor':'white','savefig.facecolor':'white','pdf.fonttype':42})
colors={'mean_prediction':'#009E73','mean_bi':'#D55E00','mean_bu':'#61B8E8','mean_interaction':'#0072B2','mu':'#D7AA00','mae':'#D62728'}
labels={'mean_prediction':r'Prediction $\hat r_{ui}$','mean_bi':r'Book bias $b_i$','mean_bu':r'User bias $b_u$','mean_interaction':r'Dot product $p_u^\top q_i$','mu':r'Global mean $\mu$','mae':'MAE'}
fig=plt.figure(figsize=(7.1,8.6))
outer=fig.add_gridspec(2,1,left=.085,right=.985,bottom=.065,top=.83,hspace=.30)
audit=[]
for section,view in enumerate(['all_training','synthetic_training']):
 grid=outer[section].subgridspec(2,2,hspace=.39,wspace=.19)
 for row,group in enumerate(['target','non_target']):
  axs=[]
  ds=[]
  for col,model in enumerate(['svd','nmf']):
   ax=fig.add_subplot(grid[row,col],sharey=axs[0] if axs else None);axs.append(ax)
   d=f[(f.view==view)&(f.group==group)&(f.model==model)].sort_values('n_fake')
   e=err[(err.group==group)&(err.model==model)].sort_values('n_fake')
   if view=='synthetic_training':e=e[e.n_fake>0]
   assert d.n_fake.tolist()==e.n_fake.tolist()==([0] if view=='all_training' else [])+[2,4,6,25,50,100,200,500]
   assert np.allclose(d.mean_prediction,d.mu+d.mean_bu+d.mean_bi+d.mean_interaction,atol=1e-10)
   x=np.arange(len(d));mae=e.all_mae if view=='all_training' else e.synthetic_mae
   assert np.isfinite(mae).all() and (mae>=0).all()
   for key in ['mu','mean_bi','mean_bu','mean_interaction','mean_prediction']:
    ax.plot(x,d[key],color=colors[key],lw=1.65 if key=='mean_prediction' else .95,ls='--' if key=='mu' else '-',marker='o' if key=='mean_prediction' else None,ms=2,zorder=5 if key=='mean_prediction' else 3)
   ax.plot(x,mae,color=colors['mae'],lw=1.05,marker='s',ms=1.9,zorder=6)
   ax.axhline(0,color='#A8B1B9',lw=.45,zorder=1);ax.grid(axis='y',color='#DEE5EA',lw=.4);ax.set_axisbelow(True)
   ax.set_xticks(x,['Base' if n==0 else str(n) for n in d.n_fake]);ax.tick_params(length=0,pad=2.5)
   ax.set_xlim(-.25,len(d)-.75);ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
   panel=chr(97+section*4+row*2+col)
   ax.set_title(f'({panel}) {model.upper()} | '+('Thriller' if group=='target' else 'Non-Thriller'),loc='left',pad=5)
   if col==0:ax.set_ylabel('Mean score / MAE',labelpad=3)
   ax.set_xlabel('Injected users',labelpad=2)
   ds.append(np.r_[d[['mean_prediction','mean_bi','mean_bu','mean_interaction','mu']].to_numpy().ravel(),mae])
   audit.append(dict(panel=panel,view=view,group=group,model=model,n_points=len(d)))
  values=np.concatenate(ds);span=values.max()-values.min();pad=max(span*.08,.15);axs[0].set_ylim(values.min()-pad,values.max()+pad)
 # Section labels above the corresponding 2x2 block.
 box=outer[section].get_position(fig)
 fig.text(.085,box.y1+.035,'All users (real + synthetic)' if section==0 else 'Synthetic users only',ha='left',va='bottom',fontsize=9.3,fontweight='bold')
fig.text(.5,.975,r'$\hat r_{ui}=\mu+b_u+b_i+p_u^\top q_i$',ha='center',va='top',fontsize=11)
handles=[Line2D([0],[0],color=colors[k],lw=1.65 if k=='mean_prediction' else 1,ls='--' if k=='mu' else '-',label=labels[k]) for k in colors]
fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.53,.944),ncol=3,frameon=False,fontsize=7.3,columnspacing=1.7,handlelength=2.2,labelspacing=.55)
name='training_thriller_all_and_synthetic_8panel_paper.png'
fig.savefig(O/name,dpi=600)
# Existing figure folder also holds the inspection preview; no new directories.
fig.savefig(Path('/tmp/training_thriller_all_and_synthetic_8panel_preview.png'),dpi=170)
plt.close(fig)
(D/'training_eight_panel_validation.json').write_text(json.dumps({'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest(),ep.name:hashlib.sha256(ep.read_bytes()).hexdigest()},'panels':audit,'n_panels':8,'prediction_identity_checked':True,'mae_finite_nonnegative':True,'no_retraining':True,'size_inches':[7.1,8.6],'dpi':600,'output':str(O/name)},indent=2))
print(O/name)
