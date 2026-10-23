from pathlib import Path
import json,io,subprocess,hashlib,sys
import pandas as pd,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
R=Path('/home/moshtasa/Research/phd-recsys');D=R/'First_paper_material/analysis/score_components';O=R/'First_paper_material/figures/first_reviewer';O.mkdir(exist_ok=True)
f=pd.read_csv(D/'components.csv')
cols=['mean_prediction','mean_bi','mean_bu','mean_interaction','mu']
rows=[]
for (model,n,group),d in f[f.view.isin(['real_observed_training','synthetic_training'])].groupby(['model','n_fake','group']):
 assert len(d)==(1 if n==0 else 2)
 weights=d.n_pairs.to_numpy(float)
 vals={key:float(np.average(d[key],weights=weights)) for key in cols}
 assert np.isclose(vals['mean_prediction'],vals['mu']+vals['mean_bu']+vals['mean_bi']+vals['mean_interaction'],atol=1e-10)
 rows.append(dict(model=model,n_fake=n,group=group,view='all_training',n_pairs=int(weights.sum()),weighting='equal observed rating pair, real plus synthetic',**vals))
f=pd.concat([f,pd.DataFrame(rows)],ignore_index=True)

# Exact absolute errors from existing checkpoint diagnostics, not errors of group means.
mae_rows=[];provenance=[]
for model in ['svd','nmf']:
 tables={}
 for filename in ['real_observed_score_components','synthetic_score_components']:
  relative=f'Book/current/PerturbRec/svd_nmf/layer1/result/reviewer_hypothesis/{model.upper()}/Thriller/{filename}.csv'
  path=R/relative
  raw=path.read_bytes() if path.exists() else subprocess.check_output(['git','show','HEAD:'+relative],cwd=R)
  tables[filename]=pd.read_csv(io.BytesIO(raw));provenance.append(dict(path=relative,sha256=hashlib.sha256(raw).hexdigest()))
 for group in ['target','non_target']:
  obs=tables['real_observed_score_components'];obs=obs[obs.pair_group==group+'_observed']
  fake=tables['synthetic_score_components'];fake=fake[fake.pair_group==group]
  assert np.allclose(obs.mae_before,obs.mae_before.iloc[0])
  for n in [0,2,4,6,25,50,100,200,500]:
   real=obs[obs.n_fake==n].iloc[0] if n else obs.iloc[0]
   mae_real=float(real.mae_after if n else real.mae_before);nr=int(real.n_pairs)
   if n:
    sy=fake[fake.n_fake==n].iloc[0];nf=int(sy.n_pairs);mae_fake=float(sy.mae)
    for view,expected in [('synthetic_training',sy.mean_prediction),('real_observed_training',real.mean_prediction_after)]:
     check=f[(f.model==model)&(f.n_fake==n)&(f.view==view)&(f.group==group)].iloc[0]
     assert np.isclose(check.mean_prediction,expected,atol=1e-10)
   else:nf=0;mae_fake=0.
   total=(nr*mae_real+nf*mae_fake)/(nr+nf)
   assert total>=0 and np.isfinite(total)
   mae_rows.append(dict(model=model,group=group,n_fake=n,n_real_ratings=nr,n_synthetic_ratings=nf,real_mae=mae_real,synthetic_mae=mae_fake if n else None,all_mae=total))
maes=pd.DataFrame(mae_rows);maes.to_csv(D/'training_all_users_mae.csv',index=False)
(D/'training_mae_validation.json').write_text(json.dumps(dict(sources=provenance,definition='mean(abs(recorded rating - raw predicted score))',weighting='equal observed training rating within each book group, real plus synthetic',error_of_mean_not_used=True,no_retraining=True),indent=2))

f[f.view.isin(['all_training','synthetic_training'])].to_csv(D/'training_components.csv',index=False)

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':14,'axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False,'axes.titleweight':'bold','axes.labelcolor':'#163248','text.color':'#163248','xtick.color':'#536471','ytick.color':'#536471','figure.facecolor':'white','savefig.facecolor':'white'})
colors={'mean_prediction':'#009E73','mean_bi':'#D55E00','mean_bu':'#61B8E8','mean_interaction':'#0072B2','mu':'#D7AA00'}
labels={'mean_prediction':r'Predicted rating $\hat r_{ui}$','mean_bi':r'Book bias $b_i$','mean_bu':r'User bias $b_u$','mean_interaction':r'Vector dot product $p_u^\top q_i$','mu':r'Global rating mean $\mu$'}
names={'all_training':('training_all_users_thriller_non_thriller_4panel.png','Training fit | All users','Real + synthetic observed ratings; equal weight per rating within each book group.'),'synthetic_training':('training_synthetic_users_thriller_non_thriller_4panel.png','Training fit | Synthetic users','Synthetic ratings: Thriller = 5; non-Thriller = 1. Model trained on all users.')}
for view,(name,title,note) in names.items():
 if '--all-only' in sys.argv and view!='all_training':continue
 fig,axs=plt.subplots(2,2,figsize=(16,11.5),sharey='row')
 for row,group in enumerate(['target','non_target']):
  for col,model in enumerate(['svd','nmf']):
   ax=axs[row,col];d=f[(f.view==view)&(f.group==group)&(f.model==model)].sort_values('n_fake')
   assert len(d)==(8 if view=='synthetic_training' else 9)
   assert np.allclose(d.mean_prediction,d.mu+d.mean_bu+d.mean_bi+d.mean_interaction,atol=1e-10)
   x=np.arange(len(d))
   for key in ['mu','mean_bi','mean_bu','mean_interaction','mean_prediction']:
    ax.plot(x,d[key],color=colors[key],lw=4.2 if key=='mean_prediction' else 2.2,marker='o' if key=='mean_prediction' else None,ms=5,zorder=5 if key=='mean_prediction' else 3,ls='--' if key=='mu' else '-')
   err=maes[(maes.model==model)&(maes.group==group)].sort_values('n_fake')
   if view=='synthetic_training':err=err[err.n_fake>0]
   assert err.n_fake.tolist()==d.n_fake.tolist()
   values=err.all_mae if view=='all_training' else err.synthetic_mae
   ax.plot(x,values,color='#D62728',lw=2.8,marker='s',ms=5,zorder=6)
   ax.axhline(0,color='#A6B2BB',lw=.7,zorder=1)
   ax.set_title(model.upper()+' | '+('Thriller books' if group=='target' else 'Non-Thriller books'),loc='left',pad=16,fontsize=19)
   ax.set_xticks(x,['Base' if n==0 else str(n) for n in d.n_fake]);ax.set_xlabel('Injected users',labelpad=8);ax.set_ylabel('Mean score / absolute error');ax.grid(axis='y',color='#DEE5EA',lw=.7);ax.tick_params(length=0,pad=7,labelleft=True)
 fig.suptitle(title+' | Thriller injection',x=.065,ha='left',y=.975,fontsize=25,fontweight='bold')
 fig.text(.5,.91,r'$\hat r_{ui}=\mu+b_u+b_i+p_u^\top q_i$',ha='center',fontsize=23)
 handles=[Line2D([0],[0],color=colors[k],lw=4 if k=='mean_prediction' else 2.2,ls='--' if k=='mu' else '-',label=labels[k]) for k in colors]
 handles.append(Line2D([0],[0],color='#D62728',lw=2.8,marker='s',label=r'MAE: mean $|r_{ui}-\hat r_{ui}|$'))
 fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.51,.87),ncol=3,frameon=False,fontsize=12,columnspacing=1.6)
 fig.text(.5,.025,note+' MAE: lower is better. End-of-training measurements.',ha='center',fontsize=11,color='#536471')
 fig.subplots_adjust(left=.065,right=.98,top=.74,bottom=.09,wspace=.16,hspace=.46)
 fig.savefig(O/name,dpi=400);fig.savefig(D/name.replace('.png','_preview.png'),dpi=100);plt.close(fig)
 print(name,flush=True)
