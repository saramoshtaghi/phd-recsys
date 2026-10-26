#!/usr/bin/env python3
"""Adventure/Drama counterparts of the two Thriller reviewer figures; saved models only."""
import os
for k in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,importlib.util,argparse,hashlib,time
from concurrent.futures import ProcessPoolExecutor
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
R=Path('/home/moshtasa/Research/phd-recsys');NB=R/'First_paper_material/notebook';ROOT=R/'Book/current/PerturbRec/svd_nmf';E=ROOT/'layer2/embeddings';OUT=R/'First_paper_material/figures/first_reviewer';DRAMA_COUNTS=[0,2,4,6,25,50,100,200,500,1000,2000,2500,3000]
COLORS={'mean_prediction':'#009E73','mean_bi':'#D55E00','mean_bu':'#61B8E8','mean_interaction':'#0072B2','mu':'#D7AA00','mae':'#D62728'}
LABELS={'mean_prediction':r'Prediction $\hat r_{ui}$','mean_bi':r'Book bias $b_i$','mean_bu':r'User bias $b_u$','mean_interaction':r'Dot product $p_u^\top q_i$','mu':r'Global mean $\mu$','mae':'MAE'}

def drama_case(job):
 model,n=job;path=E/'pilot'/f'{model}_baseline_a.npz' if n==0 else E/'controlled'/model/'drama'/f'{model}_drama_{n}.npz'
 with np.load(path) as z:m=A.align(z);minp=float(z['pu'].min());minq=float(z['qi'].min())
 assert len(m['fp'])==n and m['p'].shape==(53424,8) and m['q'].shape==(9415,8)
 rows=A.components(m,n,model);expected=(REAL_SUM+n*SYNTH_SUM)/(len(A.RATING)+n*len(A.ITEMS));assert np.isclose(expected,m['mu'],rtol=0,atol=1e-12)
 if n:
  for group,mask in [('target',A.TARGET),('non_target',~A.TARGET)]:
   r=next(v for v in rows if v['view']=='synthetic_training' and v['group']==group);dot=float(m['fp'].mean(0)@m['q'][mask].mean(0));assert np.isclose(dot,r['mean_interaction'],rtol=1e-10,atol=1e-10)
 valid=bool(model!='nmf' or min(minp,minq)>=-1e-12);print('DRAMA COMPONENTS',model,n,'sign check',valid,flush=True)
 return rows,dict(model=model,n_fake=n,checkpoint=str(path),min_pu=minp,min_qi=minq,nmf_nonnegative_constraint_ok=valid)

def compute_drama():
 global A,REAL_SUM,SYNTH_SUM
 dest=NB/'drama_reviewer_score_components.csv';audit=NB/'drama_reviewer_components_validation.json'
 if dest.exists() and audit.exists():return
 spec=importlib.util.spec_from_file_location('adventure_core',NB/'analyze_adventure.py');A=importlib.util.module_from_spec(spec);spec.loader.exec_module(A)
 df=pd.read_csv(R/'Book/Archive/data/genre_clean.csv',usecols=['user_id','book_id','rating','genres']);A.USERS=np.sort(df.user_id.unique());A.ITEMS=np.sort(df.book_id.unique());A.N=len(A.USERS);A.UI=np.searchsorted(A.USERS,df.user_id.to_numpy());A.II=np.searchsorted(A.ITEMS,df.book_id.to_numpy());A.RATING=df.rating.to_numpy(float);tags=df.drop_duplicates('book_id').set_index('book_id').loc[A.ITEMS].genres;A.TARGET=np.array(['Drama' in {v.strip() for v in str(x).split(',')} for x in tags]);REAL_SUM=float(A.RATING.sum());SYNTH_SUM=float(np.where(A.TARGET,5,1).sum());assert A.N==53424 and len(A.ITEMS)==9415 and len(df)==5815569
 with ProcessPoolExecutor(max_workers=2) as pool:results=list(pool.map(drama_case,[(m,n) for m in ['svd','nmf'] for n in DRAMA_COUNTS]))
 f=pd.DataFrame([row for rows,_ in results for row in rows]);f.to_csv(dest,index=False);assert np.allclose(f.mean_prediction,f.mu+f.mean_bu+f.mean_bi+f.mean_interaction,atol=1e-10);assert (f.loc[f.view!='real_unseen_prediction','mae']>=0).all()
 audit.write_text(json.dumps(dict(status='complete',genre='Drama',counts=DRAMA_COUNTS,n_real=53424,n_items=9415,target_books=int(A.TARGET.sum()),no_retraining=True,clipped=False,prediction_identity_verified=True,synthetic_interaction_independently_verified=True,global_mean_formula_verified=True,checkpoints=[v for _,v in results]),indent=2))

def setup_small():
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7,'axes.titlesize':7.2,'axes.titleweight':'bold','axes.labelsize':6.5,'xtick.labelsize':6,'ytick.labelsize':6,'axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False,'axes.linewidth':.5,'axes.edgecolor':'#A8B1B9','text.color':'#163248','axes.labelcolor':'#344755','xtick.color':'#536471','ytick.color':'#536471','figure.facecolor':'white','savefig.facecolor':'white','pdf.fonttype':42})
def ticks(ns,model,bad):return [('Base' if n==0 else str(n))+('†' if model=='nmf' and n in bad else '') for n in ns]
def warning(fig,bad,small):
 if bad:fig.text(.5,.014 if small else .005,'† NMF factor sign check failed at n = '+', '.join(map(str,bad))+'. Raw results retained.',ha='center',fontsize=5.6 if small else 9,color='#9D211C')
def eight(f,genre,folder,ns,bad):
 setup_small();fig=plt.figure(figsize=(5.3,5.8));outer=fig.add_gridspec(2,1,left=.10,right=.985,bottom=.105 if len(ns)>9 else .095,top=.80,hspace=.48)
 for section,view in enumerate(['all_training','synthetic_training']):
  grid=outer[section].subgridspec(2,2,hspace=.35,wspace=.25)
  for row,group in enumerate(['target','non_target']):
   axs=[];values=[]
   for col,model in enumerate(['svd','nmf']):
    ax=fig.add_subplot(grid[row,col],sharey=axs[0] if axs else None);axs.append(ax);d=f[(f.view==view)&(f.group==group)&(f.model==model)&f.n_fake.isin(ns)].sort_values('n_fake');expected=ns if view=='all_training' else [n for n in ns if n];assert d.n_fake.tolist()==expected
    assert np.allclose(d.mean_prediction,d.mu+d.mean_bu+d.mean_bi+d.mean_interaction,atol=1e-10);x=np.arange(len(d));mae=d.mae.to_numpy();assert np.isfinite(mae).all() and (mae>=0).all()
    for key in ['mu','mean_bi','mean_bu','mean_interaction','mean_prediction']:ax.plot(x,d[key],color=COLORS[key],lw=1.65 if key=='mean_prediction' else .95,ls='--' if key=='mu' else '-',marker='o' if key=='mean_prediction' else None,ms=2,zorder=5 if key=='mean_prediction' else 3)
    ax.plot(x,mae,color=COLORS['mae'],lw=1.05,marker='s',ms=1.9,zorder=6);ax.axhline(0,color='#A8B1B9',lw=.45,zorder=1);ax.grid(axis='y',color='#DEE5EA',lw=.4);ax.set_axisbelow(True);ax.set_xticks(x,ticks(expected,model,bad),rotation=55 if len(ns)>9 else 0,ha='right' if len(ns)>9 else 'center');ax.tick_params(length=0,pad=2.5,labelsize=5.5 if len(ns)>9 else 6);ax.set_xlim(-.25,len(d)-.75);ax.yaxis.set_major_locator(MaxNLocator(nbins=4));panel=chr(97+section*4+row*2+col);ax.set_title(f'({panel}) {model.upper()} | '+(genre if group=='target' else 'Non-'+genre),loc='left',pad=5)
    if col==0:ax.set_ylabel('Mean score / MAE',labelpad=3)
    if row==1:ax.set_xlabel('Injected users',labelpad=2)
    else:ax.tick_params(axis='x',labelbottom=False)
    values.append(np.r_[d[['mean_prediction','mean_bi','mean_bu','mean_interaction','mu']].to_numpy().ravel(),mae])
   vv=np.concatenate(values);pad=max((vv.max()-vv.min())*.08,.15);axs[0].set_ylim(vv.min()-pad,vv.max()+pad)
  box=outer[section].get_position(fig);fig.text(.10,box.y1+.048,'All users (real + synthetic)' if section==0 else 'Synthetic users only',ha='left',va='bottom',fontsize=8.2,fontweight='bold')
 fig.text(.5,.98,r'$\hat r_{ui}=\mu+b_u+b_i+p_u^\top q_i$',ha='center',va='top',fontsize=10);handles=[Line2D([0],[0],color=COLORS[k],lw=1.65 if k=='mean_prediction' else 1,ls='--' if k=='mu' else '-',label=LABELS[k]) for k in COLORS];fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.53,.935),ncol=3,frameon=False,fontsize=6.8,columnspacing=1.7,handlelength=2.2,labelspacing=.55);warning(fig,bad,True);path=folder/f'training_{genre.lower()}_all_and_synthetic_8panel_paper.png';fig.savefig(path,dpi=600);plt.close(fig);return path

def prediction(f,genre,folder,ns,bad):
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':14,'axes.labelsize':14,'xtick.labelsize':14,'ytick.labelsize':14,'axes.linewidth':.8,'axes.edgecolor':'black','axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False,'axes.titleweight':'bold','axes.labelcolor':'#163248','text.color':'#163248','xtick.color':'#536471','ytick.color':'#536471','figure.facecolor':'white','savefig.facecolor':'white'})
 fig,axs=plt.subplots(2,2,figsize=(16,11.5),sharey='row')
 for row,group in enumerate(['target','non_target']):
  for col,model in enumerate(['svd','nmf']):
   ax=axs[row,col];d=f[(f.view=='real_unseen_prediction')&(f.group==group)&(f.model==model)&f.n_fake.isin(ns)].sort_values('n_fake');assert d.n_fake.tolist()==ns;assert np.allclose(d.mean_prediction,d.mu+d.mean_bu+d.mean_bi+d.mean_interaction,atol=1e-10);x=np.arange(len(d))
   for key in ['mu','mean_bi','mean_bu','mean_interaction','mean_prediction']:ax.plot(x,d[key],color=COLORS[key],lw=4.2 if key=='mean_prediction' else 2.2,marker='o' if key=='mean_prediction' else None,ms=5,zorder=5 if key=='mean_prediction' else 3,ls='--' if key=='mu' else '-')
   ax.axhline(0,color='#A6B2BB',lw=.7,zorder=1);ax.set_title(model.upper()+' | '+(genre+' books' if group=='target' else 'Non-'+genre+' books'),loc='left',pad=16,fontsize=19);ax.set_xticks(x,ticks(ns,model,bad),rotation=45 if len(ns)>9 else 0,ha='right' if len(ns)>9 else 'center');ax.set_xlabel('Injected users',labelpad=8);ax.set_ylabel('Mean raw score component');ax.grid(axis='y',color='#DEE5EA',lw=.7);ax.tick_params(length=0,pad=7,labelleft=True)
 fig.suptitle('Prediction | Real users, unrated books | '+genre+' injection',x=.065,ha='left',y=.975,fontsize=24,fontweight='bold');fig.text(.5,.91,r'$\hat r_{ui}=\mu+b_u+b_i+p_u^\top q_i$',ha='center',fontsize=23)
 names={'mean_prediction':r'Predicted rating $\hat r_{ui}$','mean_bi':r'Book bias $b_i$','mean_bu':r'User bias $b_u$','mean_interaction':r'Vector dot product $p_u^\top q_i$','mu':r'Global rating mean $\mu$'};handles=[Line2D([0],[0],color=COLORS[k],lw=4 if k=='mean_prediction' else 2.2,ls='--' if k=='mu' else '-',label=names[k]) for k in names];fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.51,.855),ncol=5,frameon=False,fontsize=12,columnspacing=1.6);fig.text(.5,.035 if bad else .025,'Equal weight per real user; average over all unrated books in each group (not only Top-K). Raw scores; no observed ratings for MAE.',ha='center',fontsize=11,color='#536471');warning(fig,bad,False);fig.subplots_adjust(left=.065,right=.98,top=.75,bottom=.14 if len(ns)>9 else .09,wspace=.16,hspace=.56 if len(ns)>9 else .46);path=folder/f'prediction_real_users_{genre.lower()}_non_{genre.lower()}_4panel.png';fig.savefig(path,dpi=400);plt.close(fig);return path

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--compute-only',action='store_true');ap.add_argument('--max-count',type=int,default=500);args=ap.parse_args();compute_drama()
 if args.compute_only:raise SystemExit(0)
 records=[]
 for genre,folder,source in [('Adventure',OUT/'advanture',ROOT/'layer1/tables/adventure_score_components.csv'),('Drama',OUT/'drama',NB/'drama_reviewer_score_components.csv')]:
  assert folder.is_dir();f=pd.read_csv(source);ns=sorted(f.n_fake.unique().tolist());ns=[n for n in ns if args.max_count is None or n<=args.max_count]
  if genre=='Adventure':bad=[r['n_fake'] for r in json.loads((NB/'adventure_analysis_status.json').read_text())['nonnegative_constraint_failures']]
  else:bad=[r['n_fake'] for r in json.loads((NB/'drama_reviewer_components_validation.json').read_text())['checkpoints'] if not r['nmf_nonnegative_constraint_ok']]
  bad=sorted(set(bad)&set(ns));paths=[eight(f,genre,folder,ns,bad),prediction(f,genre,folder,ns,bad)];records.append(dict(genre=genre,folder=str(folder),source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),counts=ns,images=list(map(str,paths)),nonnegative_constraint_failures=bad));print('FIGURES COMPLETE',genre,ns,flush=True)
 (NB/'genre_reviewer_figures_manifest.json').write_text(json.dumps(dict(status='complete',no_retraining=True,raw_components_preserved=True,reference_folder=str(OUT/'thriller'),figures=records,completed=time.time()),indent=2))
