"""Paper-ready Top-15 exposure bars for three domains; saved outputs only, no training."""
from pathlib import Path
import re,json,hashlib
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
ROOT=Path('/home/moshtasa/Research/phd-recsys');OUT=ROOT/'First_paper_material/Book';DATA=OUT/'data';SINGLE=OUT/'recommendation_exposure';DATA.mkdir(parents=True,exist_ok=True);SINGLE.mkdir(parents=True,exist_ok=True)
L=ROOT/'Book/current/PerturbRec/svd_nmf/layer1'
book_file=L/'tables/controlled_thriller_top15.csv';book=pd.read_csv(book_file)
configs=[('Books','Thriller',500,53424,'book_thriller_top15.png'),('Mobile apps','Shopping',300,700111,'mobile_shopping_top15.png'),('Movies','1990s',50,943,'movie_1990s_top15.png')]
records=[];sources=[]
def parse(path):
 current=None;result={}
 for line in path.read_text().splitlines():
  m=re.match(r'\s*k\s*=\s*(\d+)',line)
  if m:current=int(m.group(1));continue
  m=re.match(r'\s*(original|\d+u)\s*:\s*([0-9.]+)\s*\(users=(\d+)\)',line,re.I)
  if current==15 and m:
   n=0 if m[1].lower()=='original' else int(m[1][:-1]);assert n not in result;result[n]=(float(m[2]),int(m[3]))
 assert result and 0 in result;return result
for dataset,target,limit,N,name in configs:
 for model in ['nmf','svd']:
  if dataset=='Books':
   d=book[(book.model==model)&(book.genre=='Thriller')&(book.k==15)&(book.n_fake<=limit)].sort_values('n_fake')
   assert len(d)==9 and (d.n_users==N).all()
   pairs={int(r.n_fake):(float(r.mean_target_count),int(r.n_users)) for r in d.itertuples()};source=book_file
   check=pd.read_csv(L/'tables/topk_target_union/summary.csv');check=check[(check.model==model)&(check.genre=='Thriller')&(check.k==15)&(check.n_fake<=500)].sort_values('n_fake');assert np.allclose(d.mean_target_count.to_numpy(),check.mean_target_count.to_numpy(),atol=1e-12)
  elif dataset=='Mobile apps':
   source=ROOT/f'Mobile/archive/0405/0405_{model.upper()}/result/{model.upper()}_bar_figures/{model}_shopping_analysis/{model.upper()} - Shopping.txt';pairs={n:v for n,v in parse(source).items() if n<=limit}
   assert sorted(pairs)==[0,30,60,300]
  else:
   source=ROOT/f'Movie /current/0519/{model.upper()}_0519/result/0519_{model.upper()}_bar_figures/{model}_decade_1990_analysis/{model.upper()} - Decade 1990.txt';pairs=parse(source);assert sorted(pairs)==[0,2,6,10,50]
  for n,(value,users) in sorted(pairs.items()):
   assert 0<=value<=15 and users==N
   records.append(dict(dataset=dataset,target=target,model=model.upper(),k=15,n_fake=n,n_real_users=users,mean_target_count=value,source=str(source),source_precision='full CSV precision' if dataset=='Books' else 'four decimals in original analysis output'))
  sources.append({'dataset':dataset,'model':model.upper(),'file':str(source),'sha256':hashlib.sha256(source.read_bytes()).hexdigest()})
df=pd.DataFrame(records);assert len(df)==36;assert not df.duplicated(['dataset','model','n_fake']).any()
df.to_csv(DATA/'recommendation_exposure_top15.csv',index=False)
NAVY='#172B3A';MUTED='#4E5B65';COLORS={'NMF':'#0072B2','SVD':'#D55E00'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,'axes.titleweight':'bold','axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,'axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False,'axes.linewidth':.5,'savefig.facecolor':'white','figure.facecolor':'white'})
# Model colours are consistent across domains; baseline is neutral grey.
def panel(ax,d,model,annotate=True,compact=False):
 d=d.sort_values('n_fake');x=np.arange(len(d));v=d.mean_target_count.to_numpy()
 ax.bar(x,v,width=.76,color=['#AAB2BA' if int(n)==0 else COLORS[model] for n in d.n_fake],linewidth=.4,edgecolor='white',zorder=3)
 if annotate:
  for xx,yy in zip(x,v):ax.text(xx,yy+.23+(.65 if len(v)>8 and yy>=10 and xx%2==0 else 0),(f'{yy:.2f}' if not compact or yy<1 else f'{yy:.1f}'),ha='center',va='bottom',fontsize=6.7,color=NAVY)
 ax.set_ylim(0,16.8);ax.yaxis.set_major_locator(MultipleLocator(3));ax.set_xticks(x,['Base']+[str(n) for n in d.n_fake.iloc[1:]],rotation=45,ha='right');ax.tick_params(axis='both',length=0,pad=3,colors=MUTED)
 ax.grid(axis='y',color='#DFE5E9',linewidth=.45,zorder=0);ax.set_axisbelow(True);ax.spines['bottom'].set_color('#AEBAC2')
 ax.text(.03,.93,model,transform=ax.transAxes,ha='left',fontsize=8,fontweight='bold',color=COLORS[model])
# Combined figure: datasets are side by side; rows use the same models and y-scale.
fig,axs=plt.subplots(2,3,figsize=(7.1,4.2),sharey=True)
for ci,(dataset,target,limit,N,name) in enumerate(configs):
 for ri,model in enumerate(['NMF','SVD']):
  panel(axs[ri,ci],df[(df.dataset==dataset)&(df.model==model)],model,compact=True)
  if ri==0:axs[ri,ci].set_title(f'({chr(97+ci)}) {dataset}: {target}',loc='left',pad=11,color=NAVY)
  if ri==1:axs[ri,ci].set_xlabel('Injected users',labelpad=4,color=MUTED)
fig.supylabel('Mean target items in Top-15',x=.014,fontsize=9,color=NAVY)
fig.subplots_adjust(left=.075,right=.99,top=.89,bottom=.15,wspace=.16,hspace=.45)
fig.savefig(OUT/'recommendation_exposure_top15.png',dpi=600);plt.close(fig)
# Three individual figures use identical colours, labels, units and limits.
for dataset,target,limit,N,name in configs:
 fig,axs=plt.subplots(1,2,figsize=(5.4,2.65),sharey=True)
 for ax,model in zip(axs,['NMF','SVD']):
  panel(ax,df[(df.dataset==dataset)&(df.model==model)],model)
  ax.set_xlabel('Injected users',color=MUTED,labelpad=4)
 axs[0].set_ylabel('Mean target items in Top-15',color=NAVY)
 fig.suptitle(f'{dataset}: {target}',fontsize=10,fontweight='bold',color=NAVY,x=.55,y=.97)
 fig.subplots_adjust(left=.10,right=.99,top=.81,bottom=.28,wspace=.13)
 fig.savefig(SINGLE/name,dpi=600);plt.close(fig)
(OUT/'recommendation_exposure_caption.md').write_text('Mean number of target-category items in the top-15 recommendations for original users: (a) Thriller books, (b) Shopping applications, and (c) movies released in the 1990s. NMF and SVD are shown in the top and bottom rows, respectively. Book injections extend to 500 users, mobile injections to 300 users, and movie injections to 50 users. Baseline denotes training without synthetic users.\n\nThe x-axis is categorical within each dataset; injection counts and original user populations differ between datasets. The shared y-axis counts items out of 15.\n\nFor full-width display with the MDPI class, use an adjustwidth environment with arguments {-\\extralength}{0cm} and include the PNG at width=\\fulllength.\n')
validation={'rows':len(df),'k':15,'no_retraining':True,'book_matches_two_controlled_CSVs':True,'dose_limits':{'Books':500,'Mobile apps':300,'Movies':50},'sources':sources,'source_numeric_values_unchanged':True,'combined_layout':'2 model rows x 3 dataset columns','all_y_axes_shared':True,'png_dpi':600,'source_precision_note':'Mobile and movie means are faithfully reproduced at the four-decimal precision saved by their original analysis scripts.'}
(DATA/'recommendation_exposure_validation.json').write_text(json.dumps(validation,indent=2))
print(df[['dataset','model','n_fake','mean_target_count']].to_string(index=False));print('SAVED',OUT/'recommendation_exposure_top15.png');print('SEPARATE',SINGLE)
