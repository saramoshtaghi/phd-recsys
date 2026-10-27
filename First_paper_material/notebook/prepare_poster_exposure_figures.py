from pathlib import Path
import os,json,hashlib
os.environ['MPLBACKEND']='Agg'
import pandas as pd,numpy as np,matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
R=Path('/home/moshtasa/Research/phd-recsys');M=R/'First_paper_material'
source=M/'figures/recommendation_exposure_top15.png'
table=M/'data/recommendation_exposure_top15.csv'
out=R/'state of the art/poster_presentation';assert out.is_dir()
f=pd.read_csv(table);assert len(f)==36 and (f.k==15).all()
colors={'NMF':'#0072B2','SVD':'#D55E00'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':17,'axes.labelsize':19,'axes.titlesize':22,'axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False,'text.color':'#173243','axes.labelcolor':'#536572','xtick.color':'#536572','ytick.color':'#536572'})
records=[]
for domain,title,expected,name in [
 ('Books','Books: Thriller',[0,2,4,6,25,50,100,200,500],'books_thriller_top15_poster.png'),
 ('Mobile apps','Mobile Apps: Shopping',[0,30,60,300],'mobile_shopping_top15_poster.png'),
 ('Movies','Movies: 1990s',[0,2,6,10,50],'movies_1990s_top15_poster.png')]:
 fig,axes=plt.subplots(2,1,figsize=(10,8.7),sharey=True)
 for ax,model in zip(axes,['NMF','SVD']):
  d=f[(f.dataset==domain)&(f.model==model)].sort_values('n_fake');assert d.n_fake.tolist()==expected
  vals=d.mean_target_count.to_numpy();x=np.arange(len(d))
  ax.bar(x,vals,width=.72,color=['#AAB2BA' if n==0 else colors[model] for n in d.n_fake],edgecolor='white',linewidth=.7,zorder=3)
  for xx,value in zip(x,vals):ax.text(xx,value+.25,f'{value:.2f}',ha='center',va='bottom',fontsize=15.5,color='#173243')
  ax.set_ylim(0,17);ax.set_xticks(x,[str(n) for n in expected],rotation=40 if len(d)>5 else 0,ha='right' if len(d)>5 else 'center')
  ax.yaxis.set_major_locator(MultipleLocator(3));ax.tick_params(length=0,pad=7,labelsize=16)
  ax.grid(axis='y',color='#DFE3E6',lw=.8);ax.set_axisbelow(True);ax.spines['bottom'].set_color('#AAB4BB')
  ax.set_title(model,loc='left',fontsize=20,fontweight='bold',color=colors[model],pad=10)
 axes[1].set_xlabel('Injected users (0 = baseline)',labelpad=10)
 fig.suptitle(title,x=.13,y=.985,ha='left',fontweight='bold',fontsize=25)
 fig.supylabel('Mean target items in Top-15',x=.015,fontsize=20,color='#173243')
 fig.subplots_adjust(left=.12,right=.985,top=.87,bottom=.13,hspace=.48)
 path=out/name;fig.savefig(path,dpi=400,facecolor='white');plt.close(fig)
 records.append({'dataset':domain,'file':str(path),'counts':expected,'models_top_to_bottom':['NMF','SVD'],'colors':colors,'baseline_color':'#AAB2BA','data':f[f.dataset==domain].to_dict('records')})
 print(path,flush=True)
(M/'notebook/poster_exposure_exports_validation.json').write_text(json.dumps({'source_image':str(source),'source_image_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'source_table':str(table),'source_table_sha256':hashlib.sha256(table.read_bytes()).hexdigest(),'no_new_training':True,'same_values_and_ranges_as_source':True,'outputs':records},indent=2))

