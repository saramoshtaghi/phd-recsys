"""Render the saved paper exposure values with a consistent model palette."""
from pathlib import Path
import hashlib,json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

R=Path('/home/moshtasa/Research/phd-recsys/First_paper_material')
source=R/'data/recommendation_exposure_top15.csv'
dest=R/'figures/recommendation_exposure_top15.png'
df=pd.read_csv(source)
assert len(df)==36 and set(df.k)=={15}
colors={'SVD':'#0072B2','NMF':'#D55E00'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,'axes.titleweight':'bold','axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,'axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False,'axes.linewidth':.5,'figure.facecolor':'white','savefig.facecolor':'white'})
fig,axs=plt.subplots(2,3,figsize=(7.1,4.2),sharey=True)
series=[]
for col,(dataset,target) in enumerate([('Books','Thriller'),('Mobile apps','Shopping'),('Movies','1990s')]):
    for row,model in enumerate(['NMF','SVD']):
        ax=axs[row,col]
        d=df[(df.dataset==dataset)&(df.model==model)].sort_values('n_fake')
        assert d.n_fake.iloc[0]==0 and d.mean_target_count.between(0,15).all()
        x=list(range(len(d)));v=d.mean_target_count.to_numpy()
        ax.bar(x,v,width=.76,color=['#AAB2BA' if n==0 else colors[model] for n in d.n_fake],edgecolor='white',linewidth=.4,zorder=3)
        for j,value in enumerate(v):
            label=f'{value:.2f}' if value<1 else f'{value:.1f}'
            extra=.65 if len(v)>8 and value>=10 and j%2==0 else 0
            ax.text(j,value+.23+extra,label,ha='center',va='bottom',fontsize=6.7,color='#172B3A')
        ax.set_ylim(0,16.8);ax.yaxis.set_major_locator(MultipleLocator(3))
        ax.set_xticks(x,['Base']+[str(n) for n in d.n_fake.iloc[1:]],rotation=45,ha='right')
        ax.tick_params(axis='both',length=0,pad=3,colors='#4E5B65')
        ax.set_axisbelow(True);ax.grid(axis='y',color='#DFE5E9',lw=.45,zorder=0)
        ax.spines['bottom'].set_color('#AEBAC2')
        ax.text(.03,.93,model,transform=ax.transAxes,ha='left',fontsize=8,fontweight='bold',color=colors[model])
        if row==0:ax.set_title(f'({chr(97+col)}) {dataset}: {target}',loc='left',pad=11,color='#172B3A')
        else:ax.set_xlabel('Injected users',labelpad=4,color='#4E5B65')
        series.append({'dataset':dataset,'model':model,'n_fake':d.n_fake.tolist(),'values':v.tolist()})
fig.supylabel('Mean target items in Top-15',x=.014,fontsize=9,color='#172B3A')
fig.subplots_adjust(left=.075,right=.99,top=.89,bottom=.15,wspace=.16,hspace=.45)
fig.savefig(dest,dpi=600);plt.close(fig)
(R/'notebook/paper_exposure_palette_validation.json').write_text(json.dumps({'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'output':str(dest),'palette':colors,'no_new_experiment':True,'source_values_unchanged':True,'series':series},indent=2))
print(dest)
