from pathlib import Path
import argparse,json
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
R=Path('/home/moshtasa/Research/phd-recsys/Book/current/PerturbRec/svd_nmf');L=R/'layer1';T=L/'tables/top35_target_union'
p=argparse.ArgumentParser();p.add_argument('--metric',choices=['union','top1','top10'],default='union');a=p.parse_args()
d=pd.read_csv(T/'summary.csv');COUNTS=[0,2,4,6,25,50,100,200,500,1000,2000,2500,3000]
navy='#142E49';muted='#546576';colors=['#34495E','#4E79A7','#76B7B2','#59A14F','#B07AA1','#9C755F','#E15759','#F28E2B','#D6A300','#2A9DAB','#A0CBE8','#FF9DA7','#8064A2']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':13,'axes.spines.top':False,'axes.spines.right':False})
metric={'union':('union_repeat_percent','Repeat','Repeat = target recommendation occurrences beyond the first of each distinct title, divided by all target occurrences. This is NOT pairwise list overlap.'),'top1':('most_frequent_book_share_of_target_slots_percent','Top title','Top title = share of all target recommendations occupied by the single most frequent title; the title can differ across conditions.'),'top10':('top10_share_of_target_slots_percent','Top 10','Top 10 = share of all target recommendations occupied by the ten most frequently recommended target titles.')}[a.metric]
for genre in ['Thriller','Drama']:
 fig,axs=plt.subplots(1,2,figsize=(20,11.25),sharey=True)
 for ax,model in zip(axs,['nmf','svd']):
  z=d[(d.model==model)&(d.genre==genre)].sort_values('n_fake');assert z.n_fake.tolist()==COUNTS;x=np.arange(len(z))
  bars=ax.bar(x,z.mean_target_count,width=.9,color=colors,edgecolor='white',linewidth=.5,zorder=3)
  for j,(bar,row) in enumerate(zip(bars,z.itertuples())):
   if not row.nmf_nonnegative_constraint_ok:bar.set_hatch('///');bar.set_edgecolor(navy);bar.set_linewidth(.5)
   ax.text(j,row.mean_target_count+.4,f'{row.mean_target_count:.2f}\n({row.target_share_of_top35_percent:.1f}%)',ha='center',va='bottom',fontsize=10,color=navy,linespacing=1.2)
   ax.text(j,-.23,f'U={row.union_distinct_target_titles:,}\n{getattr(row,metric[0]):.2f}%',transform=ax.get_xaxis_transform(),ha='center',va='top',fontsize=10,color=muted,linespacing=1.5,clip_on=False)
  ax.set_title(model.upper(),fontsize=24,fontweight='bold',color=navy,pad=20)
  ax.set_ylim(0,40);ax.set_yticks(range(0,36,5));ax.set_xticks(x,['Base']+[str(n) for n in COUNTS[1:]],rotation=45,ha='right',fontsize=11,color=muted);ax.tick_params(axis='y',colors=muted,length=0)
  ax.set_xlabel('Injected users',labelpad=10,color=muted);ax.grid(axis='y',color='#DCE3E9',linewidth=.7,zorder=0);ax.set_axisbelow(True);ax.spines['left'].set_visible(False);ax.spines['bottom'].set_color('#C0CBD5')
 axs[0].set_ylabel(f'Mean {genre} books per real user (out of 35)',fontsize=15,color=navy,labelpad=15)
 fig.text(.055,.94,f'{genre} recommendations after synthetic-user injection',fontsize=25,fontweight='bold',color=navy)
 fig.text(.055,.895,'Top-35 unseen-book recommendations • Same 53,424 real users • Baseline and 12 injection sizes',fontsize=14,color=muted)
 fig.text(.055,.85,'Above each bar: mean count and share of 35. Below: distinct target titles (U), then '+metric[1]+' (%).',fontsize=13,color=muted)
 fig.subplots_adjust(left=.075,right=.975,top=.75,bottom=.33,wspace=.13)
 fig.text(.055,.115,metric[2],fontsize=11.5,color=navy)
 fig.text(.055,.08,'U is the union of target-book IDs across all real users. Rankings use saved factors + item biases; rated books excluded; book-ID-stable ties.',fontsize=11,color=muted)
 bad=d[(d.model=='nmf')&(d.genre==genre)&(~d.nmf_nonnegative_constraint_ok)]
 if len(bad):fig.text(.055,.04,'Hatched NMF bars: historical checkpoints contain negative factors; these are reproduced saved outputs, not validated nonnegative fits.',fontsize=11,color='#88482A')
 out=L/'result/final_results'/f'comparision - {genre}_top35_union_powerpoint.png';fig.savefig(out,dpi=240,facecolor='white');plt.close(fig);print(out)
manifest={'k':35,'metric':a.metric,'definition':metric[2],'n_summary_rows':len(d),'no_retraining':True,'negative_factor_checkpoints':d[~d.nmf_nonnegative_constraint_ok][['model','genre','n_fake']].to_dict('records'),'source':'saved controlled factors, original real users; recomputed full Top-35 with stable ID ties','validation':'all available Top-15 and extension Top-35 reference means matched exactly'}
(L/'log/top35_target_union_validation.json').write_text(json.dumps(manifest,indent=2))
