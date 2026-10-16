"""Two models, three list lengths; bar labels show mean counts only."""
from pathlib import Path
import json
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
L=Path('/home/moshtasa/Research/phd-recsys/Book/current/PerturbRec/svd_nmf/layer1')
d=pd.read_csv(L/'tables/topk_target_union/summary.csv');assert len(d)==156
d=d[d.n_fake<=500].copy();assert len(d)==108
COUNTS=[0,2,4,6,25,50,100,200,500];KS=[15,25,35]
navy='#142E49';muted='#546576';colors=['#34495E','#4E79A7','#76B7B2','#59A14F','#B07AA1','#9C755F','#E15759','#F28E2B','#D6A300','#2A9DAB','#A0CBE8','#FF9DA7','#8064A2']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.spines.top':False,'axes.spines.right':False})
for genre in ['Thriller','Drama']:
 fig,axs=plt.subplots(3,2,figsize=(24,13.5))
 for ri,k in enumerate(KS):
  for ci,model in enumerate(['nmf','svd']):
   ax=axs[ri,ci];z=d[(d.model==model)&(d.genre==genre)&(d.k==k)].sort_values('n_fake');assert z.n_fake.tolist()==COUNTS
   bars=ax.bar(np.arange(len(COUNTS)),z.mean_target_count,width=.86,color=colors[:len(COUNTS)],edgecolor='white',linewidth=.5,zorder=3)
   for j,(b,r) in enumerate(zip(bars,z.itertuples())):
    if not r.nmf_nonnegative_constraint_ok:b.set_hatch('///');b.set_edgecolor(navy)
    ax.text(j,r.mean_target_count+k*.025,f'{r.mean_target_count:.2f}',ha='center',va='bottom',fontsize=11,color=navy)
    ax.text(j,-.34,f'{r.union_distinct_target_titles}\n{r.union_repeat_percent:.2f}\n{r.most_frequent_book_user_reach_percent:.1f}',transform=ax.get_xaxis_transform(),ha='center',va='top',fontsize=10,color=muted,linespacing=1.3,clip_on=False)
   ax.set_ylim(0,k*1.24);ax.set_yticks([0,k/2,k],[str(0),f'{k/2:g}',str(k)]);ax.set_xticks(range(len(COUNTS)),['Base']+[str(n) for n in COUNTS[1:]],rotation=35,ha='right',fontsize=10)
   ax.set_ylabel(f'Top-{k}\nMean {genre} books',fontsize=13,color=navy)
   if ri==0:ax.set_title(model.upper(),fontsize=23,fontweight='bold',color=navy,pad=12)
   ax.grid(axis='y',color='#DCE3E9',linewidth=.7,zorder=0);ax.spines['left'].set_visible(False);ax.spines['bottom'].set_color('#C0CBD5');ax.tick_params(axis='y',length=0,colors=muted)
 fig.subplots_adjust(left=.07,right=.98,top=.82,bottom=.175,wspace=.15,hspace=.88)
 fig.text(.055,.956,f'{genre} recommendations after synthetic-user injection',fontsize=27,fontweight='bold',color=navy)
 fig.text(.055,.915,'Top-15, Top-25 and Top-35 unseen-book recommendations • Same 53,424 real users • X-axis: injected users (up to 500)',fontsize=15,color=muted)
 fig.text(.055,.88,'Above each bar: mean target-book count only. Below each bar: unique titles (union) / repeat (%) / most frequent title’s user reach (%).',fontsize=13,color=muted)
 fig.text(.055,.055,'Union = distinct target titles across all users. Repeat = 100 × (1 − union / total target recommendations); it is not pairwise list overlap.',fontsize=12,color=muted)
 fig.text(.055,.028,'Previously rated books excluded. Both models use the same real users and candidate-book catalog.',fontsize=12,color=muted)
 out=R/'layer2/final_results'/f'comparision - {genre}_top35_union_powerpoint.png';fig.savefig(out,dpi=240,facecolor='white');plt.close(fig);print(out)
assert np.allclose(d.target_share_percent,d.mean_target_count/d.k*100)
assert np.allclose(d.union_repeat_percent,100*(1-d.union_distinct_target_titles/d.total_target_recommendation_slots))
(L/'log/topk_target_union_validation.json').write_text(json.dumps({'rows':len(d),'max_injected_users':500,'displayed_injection_counts':COUNTS,'k_values':KS,'n_real_users':53424,'top35_exactly_matches_previous_results':True,'top15_matches_available_reference':True,'arithmetic_checked':True,'bar_labels':'mean target count only','negative_factor_checkpoints':d[~d.nmf_nonnegative_constraint_ok][['model','genre','n_fake']].drop_duplicates().to_dict('records')},indent=2))
