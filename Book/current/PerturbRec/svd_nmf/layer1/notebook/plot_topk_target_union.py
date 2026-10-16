"""Three aligned metrics per model; keep K=15/25/35 and remove repeat-rate clutter."""
from pathlib import Path
import json
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator,StrMethodFormatter
R=Path('/home/moshtasa/Research/phd-recsys/Book/current/PerturbRec/svd_nmf');L=R/'layer1'
d=pd.read_csv(L/'tables/topk_target_union/summary.csv');d=d[d.n_fake<=500].copy();assert len(d)==108
COUNTS=[0,2,4,6,25,50,100,200,500];KS=[15,25,35];COLORS=['#0072B2','#009E73','#D55E00'];MARKERS=['o','s','^'];NAVY='#142E49';MUTED='#546576'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':13,'axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False,'figure.facecolor':'white','savefig.facecolor':'white'})
assert np.allclose(d.target_share_percent,d.mean_target_count/d.k*100)
assert np.allclose(d.most_frequent_book_user_reach_percent,100*d.most_frequent_book_users/d.n_users)
assert d.union_distinct_target_titles.gt(0).all()
for genre in ['Thriller','Drama']:
 fig,axs=plt.subplots(3,2,figsize=(24,13.5),sharex=True)
 data=d[d.genre==genre];umax=float(data.union_distinct_target_titles.max())*1.20
 for ci,model in enumerate(['nmf','svd']):
  for ri,(key,title,ylabel) in enumerate([
   ('mean_target_count',f'1. How many {genre} books are recommended?','Mean books per user'),
   ('union_distinct_target_titles','2. How many different titles are recommended?','Distinct titles across all users'),
   ('most_frequent_book_user_reach_percent','3. How widely is the most frequent book recommended?','Users receiving the top title (%)')]):
   ax=axs[ri,ci];x=np.arange(len(COUNTS))
   for ki,(k,col,marker) in enumerate(zip(KS,COLORS,MARKERS)):
    z=data[(data.model==model)&(data.k==k)].sort_values('n_fake');assert z.n_fake.tolist()==COUNTS;v=z[key].to_numpy()
    if ri==0:
     xx=x+(ki-1)*.25;bars=ax.bar(xx,v,width=.23,color=col,zorder=3)
     for j,(bar,row) in enumerate(zip(bars,z.itertuples())):
      if not row.nmf_nonnegative_constraint_ok:bar.set_hatch('///');bar.set_edgecolor(NAVY)
      ax.text(xx[j],v[j]+.45,f'{v[j]:.1f}',ha='center',va='bottom',fontsize=8,color=NAVY)
    else:
     ax.plot(x,v,color=col,marker=marker,ms=5,lw=2.1,zorder=3)
     # Endpoint values show the starting and final states without dense number rows.
     for j in [0,len(x)-1]:
      offset=([3,15,27][ki] if ri==1 and j==len(x)-1 else [-12,1,13][ki])
      ax.annotate(f'{v[j]:,.0f}' if ri==1 else f'{v[j]:.1f}%',(x[j],v[j]),xytext=(-9 if j==0 else 9,offset),textcoords='offset points',ha='right' if j==0 else 'left',fontsize=10,color=col,fontweight='bold',clip_on=False)
   ax.set_title(title,loc='left',fontsize=15,pad=14,color=NAVY);ax.set_ylabel(ylabel,fontsize=13,color=NAVY)
   ax.set_ylim(0,39 if ri==0 else umax if ri==1 else 106);ax.set_xlim(-.65,8.65)
   ax.set_xticks(x,['Base']+[str(n) for n in COUNTS[1:]]);ax.tick_params(axis='both',length=0,colors=MUTED,labelbottom=True,pad=6,labelsize=11)
   ax.grid(axis='y',color='#DFE5E9',lw=.7);ax.set_axisbelow(True);ax.spines['bottom'].set_color('#BECAD3')
   ax.yaxis.set_major_locator(MaxNLocator(nbins=5,integer=True));ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
   if ri==2:ax.set_xlabel('Injected users',labelpad=8,fontsize=13,color=MUTED)
  axs[0,ci].text(.5,1.38,model.upper(),transform=axs[0,ci].transAxes,ha='center',va='bottom',fontsize=24,fontweight='bold',color=NAVY)
 fig.suptitle(f'{genre}: recommendation volume, title variety and concentration',x=.065,ha='left',y=.975,fontsize=27,fontweight='bold',color=NAVY)
 handles=[Line2D([0],[0],color=c,marker=m,lw=2.5,label=f'Top-{k}') for k,c,m in zip(KS,COLORS,MARKERS)]
 fig.legend(handles=handles,loc='upper right',bbox_to_anchor=(.965,.95),ncol=3,frameon=False,fontsize=14,handlelength=2.2,columnspacing=2)
 fig.subplots_adjust(left=.07,right=.96,top=.80,bottom=.075,hspace=.55,wspace=.22)
 out=R/'layer2/final_results'/f'comparision - {genre}_top35_union_powerpoint.png';fig.savefig(out,dpi=240);preview=L/'log'/f'{genre}_union_preview.png';fig.savefig(preview,dpi=90);plt.close(fig);print(out,flush=True)
(L/'log/topk_target_union_validation.json').write_text(json.dumps({'rows':len(d),'max_injected_users':500,'displayed_injection_counts':COUNTS,'k_values':KS,'n_real_users':53424,'three_metrics':['mean_target_count','union_distinct_target_titles','most_frequent_book_user_reach_percent'],'repeat_rate_removed_from_figure':True,'no_recomputation_of_recommendations':True,'existing_values_preserved':True,'top_title_user_reach_arithmetic_checked':True,'negative_factor_checkpoints':d[~d.nmf_nonnegative_constraint_ok][['model','genre','n_fake']].drop_duplicates().to_dict('records')},indent=2))
