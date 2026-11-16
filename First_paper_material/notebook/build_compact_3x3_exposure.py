from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
R=Path("/home/moshtasa/Research/phd-recsys")
M=R/"First_paper_material";N=M/"notebook"
T=R/"Book/current/PerturbRec/svd_nmf/layer1/tables"
book=pd.read_csv(T/"topk_target_union/summary.csv")
adv=pd.read_csv(T/"adventure_recommendation_summary.csv")
layout=[("Books",["Thriller","Drama","Adventure"]),("Movies",["1990s","1950s","1920s"]),("Applications",["Shopping","Casual","Beauty"])]
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":7,"axes.titlesize":8,"axes.titleweight":"bold","axes.labelsize":6.8,"xtick.labelsize":6.2,"ytick.labelsize":6.3,"axes.spines.top":False,"axes.spines.right":False,"axes.linewidth":.55,"figure.facecolor":"white","savefig.facecolor":"white"})
colors={"nmf":"#0072B2","svd":"#D55E00"}
fig,axs=plt.subplots(3,3,figsize=(8.2,6.3),sharey=True)
for row,(domain,categories) in enumerate(layout):
 for col,cat in enumerate(categories):
  if domain=="Books":
   frame=adv if cat=="Adventure" else book
   d=frame[(frame.genre==cat)&(frame.k==15)&(frame.n_fake<=500)]
  else:
   stem="movies_"+cat[:-1] if domain=="Movies" else cat.lower()
   d=pd.read_csv(N/(stem+"_top15_diagnostic_counts.csv"))
  ns=sorted(d.n_fake.unique());x=np.arange(len(ns))
  ax=axs[row,col]
  for j,model in enumerate(["nmf","svd"]):
   dm=d[d.model==model].set_index("n_fake")
   assert not dm.index.duplicated().any()
   vals=dm.loc[ns,"mean_target_count"].to_numpy()
   assert np.isfinite(vals).all() and ((vals>=0)&(vals<=15)).all()
   w=.34 if len(ns)>6 else .37
   ax.bar(x+(-w/2 if j==0 else w/2),vals,w,color=colors[model],edgecolor="white",linewidth=.35,zorder=3)
  ax.set_title(cat,loc="left",color="#183247",pad=4)
  ax.set_ylim(0,15);ax.set_yticks([0,5,10,15])
  ax.set_xticks(x,["Base" if n==0 else str(n) for n in ns],rotation=40,ha="right")
  ax.tick_params(axis="both",length=0,pad=2,colors="#52616B")
  ax.grid(axis="y",color="#E3E9ED",linewidth=.45,zorder=0);ax.set_axisbelow(True)
  ax.spines["bottom"].set_color("#AFBAC1")
 for_label=axs[row,0]
 for_label.annotate(domain,xy=(-.30,.5),xycoords="axes fraction",rotation=90,ha="center",va="center",fontsize=8,fontweight="bold",color="#183247")
axs[0,0].set_ylabel("Mean target items in Top-15",color="#183247",labelpad=4)
fig.legend(handles=[Patch(facecolor=colors[m],label=m.upper()) for m in ["nmf","svd"]],loc="upper center",bbox_to_anchor=(.57,.984),ncol=2,frameon=False,fontsize=7.5,handlelength=1,columnspacing=1.5)
fig.suptitle("Target-Category Items in Top-15 Recommendations",x=.5,y=.997,fontsize=10,fontweight="bold",color="#183247")
fig.subplots_adjust(left=.12,right=.99,top=.925,bottom=.105,hspace=.46,wspace=.25)
out=M/"figures/recommendation_mobile_shopping.png"
fig.savefig(out,dpi=600,bbox_inches="tight",pad_inches=.04)
print(out)
