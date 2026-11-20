from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
R=Path("/home/moshtasa/Research/phd-recsys")
M=R/"First_paper_material";N=M/"notebook"
T=R/"Book/current/PerturbRec/svd_nmf/layer1/tables"
book=pd.read_csv(T/"topk_target_union/summary.csv")
adv=pd.read_csv(T/"adventure_recommendation_summary.csv")
layout=[("Books",["Thriller","Drama","Adventure"]),("Movies",["1990s","1950s","1920s"]),("Applications",["Shopping","Casual","Beauty"])]
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":7,"axes.titlesize":7,"axes.titleweight":"bold","axes.labelsize":6.8,"xtick.labelsize":5.3,"ytick.labelsize":5.8,"axes.spines.top":False,"axes.spines.right":False,"axes.linewidth":.55,"figure.facecolor":"white","savefig.facecolor":"white"})
colors={"nmf":"#D55E00","svd":"#0072B2"}
fig,axs=plt.subplots(3,3,figsize=(6.0,3.6),sharey=False)
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
  ax.set_box_aspect(.36)
  panel_max=0.0
  for j,model in enumerate(["svd","nmf"]):
   dm=d[d.model==model].set_index("n_fake")
   assert not dm.index.duplicated().any()
   vals=dm.loc[ns,"mean_target_count"].to_numpy()
   assert np.isfinite(vals).all() and ((vals>=0)&(vals<=15)).all()
   panel_max=max(panel_max,float(vals.max()))
   w=.23
   ax.bar(x+(-w/2 if j==0 else w/2),vals,w,color=colors[model],edgecolor="white",linewidth=.35,zorder=3)
  ax.set_title(cat,loc="left",color="#183247",pad=4)
  # Give each category its own readable count scale, based on both models.
  # This keeps small-range categories (for example, 1920s movies) legible.
  ymax=max(0.5,panel_max*1.22)
  ax.set_ylim(0,ymax)
  ax.yaxis.set_major_locator(MaxNLocator(nbins=4,steps=[1,2,2.5,5,10],integer=panel_max>=1))
  ax.set_xticks(x,["Base" if n==0 else str(n) for n in ns],rotation=40,ha="right")
  ax.tick_params(axis="both",length=0,pad=2,colors="#52616B")
  ax.grid(axis="y",color="#E3E9ED",linewidth=.45,zorder=0);ax.set_axisbelow(True)
  ax.spines["bottom"].set_color("#AFBAC1")
 for_label=axs[row,0]
 for_label.annotate(domain,xy=(-.30,.5),xycoords="axes fraction",rotation=90,ha="center",va="center",fontsize=7,fontweight="bold",color="#183247")
fig.legend(handles=[Patch(facecolor=colors[m],label=m.upper()) for m in ["svd","nmf"]],loc="upper center",bbox_to_anchor=(.57,.984),ncol=2,frameon=False,fontsize=7,handlelength=1,columnspacing=1.5)
fig.suptitle("Top-15 Recommendation Exposure",x=.5,y=.997,fontsize=9,fontweight="bold",color="#183247")
fig.subplots_adjust(left=.13,right=.99,top=.88,bottom=.17,hspace=.48,wspace=.30)
out=M/"figures/recommendation_exposure_top15.png"
fig.savefig(out,dpi=600,bbox_inches="tight",pad_inches=.04)
print(out)
