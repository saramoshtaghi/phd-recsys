from pathlib import Path
import re,os
os.environ.setdefault('MPLCONFIGDIR','/tmp/thriller_powerpoint_mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path('/home/moshtasa/Research/phd-recsys/Book/current/0420')
SOURCES={
'NMF':ROOT/'NMF-0420/result/0420_NMF_bar_figures/nmf_thriller_analysis/NMF - Thriller.txt',
'SVD':ROOT/'SVD-0420/result/0420_SVD_bar_figures/SVD_thriller_analysis/SVD - Thriller.txt'}
DEST=ROOT/'SVD-0420/result/0420_SVD_bar_figures/SVD_thriller_analysis/comparision - Thriller_powerpoint.png'
def read_values(path):
 data={};k=None
 for line in path.read_text().splitlines():
  m=re.match(r'k\s*=\s*(\d+)',line.strip())
  if m:k=int(m[1]);data[k]={}
  m=re.match(r'\s*(original|\d+u):\s*([\d.]+)',line)
  if m and k:data[k][m[1]]=float(m[2])
 return data
parsed={m:read_values(p) for m,p in SOURCES.items()}
ks=[15,25,35];labels=['original','2u','4u','6u','25u','50u','100u','200u','500u','1000u','2000u']
assert all(set(parsed[m][k])==set(labels) for m in parsed for k in ks)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':15})
fig,axes=plt.subplots(1,2,figsize=(16,9),sharey=True)
navy='#142E49';muted='#546576'
colors=['#34495E','#4E79A7','#76B7B2','#59A14F','#B07AA1','#9C755F','#E15759','#F28E2B','#D6A300','#2A9DAB','#A0CBE8']
handles=[]
for ax,m in zip(axes,['NMF','SVD']):
 x=np.arange(3)*1.3;width=.098
 for i,(lab,col) in enumerate(zip(labels,colors)):
  vals=[parsed[m][k][lab] for k in ks]
  bars=ax.bar(x+(i-5)*width,vals,width=width*.94,color=col,edgecolor='white',linewidth=.5,zorder=3)
  if m=='NMF':handles.append(bars[0])
 ax.set_title(m,fontsize=23,color=navy,weight='bold',pad=16)
 ax.set_xticks(x);ax.set_xticklabels(['Top-15','Top-25','Top-35'],fontsize=17,color=navy)
 ax.set_xlabel('Recommendation list size',fontsize=15,color=muted,labelpad=16)
 ax.set_ylim(0,26);ax.set_yticks(np.arange(0,26,5));ax.tick_params(axis='y',labelsize=14,colors=muted,length=0)
 ax.grid(axis='y',color='#DCE3E9',linewidth=.7,zorder=0)
 for side in ['top','right','left']:ax.spines[side].set_visible(False)
 ax.spines['bottom'].set_color('#C0CBD5');ax.tick_params(axis='x',length=0,pad=10)
axes[0].set_ylabel('Mean Thriller books per real user',fontsize=16,color=navy,labelpad=14)
fig.subplots_adjust(left=.085,right=.975,bottom=.21,top=.72,wspace=.12)
fig.text(.055,.943,'Thriller recommendations after synthetic-user injection',fontsize=26,weight='bold',color=navy)
fig.text(.055,.892,'Baseline and 10 injection sizes • Top-15, Top-25 and Top-35 recommendations',fontsize=13,color=muted)
fig.legend(handles,['Baseline']+[f'{lab[:-1]} users' for lab in labels[1:]],loc='upper center',bbox_to_anchor=(.51,.856),ncol=6,frameon=False,fontsize=13,columnspacing=1.7,handlelength=1.3,handletextpad=.6)
fig.text(.055,.075,'Each bar averages the number of Thriller-tagged books in recommendations across 53,424 real users.',fontsize=13,color=navy)
fig.text(.055,.037,'Shared linear y-axis for direct model comparison. Synthetic users: Thriller ratings = 5; other-book ratings = 1.',fontsize=11.5,color=muted)
fig.savefig(DEST,dpi=240,facecolor='white')
print(DEST)
