"""48 x 36 inch scientific poster; verified stage-one results only."""
from pathlib import Path
import os
os.environ['MPLBACKEND'] = 'Agg'
import json, hashlib, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
from matplotlib.lines import Line2D
from matplotlib.font_manager import FontProperties

R = Path('/home/moshtasa/Research/phd-recsys')
M = R / 'First_paper_material'
L = R / 'Book/current/PerturbRec/svd_nmf'
DATA = M / 'data'
OUT = M / 'figures'
stats = json.loads((DATA/'dataset_overview_counts.json').read_text())
exposure = pd.read_csv(DATA/'recommendation_exposure_top15.csv')
check = pd.read_csv(L/'layer1/tables/topk_target_union/summary.csv')
book = exposure[exposure.dataset.eq('Books')]
for model in ['SVD','NMF']:
    a = book[book.model.eq(model)].sort_values('n_fake')
    b = check[(check.model.eq(model.lower())) & check.genre.eq('Thriller') & check.k.eq(15) & (check.n_fake<=500)].sort_values('n_fake')
    assert a.n_fake.tolist() == b.n_fake.tolist()
    assert np.allclose(a.mean_target_count,b.mean_target_count,rtol=0,atol=1e-12)
    assert (a.n_real_users==53424).all()
# Verify every mobile/movie value directly against its original text output.
for (domain,model), group in exposure[~exposure.dataset.eq('Books')].groupby(['dataset','model']):
    source=Path(group.source.iloc[0]); values={}; k=None
    for line in source.read_text().splitlines():
        match=re.match(r'\s*k\s*=\s*(\d+)',line)
        if match: k=int(match[1]); continue
        match=re.match(r'\s*(original|\d+u)\s*:\s*([0-9.]+)\s*\(users=(\d+)\)',line,re.I)
        if k==15 and match:
            n=0 if match[1].lower()=='original' else int(match[1][:-1])
            values[n]=(float(match[2]),int(match[3]))
    for row in group.itertuples():
        assert values[row.n_fake]==(row.mean_target_count,row.n_real_users)

raw = pd.read_csv(stats['Books']['source'],usecols=['book_id','genres'])
tags=raw.drop_duplicates('book_id').set_index('book_id').genres
assert len(tags)==9415
target=tags.map(lambda v:'Thriller' in {x.strip() for x in v.split(',')})
assert int(target.sum())==1606
counts=[0,2,4,6,25,50,100,200,500]
bias=[]
for model in ['svd','nmf']:
    for n in counts:
        p=L/'layer2/embeddings/pilot'/f'{model}_baseline_a.npz' if n==0 else L/'layer2/embeddings/controlled'/model/'thriller'/f'{model}_thriller_{n}.npz'
        with np.load(p) as z:
            mask=target.loc[z['items']].to_numpy()
            bias.append(dict(model=model.upper(),n_fake=n,target=float(z['bi'][mask].mean()),non_target=float(z['bi'][~mask].mean()),source=str(p)))
bias=pd.DataFrame(bias)

W,H=48,36
NAVY='#172B3A'; RED='#C8102E'; BG='#F4F6F8'; MUTED='#51616D'; GRID='#DFE6EB'
BLUE='#0072B2'; ORANGE='#D55E00'; TEAL='#007C78'
C={'SVD':BLUE,'NMF':ORANGE}
plt.rcParams.update({'font.family':'DejaVu Sans','text.color':NAVY,'axes.labelcolor':MUTED,'xtick.color':MUTED,'ytick.color':MUTED,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
fig=plt.figure(figsize=(W,H),dpi=100,facecolor=BG)
base=fig.add_axes([0,0,1,1]);base.set_xlim(0,W);base.set_ylim(H,0);base.axis('off')
texts=[]
def rect(x,y,w,h,color='white',radius=.16,edge=None):
    p=FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={radius}',facecolor=color,edgecolor=edge or color,lw=.8)
    base.add_patch(p);return p
def text(x,y,s,size=24,color=NAVY,weight='normal',ha='left',va='top',**kw):
    t=base.text(x,y,s,fontsize=size,color=color,fontweight=weight,ha=ha,va=va,linespacing=1.2,**kw);texts.append(t);return t
fig.canvas.draw(); renderer=fig.canvas.get_renderer()
def para(x,y,s,w,size=24,color=NAVY,weight='normal',leading=1.30):
    prop=FontProperties(family='DejaVu Sans',size=size,weight=weight)
    lines=[]
    for block in s.split('\n'):
        words=block.split();line=''
        for word in words:
            candidate=(line+' '+word).strip()
            if line and renderer.get_text_width_height_descent(candidate,prop,False)[0] > w*fig.dpi:
                lines.append(line);line=word
            else:line=candidate
        lines.append(line)
    text(x,y,'\n'.join(lines),size,color,weight)
    return y+len(lines)*size/72*leading
def heading(x,y,n,title,w):
    text(x,y,n,22,RED,'bold');text(x+.65,y-.04,title,31,NAVY,'bold')
    base.plot([x,x+w],[y+.64,y+.64],color=GRID,lw=1.4)
    return y+.95
def axis(x,y,w,h):return fig.add_axes([x/W,(H-y-h)/H,w/W,h/H],facecolor='white')

# Identity and a short, evidence-led message.
base.add_patch(Rectangle((0,0),W,.22,color=RED))
text(1,.72,'Synthetic Ratings,\nDifferent Recommendations',78,NAVY,'bold')
text(1,3.19,'Comparing SVD and NMF under controlled category-level injection',31,MUTED)
text(1,3.91,'Sara Moshtaghi Largani  |  Raj Bhatnagar  |  Vikram Ravindra',27,NAVY,'bold')
text(1,4.44,'Department of Computer Science, CEAS, University of Cincinnati',23,MUTED)
text(46.9,1.12,'UNIVERSITY OF',25,RED,'bold',ha='right')
text(46.9,1.69,'CINCINNATI',44,RED,'bold',ha='right')
text(46.9,2.48,'Recommender systems research',21,MUTED,ha='right')
rect(1,5.30,46,2.16,NAVY)
text(1.60,5.65,'Same injected profiles. Different model responses.',35,'white','bold')
text(1.60,6.31,'Books / Thriller / 500 synthetic users / Top-15',25,'#CFD9E0')
text(31.30,5.54,'NMF',23,'#FFB98F','bold')
text(31.30,6.00,'10.84',52,'white','bold')
text(37.7,5.54,'SVD',23,'#89CEEF','bold')
text(37.7,6.00,'2.28',52,'white','bold')
text(43.00,5.92,'Thriller books\nper list',22,'#CFD9E0')

LX,LW=1,11.85; CX,CW=13.45,20.55; RX,RW=34.60,12.40
# Left column: question, exact dataset counts and intervention.
rect(LX,8.05,LW,4.02)
y=heading(LX+.42,8.42,'01','Research question',LW-.84)
para(LX+.42,y,'How does the same repeated rating pattern change what SVD and NMF recommend to real users?',LW-.84,28,weight='bold')
para(LX+.42,10.65,'We measure category exposure: how many target-category items appear in each user\'s Top-15 list.',LW-.84,24)

rect(LX,12.48,LW,8.26)
y=heading(LX+.42,12.85,'02','Three real datasets',LW-.84)
cards=[('BOOKS','53,424 users','9,415 books  |  5,815,569 ratings','13 assigned genres; overlapping labels.'),('MOVIES','943 users','1,682 movies  |  100,000 ratings','MovieLens 100K; target: 1990s releases.'),('MOBILE APPS','700,111 users','10,173 apps  |  19,297,019 ratings','48 original categories; target: Shopping.')]
for j,(name,users,detail,note) in enumerate(cards):
    yy=14.15+j*1.94
    text(LX+.45,yy,name,20,TEAL,'bold')
    text(LX+.45,yy+.40,users,30,NAVY,'bold')
    text(LX+.45,yy+.98,detail,21)
    text(LX+.45,yy+1.40,note,19,MUTED)
    if j<2:base.plot([LX+.45,LX+LW-.45],[yy+1.78]*2,color=GRID,lw=.8)

rect(LX,21.16,LW,12.48)
y=heading(LX+.42,21.53,'03','Controlled injection',LW-.84)
para(LX+.42,y,'Every fake user rates the existing catalog using the same rule:',LW-.84,24)
rect(LX+.45,23.78,5.15,1.42,'#FFF0E6')
text(LX+.70,24.02,'TARGET BOOKS',19,ORANGE,'bold');text(LX+.70,24.43,'rating 5',32,ORANGE,'bold')
rect(LX+5.90,23.78,5.50,1.42,'#EAF3F8')
text(LX+6.15,24.02,'ALL OTHER BOOKS',19,BLUE,'bold');text(LX+6.15,24.43,'rating 1',32,BLUE,'bold')
text(LX+.45,25.64,'Thriller example: rating rule (schematic)',21,NAVY,'bold')
mx= LX+.45; my=26.25;cellw=1.27
for j in range(8):
    is_target=j<3
    rect(mx+j*cellw,my,1.12,.80,'#FFE0CC' if is_target else '#D6EAF5',radius=.06)
    text(mx+j*cellw+.56,my+.13,'5' if is_target else '1',30,ORANGE if is_target else BLUE,'bold',ha='center')
text(LX+.45,27.32,'1,606 Thriller + 7,809 non-Thriller = 9,415 ratings',20,MUTED)
para(LX+.45,28.10,'500 fake users add 4,707,500 ratings: 80.9% of the original book rating count.',LW-.9,25,NAVY,'bold')
para(LX+.45,29.72,'Original users, ratings and books stay unchanged. Fake users enter training only; recommendations are evaluated for real users.',LW-.9,23)
text(LX+.45,31.55,'SHARED TRAINING SETTINGS',19,TEAL,'bold')
para(LX+.45,32.02,'8 latent factors  |  180 epochs\nBiased Surprise implementations; seed 42.',LW-.9,21)

# Centre column: three comparable recommendation-count charts.
rect(CX,8.05,CW,25.59)
y=heading(CX+.47,8.42,'04','What changes in Top-15?',CW-.94)
text(CX+.50,9.52,'Mean target-category items per recommendation list',24,MUTED)
fig.legend(handles=[Line2D([0],[0],color=BLUE,lw=8,label='SVD'),Line2D([0],[0],color=ORANGE,lw=8,label='NMF')],loc='upper right',bbox_to_anchor=(33.3/W,1-9.32/H),frameon=False,ncol=2,fontsize=23,handlelength=1.2,columnspacing=1.5)
configs=[('Books','Thriller',500,10.95),('Mobile apps','Shopping',300,18.30),('Movies','1990s',50,25.65)]
for idx,(domain,targetname,limit,top) in enumerate(configs):
    labels=sorted(exposure[exposure.dataset.eq(domain)].n_fake.unique())
    title=f'{chr(65+idx)}  {domain}: {targetname}'
    text(CX+.53,top-.18,title,28,NAVY,'bold')
    ax=axis(CX+1.15,top+.55,CW-1.95,4.60)
    xx=np.arange(len(labels))
    for j,model in enumerate(['SVD','NMF']):
        d=exposure[(exposure.dataset.eq(domain)) & exposure.model.eq(model)].set_index('n_fake').loc[labels]
        values=d.mean_target_count.to_numpy();offset=-.19 if j==0 else .19
        ax.bar(xx+offset,values,width=.36,color=C[model],zorder=3)
        for k,v in enumerate(values):
            ax.text(k+offset,v+.20,f'{v:.2f}',ha='center',va='bottom',fontsize=18.5,color=C[model],fontweight='bold' if k==len(values)-1 else 'normal')
    ax.set_ylim(0,16.5);ax.set_yticks([0,5,10,15]);ax.set_xticks(xx,[str(n) for n in labels]);ax.tick_params(length=0,labelsize=22,pad=9)
    ax.set_ylabel('Mean items',fontsize=22,labelpad=11)
    ax.set_xlabel('Injected users (0 = baseline)',fontsize=22,labelpad=11)
    ax.grid(axis='y',color=GRID,lw=1);ax.set_axisbelow(True);ax.spines['left'].set_visible(False);ax.spines['bottom'].set_color(GRID)
text(CX+.55,32.40,'All panels count items out of 15. Injection counts are categorical.',20,MUTED)
text(CX+.55,32.88,'Counts and rating budgets differ across datasets.',20,MUTED)
text(CX+.55,33.29,'The movie target already has high baseline exposure.',18,MUTED)

# Right column: shared prediction and an interpretable stage-one parameter plot.
rect(RX,8.05,RW,7.36)
y=heading(RX+.43,8.42,'05','Two factorization models',RW-.86)
text(RX+.48,9.70,r'$\hat r_{ui}=\mu+b_u+b_i+p_u^\top q_i$',34,NAVY)
para(RX+.48,10.64,'Predicted rating = global mean + user bias + item bias + user-item dot product.',RW-.96,23)
text(RX+.48,12.16,'SVD',26,BLUE,'bold')
para(RX+2.02,12.18,'Signed user and item factors; additive gradient updates.',RW-2.50,23)
text(RX+.48,13.44,'NMF',26,ORANGE,'bold')
para(RX+2.02,13.46,'Non-negative factors; multiplicative factor updates. Biases can be negative.',RW-2.50,23)

rect(RX,15.82,RW,9.49)
y=heading(RX+.43,16.20,'06','A closer look at item bias',RW-.86)
text(RX+.48,17.32,r'Books / Thriller injection / learned $b_i$',22,MUTED)
ax=axis(RX+1.05,18.05,RW-1.80,4.40)
xx=np.arange(len(counts))
for model in ['SVD','NMF']:
    d=bias[bias.model.eq(model)].set_index('n_fake').loc[counts]
    for key,ls,marker in [('target','-','o'),('non_target','--','s')]:
        ax.plot(xx,d[key],color=C[model],ls=ls,marker=marker,ms=7,lw=3.5)
ax.axhline(0,color='#A1AFB8',lw=1)
ax.set_xticks(xx,[str(n) for n in counts]);ax.tick_params(labelsize=18,length=0,pad=8)
ax.set_ylabel('Mean item bias',fontsize=22,labelpad=8);ax.set_xlabel('Injected users',fontsize=21,labelpad=9)
ax.grid(axis='y',color=GRID,lw=1);ax.set_axisbelow(True);ax.spines['left'].set_visible(False);ax.spines['bottom'].set_color(GRID)
ax.legend(handles=[Line2D([0],[0],color=MUTED,lw=3,label='Thriller'),Line2D([0],[0],color=MUTED,lw=3,ls='--',label='Non-Thriller')],loc='upper left',frameon=False,fontsize=18)
para(RX+.48,23.32,'NMF shows a larger target / non-target item-bias separation. This accompanies its stronger Thriller exposure.',RW-.96,23)
text(RX+.48,24.71,'Each book has equal weight in this bias panel.',18,MUTED)

rect(RX,25.72,RW,7.92)
y=heading(RX+.43,26.10,'07','What we learn',RW-.86)
items=[('Exposure changes.','Repeated synthetic ratings reshape the recommendations received by real users.'),('The model matters.','NMF amplifies Thriller more; SVD has higher Shopping exposure in the displayed mobile range.'),('The result is diagnostic.','These runs measure exposure and fitted parameters, not held-out accuracy or a validated defense.')]
yy=27.34
for title,body in items:
    text(RX+.48,yy,title,25,NAVY,'bold');yy=para(RX+.48,yy+.53,body,RW-.96,22)+.45

# Provenance and scope, with no unsupported claims about improved quality.
base.plot([1,47],[34.15,34.15],color='#BFCBD3',lw=1.2)
text(1,34.48,'DATA: Goodbooks-10K  |  MovieLens 100K  |  MobileRec',20,NAVY,'bold')
text(1,35.03,'Book genres: GPT-3.5-assigned labels; exposure is conditional on these labels. Baseline = no synthetic users.',18,MUTED)
text(47,34.48,'CONTACT',18,RED,'bold',ha='right')
text(47,34.96,'moshtasa@mail.uc.edu',23,NAVY,ha='right')
text(47,35.48,'48 x 36 in  |  Stage-one results',16,MUTED,ha='right')

# Validate the final summary and the bounds of poster-level text.
book_end=book[book.n_fake.eq(500)].set_index('model').mean_target_count
assert round(book_end['NMF'],2)==10.84 and round(book_end['SVD'],2)==2.28
assert round(100*(500*9415)/5815569,1)==80.9
fig.canvas.draw(); renderer=fig.canvas.get_renderer()
outside=[]
for t in texts:
    box=t.get_window_extent(renderer)
    if box.x0<0 or box.y0<0 or box.x1>W*fig.dpi or box.y1>H*fig.dpi:
        outside.append(t.get_text())
assert not outside,outside
pdf=OUT/'poster_svd_nmf_stage1.pdf';png=OUT/'poster_svd_nmf_stage1.png'
fig.savefig(pdf,facecolor=BG)
fig.savefig(png,dpi=150,facecolor=BG)
fig.savefig(OUT/'poster_svd_nmf_stage1_preview.png',dpi=55,facecolor=BG)
plt.close(fig)
manifest={'poster_size_inches':[48,36],'scope':'stage one; exposure and learned item biases','book_baseline_and_n500':book[book.n_fake.isin([0,500])].to_dict('records'),'dataset_counts':stats,'bias_equal_weight_per_item':bias.to_dict('records'),'no_new_training':True,'book_values_crosschecked_against_current_topk_summary':True,'mobile_movie_values_crosschecked_against_original_outputs':True,'claim_of_accuracy_improvement':False,'pdf':str(pdf),'png':str(png),'source_table_sha256':hashlib.sha256((DATA/'recommendation_exposure_top15.csv').read_bytes()).hexdigest()}
(M/'notebook/poster_svd_nmf_stage1_validation.json').write_text(json.dumps(manifest,indent=2))
print('SAVED',pdf,png,flush=True)
