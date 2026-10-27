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
NAVY='#142C3A'; RED='#C8102E'; BG='#F1F4F6'; MUTED='#526571'; GRID='#DBE4E9'
BLUE='#0072B2'; ORANGE='#D55E00'; TEAL='#007B77'; C={'SVD':BLUE,'NMF':ORANGE}
plt.rcParams.update({'font.family':'DejaVu Sans','text.color':NAVY,'axes.labelcolor':MUTED,'xtick.color':MUTED,'ytick.color':MUTED,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
fig=plt.figure(figsize=(W,H),dpi=100,facecolor=BG)
base=fig.add_axes([0,0,1,1]);base.set_xlim(0,W);base.set_ylim(H,0);base.axis('off');texts=[]
def rect(x,y,w,h,color='white',radius=.2,edge=None):
    p=FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={radius}',facecolor=color,edgecolor=edge or color,lw=1);base.add_patch(p);return p
def text(x,y,s,size=24,color=NAVY,weight='normal',ha='left',va='top',**kw):
    t=base.text(x,y,s,fontsize=size,color=color,fontweight=weight,ha=ha,va=va,linespacing=1.15,**kw);texts.append(t);return t
fig.canvas.draw();renderer=fig.canvas.get_renderer()
def para(x,y,s,w,size=24,color=NAVY,weight='normal',leading=1.28):
    prop=FontProperties(family='DejaVu Sans',size=size,weight=weight);lines=[]
    for block in s.split('\n'):
        line=''
        for word in block.split():
            new=(line+' '+word).strip()
            if line and renderer.get_text_width_height_descent(new,prop,False)[0]>w*fig.dpi:lines.append(line);line=word
            else:line=new
        lines.append(line)
    text(x,y,'\n'.join(lines),size,color,weight);return y+len(lines)*size/72*leading
def axis(x,y,w,h):return fig.add_axes([x/W,(H-y-h)/H,w/W,h/H],facecolor='white')
def tidy(ax):
    ax.grid(axis='y',color=GRID,lw=1.1);ax.set_axisbelow(True);ax.spines['left'].set_visible(False);ax.spines['bottom'].set_color(GRID);ax.tick_params(length=0,pad=10,labelsize=23)
def arrow(x1,y1,x2,y2,color=MUTED):base.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=26,lw=2.5,color=color))

# A large headline; all supporting copy serves the measured result.
base.add_patch(Rectangle((0,0),48,4.60,color=NAVY))
base.add_patch(Rectangle((0,0),.26,4.60,color=RED))
text(1,.48,'CONTROLLED BIAS IN RECOMMENDER SYSTEMS',21,'#92BDCB','bold')
text(1,1.07,'Fake ratings change\nreal recommendations',73,'white','bold')
text(1,3.34,'Sara Moshtaghi Largani  |  Raj Bhatnagar  |  Vikram Ravindra',26,'white','bold')
text(1,3.92,'Computer Science, CEAS, University of Cincinnati',23,'#CAD8DF')
text(46.90,.65,'UNIVERSITY OF',23,'#CAD8DF','bold',ha='right')
text(46.90,1.14,'CINCINNATI',40,'white','bold',ha='right')
text(46.90,2.74,'Same synthetic profiles.\nDifferent SVD and NMF responses.',27,'white',ha='right')

# A visually explicit count out of fifteen, rather than a large paragraph.
text(1,5.14,'THE BOOK RESULT',20,RED,'bold')
text(1,5.71,'At 500 fake profiles, NMF recommends far more Thriller.',36,NAVY,'bold')
text(1,6.44,'Mean target books in each real user\'s Top-15 list',25,MUTED)
for x,model,value in [(26.35,'SVD',2.279556004791854),(37.0,'NMF',10.839248277927522)]:
    rect(x,5.10,9.95,2.22,'white')
    text(x+.35,5.35,model,23,C[model],'bold')
    text(x+9.55,5.32,f'{value:.2f} / 15',40,C[model],'bold',ha='right')
    bx=x+.35;by=6.35;ww=9.20
    base.add_patch(Rectangle((bx,by),ww,.34,facecolor='#E4EBEF',edgecolor='none'))
    base.add_patch(Rectangle((bx,by),ww*value/15,.34,facecolor=C[model],edgecolor='none'))
    text(x+.35,6.91,f'{100*value/15:.1f}% of recommendation slots',18,MUTED)

# Dataset facts form a single band, not a separate mini-essay.
cards=[(1,'BOOKS','53,424','9,415','5,815,569','13 assigned genres'),(16.5,'MOVIES','943','1,682','100,000','Release-decade categories'),(32,'MOBILE APPS','700,111','10,173','19,297,019','48 original categories')]
for x,label,users,items,ratings,note in cards:
    rect(x,7.88,15,2.30,'white')
    text(x+.38,8.13,label,21,TEAL,'bold')
    text(x+.38,8.65,users,34,NAVY,'bold');text(x+5.70,8.79,'real users',21,MUTED)
    text(x+.38,9.34,f'{items} items  |  {ratings} ratings',22,NAVY)
    text(x+14.60,8.20,note,17,MUTED,ha='right')

# A four-step diagram replaces most of the method prose.
rect(1,10.73,46,3.55,'white')
text(1.45,11.03,'HOW THE EXPERIMENT WORKS',21,RED,'bold')
stages=[(1.50,'01','Create fake profiles','Target items = 5\nAll other items = 1'),(13.2,'02','Append their ratings','Keep real users, ratings\nand the catalog unchanged'),(24.9,'03','Train both models','Biased Surprise SVD / NMF\n8 factors; 180 epochs; seed 42'),(36.60,'04','Measure real-user output','Rank unseen items\nCount target items in Top-15')]
for x,n,title,body in stages:
    text(x,11.69,n,22,RED,'bold');text(x+.75,11.67,title,26,NAVY,'bold')
    text(x+.75,12.30,body,23,MUTED)
for x in [12.45,24.15,35.85]:arrow(x-.38,12.48,x+.32,12.48)
text(1.50,13.51,'Dense profiles: 500 fake book users add 4,707,500 ratings - 80.9% of the original book rating count.',22,NAVY,'bold')

# The book trajectory is the main evidence and gets most of the visual space.
rect(1,14.88,29.85,9.17,'white')
text(1.48,15.28,'01  BOOKS / THRILLER',29,NAVY,'bold')
text(1.48,15.93,'The response diverges as more identical profiles are injected.',24,MUTED)
ax=axis(2.05,17.18,26.75,5.38)
doses=sorted(book.n_fake.unique());xx=np.arange(len(doses))
for model in ['SVD','NMF']:
    d=book[book.model.eq(model)].set_index('n_fake').loc[doses]
    ax.plot(xx,d.mean_target_count,color=C[model],lw=5,marker='o',ms=12,label=model)
    v=d.mean_target_count.iloc[-1]
    ax.annotate(f'{v:.2f}',(xx[-1],v),xytext=(12,2),textcoords='offset points',fontsize=27,fontweight='bold',color=C[model],va='center')
ax.set_xlim(-.12,len(doses)-.50);ax.set_ylim(0,15);ax.set_yticks([0,5,10,15]);ax.set_xticks(xx,[str(n) for n in doses]);ax.set_xlabel('Injected users (0 = baseline)',fontsize=24,labelpad=14);ax.set_ylabel('Mean Thriller books / Top-15',fontsize=24,labelpad=16);tidy(ax)
ax.legend(loc='upper left',ncol=2,frameon=False,fontsize=24,handlelength=2)
text(1.48,23.22,'Baseline: NMF 1.73 / 15   |   SVD 1.58 / 15',23,MUTED)

# Cross-domain checks make clear that there is no universal model ranking.
for x,domain,targetname,num in [(1,'Mobile apps','Shopping','02'),(16.30,'Movies','1990s','03')]:
    rect(x,24.63,14.55,7.36,'white')
    text(x+.43,25.02,f'{num}  {domain.upper()} / {targetname.upper()}',25,NAVY,'bold')
    ax=axis(x+1.05,26.03,12.45,4.27)
    doses=sorted(exposure[exposure.dataset.eq(domain)].n_fake.unique());xx=np.arange(len(doses))
    for j,model in enumerate(['SVD','NMF']):
        d=exposure[exposure.dataset.eq(domain)&exposure.model.eq(model)].set_index('n_fake').loc[doses]
        off=-.19 if j==0 else .19
        ax.bar(xx+off,d.mean_target_count,width=.36,color=C[model])
        for k,v in enumerate(d.mean_target_count):ax.text(k+off,v+.20,f'{v:.2f}',ha='center',fontsize=18,color=C[model],fontweight='bold' if k==len(d)-1 else 'normal')
    ax.set_ylim(0,16.7);ax.set_yticks([0,5,10,15]);ax.set_xticks(xx,[str(n) for n in doses]);tidy(ax);ax.tick_params(labelsize=21)
    ax.set_xlabel('Injected users',fontsize=22,labelpad=10);ax.set_ylabel('Target items / Top-15',fontsize=21,labelpad=10)

# The parameter plot is secondary evidence, clearly labelled as such.
rect(31.45,14.88,15.55,6.78,'white')
text(31.93,15.28,'SAME PREDICTION FORMULA',23,RED,'bold')
text(31.93,16.03,r'$\hat r_{ui}=\mu+b_u+b_i+p_u^\top q_i$',39,NAVY)
para(31.93,17.04,'Global mean + user bias + book bias + user-item dot product.',14.50,24,MUTED)
text(31.93,18.25,'SVD',26,BLUE,'bold');para(34.07,18.29,'Signed latent factors; additive gradient updates.',12.20,24)
text(31.93,19.45,'NMF',26,ORANGE,'bold');para(34.07,19.49,'Non-negative factors; multiplicative factor updates. Biases can be negative.',12.20,24)
rect(31.45,22.24,15.55,9.75,'white')
text(31.93,22.65,'WHAT CHANGES IN BOOK BIAS?',25,NAVY,'bold')
text(31.93,23.21,'Thriller injection; equal weight per book in this mean',20,MUTED)
ax=axis(32.55,24.30,13.30,4.70);xx=np.arange(len(counts))
for model in ['SVD','NMF']:
    d=bias[bias.model.eq(model)].set_index('n_fake').loc[counts]
    for key,ls in [('target','-'),('non_target','--')]:ax.plot(xx,d[key],ls=ls,color=C[model],lw=3.8,marker='o' if key=='target' else 's',ms=7)
ax.axhline(0,color='#A1AFB8',lw=1);ax.set_xticks(xx,[str(n) for n in counts]);ax.set_ylabel(r'Mean $b_i$',fontsize=23,labelpad=10);ax.set_xlabel('Injected users',fontsize=22,labelpad=10);tidy(ax);ax.tick_params(labelsize=19)
ax.legend(handles=[Line2D([0],[0],color=MUTED,lw=3,label='Thriller'),Line2D([0],[0],color=MUTED,lw=3,ls='--',label='Non-Thriller')],frameon=False,fontsize=20,loc='upper left')
para(31.93,30.15,'NMF develops a larger book-bias separation between target and non-target books, alongside stronger Thriller exposure.',14.50,24,NAVY,'bold')

# A single final message, with tightly scoped interpretation.
rect(1,32.58,46,1.65,NAVY)
text(1.50,32.91,'TAKEAWAY',21,'#92BDCB','bold')
text(7.00,32.86,'Identical synthetic ratings can produce very different exposure shifts.',31,'white','bold')
text(7.00,33.46,'The response depends on the model, target category and injection budget.',23,'#D5E0E7')
text(1,34.69,'SCOPE',18,RED,'bold')
para(2.35,34.66,'Exposure and fitted parameters, not held-out accuracy. Movie target has high baseline exposure; user counts and rating budgets differ across domains.',32,18,MUTED)
text(1,35.46,'DATA: Goodbooks-10K  |  MovieLens 100K  |  MobileRec. Book genre labels assigned with GPT-3.5.',17,MUTED)
text(47,34.70,'moshtasa@mail.uc.edu',24,NAVY,'bold',ha='right')
text(47,35.40,'University of Cincinnati  |  Stage-one study',17,MUTED,ha='right')

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
