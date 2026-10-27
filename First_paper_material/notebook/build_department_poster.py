"""Poster using the supplied department poster's four-column red-bar template."""
from pathlib import Path
import os
os.environ['MPLBACKEND']='Agg'
import json,re,hashlib
import numpy as np,pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,FancyArrowPatch
from matplotlib.lines import Line2D
from matplotlib.font_manager import FontProperties
from PIL import Image

R=Path('/home/moshtasa/Research/phd-recsys');M=R/'First_paper_material';NB=M/'notebook';OUT=M/'figures';L=R/'Book/current/PerturbRec/svd_nmf/layer1'
stats=json.loads((M/'data/dataset_overview_counts.json').read_text())
cats=json.loads((M/'data/dataset_category_counts.json').read_text())
book=pd.read_csv(L/'tables/topk_target_union/summary.csv')
book=book[(book.genre=='Thriller')&(book.k==15)].sort_values(['model','n_fake'])
assert not book.duplicated(['model','n_fake']).any()
records=[]
for row in book.itertuples():
 records.append(dict(dataset='Books',model=row.model.upper(),n_fake=int(row.n_fake),mean=float(row.mean_target_count),sign_ok=bool(row.nmf_nonnegative_constraint_ok),source=str(L/'tables/topk_target_union/summary.csv')))
for domain in ['Mobile apps','Movies']:
 for model in ['SVD','NMF']:
  p=R/f'Mobile/archive/0405/0405_{model}/result/{model}_bar_figures/{model.lower()}_shopping_analysis/{model} - Shopping.txt' if domain=='Mobile apps' else R/f'Movie /current/0519/{model}_0519/result/0519_{model}_bar_figures/{model.lower()}_decade_1990_analysis/{model} - Decade 1990.txt'
  k=None
  for line in p.read_text().splitlines():
   z=re.match(r'\s*k\s*=\s*(\d+)',line)
   if z:k=int(z[1]);continue
   z=re.match(r'\s*(original|\d+u)\s*:\s*([0-9.]+)\s*\(users=(\d+)\)',line,re.I)
   if k==15 and z:
    assert int(z[3])==stats[domain]['users']
    records.append(dict(dataset=domain,model=model,n_fake=0 if z[1].lower()=='original' else int(z[1][:-1]),mean=float(z[2]),sign_ok=True,source=str(p)))
ex=pd.DataFrame(records)
assert len(ex)==50 and not ex.duplicated(['dataset','model','n_fake']).any()
components=pd.read_csv(NB/'three_genre_panel_independent_components.csv')
components=components[(components.genre=='Thriller')&(components.n_fake==500)]
for row in components.itertuples():assert np.isclose(row.mean_prediction,row.mu+row.mean_bu+row.mean_bi+row.mean_interaction,atol=1e-10)
def comp(model,view,group):return components[(components.model==model.lower())&(components.view==view)&(components.group==group)].iloc[0]

W,H=48,36;RED='#CF102D';DARK='#111111';MUTED='#4C5861';GRID='#DFE3E6';BLUE='#0072B2';ORANGE='#D55E00';GREEN='#008C6C';COL={'SVD':BLUE,'NMF':ORANGE}
plt.rcParams.update({'font.family':'DejaVu Sans','text.color':DARK,'axes.labelcolor':DARK,'xtick.color':MUTED,'ytick.color':MUTED,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig=plt.figure(figsize=(W,H),dpi=100,facecolor='white');base=fig.add_axes([0,0,1,1]);base.set_xlim(0,W);base.set_ylim(H,0);base.axis('off');texts=[]
def text(x,y,s,size=22,color=DARK,weight='normal',ha='left',**kw):
 t=base.text(x,y,s,fontsize=size,color=color,fontweight=weight,ha=ha,va='top',linespacing=1.18,**kw);texts.append(t);return t
fig.canvas.draw();renderer=fig.canvas.get_renderer()
def para(x,y,s,w,size=22,color=DARK,weight='normal',leading=1.32):
 prop=FontProperties(family='DejaVu Sans',size=size,weight=weight);lines=[]
 for part in s.split('\n'):
  line=''
  for word in part.split():
   trial=(line+' '+word).strip()
   if line and renderer.get_text_width_height_descent(trial,prop,False)[0]>w*fig.dpi:lines.append(line);line=word
   else:line=trial
  lines.append(line)
 text(x,y,'\n'.join(lines),size,color,weight);return y+len(lines)*size/72*leading
def bar(x,y,w,title,size=25):
 base.add_patch(Rectangle((x,y),w,.52,color=RED));text(x+w/2,y+.09,title,size,'white','bold',ha='center')
def axis(x,y,w,h):return fig.add_axes([x/W,(H-y-h)/H,w/W,h/H],facecolor='white')
def tidy(ax,size=19):
 ax.grid(axis='y',color=GRID,lw=.8);ax.set_axisbelow(True);ax.spines['left'].set_visible(False);ax.spines['bottom'].set_color('#AAB4BB');ax.tick_params(length=0,labelsize=size,pad=7)
def photo(x,y,w,h,name):
 base.imshow(Image.open(NB/name),extent=(x,x+w,y+h,y),aspect='auto',zorder=4)

# Reuse the actual decorative header and UC wordmark from the user's template.
base.imshow(Image.open(NB/'poster_header_reference.png'),extent=(0,48,5.333,0),aspect='auto')
text(19.65,.73,'Training Recommender Systems with Biasing Data:\nA Comparison of SVD and NMF',62,'white','bold',ha='center')
text(19.65,3.30,'Sara Moshtaghi Largani, Raj Bhatnagar, and Vikram Ravindra',32,'white','bold',ha='center')
text(19.65,4.03,'Department of Computer Science, University of Cincinnati',27,'white','bold',ha='center')
X=[.45,12.35,24.25,36.15];CW=11.40;PAD=.16

# COLUMN 1: introduce all three datasets before any result is shown.
x=X[0];bar(x,5.62,CW,'1. Three Datasets')
para(x+PAD,6.38,'Recommenders learn from user-item ratings to rank unseen items. We study how added synthetic ratings change category exposure for real users.',CW-2*PAD,23)
datasets=[('Books: Goodbooks-10K','Books',8.28,'13 assigned genres; books may have two labels.'),('Movies: MovieLens 100K','Movies',12.26,'Release decades are the target categories.'),('Mobile Apps: MobileRec','Mobile apps',16.24,'48 original categories; Shopping is the example target.')]
for title,key,y,note in datasets:
 bar(x,y,CW,title,23)
 s=stats[key];text(x+PAD,y+.82,f"{s['users']:,} real users",29,DARK,'bold')
 text(x+PAD,y+1.52,f"{s['items']:,} items",26)
 text(x+PAD,y+2.14,f"{s['observed_ratings']:,} observed ratings",24)
 para(x+PAD,y+2.84,note,CW-2*PAD,21,MUTED)
bar(x,20.30,CW,'Dataset Scale',23)
ax=axis(x+.92,21.32,CW-1.28,4.35)
names=['Books','Movies','Mobile apps'];xx=np.arange(3)
for j,(key,label,color) in enumerate([('users','Users','#0072B2'),('items','Items','#E69F00'),('observed_ratings','Ratings','#009E73')]):
 vals=[stats[n][key] for n in names];ax.bar(xx+(j-1)*.24,vals,width=.22,color=color,label=label)
ax.set_yscale('log');ax.set_ylim(500,5e7);ax.set_xticks(xx,['Books','Movies','Mobile']);ax.set_ylabel('Count (log scale)',fontsize=20);tidy(ax,18);ax.legend(frameon=False,fontsize=18,ncol=3,loc='upper left')
bar(x,26.60,CW,'Book Genres After Labeling',23)
ax=axis(x+3.50,27.62,CW-3.85,6.25)
genres=cats['books']['counts'];labels=list(genres);values=list(genres.values())
ax.barh(range(len(labels)),values,color=[RED if v=='Thriller' else '#6395B3' for v in labels],height=.64)
ax.set_yticks(range(len(labels)),labels);ax.invert_yaxis();ax.set_xlim(0,3500);ax.set_xlabel('Unique books',fontsize=19);tidy(ax,17);ax.grid(axis='y',visible=False);ax.grid(axis='x',color=GRID,lw=.7)
for i,v in enumerate(values):ax.text(v+35,i,f'{v:,}',va='center',fontsize=15)
para(x+PAD,34.62,'Ratings are 1-5. Book genres were assigned with GPT-3.5; exposure measurements depend on these labels.',CW-2*PAD,17,MUTED)

# COLUMN 2: full stored injection ranges, consistently using Top-15.
x=X[1];bar(x,5.62,CW,'2. Results Across All Three Datasets',23)
para(x+PAD,6.38,'Mean number of target-category items in each real user\'s Top-15. Zero is the baseline; each point is a separate training run.',CW-2*PAD,22)
fig.legend(handles=[Line2D([0],[0],color=BLUE,lw=7,label='SVD'),Line2D([0],[0],color=ORANGE,lw=7,label='NMF')],loc='upper center',bbox_to_anchor=((x+CW/2)/W,1-7.78/H),frameon=False,ncol=2,fontsize=21)
for domain,target,y in [('Books','Thriller',8.70),('Mobile apps','Shopping',16.40),('Movies','1990s',24.10)]:
 bar(x,y,CW,domain+': '+target,23)
 d=ex[ex.dataset==domain];ns=sorted(d.n_fake.unique());xx=np.arange(len(ns));ax=axis(x+.78,y+1.22,CW-1.08,4.75)
 for j,model in enumerate(['SVD','NMF']):
  v=d[d.model==model].set_index('n_fake').loc[ns];off=-.19 if j==0 else .19
  for k,row in enumerate(v.itertuples()):
   valid=bool(row.sign_ok);ax.bar(k+off,row.mean,width=.36,color=COL[model] if valid else 'white',edgecolor=COL[model],hatch=None if valid else '////',lw=.9)
   if k==len(ns)-1 or (domain!='Books' and k==len(ns)-2):
    ax.text(k+off,row.mean+.22,f'{row.mean:.2f}'+('†' if not valid else ''),ha='center',va='bottom',color=COL[model],fontsize=16,fontweight='bold')
 ax.set_ylim(0,16.7);ax.set_yticks([0,5,10,15]);ax.set_xticks(xx,[str(n) for n in ns],rotation=55 if len(ns)>8 else 35,ha='right');ax.set_xlabel('Injected users',fontsize=20,labelpad=10);ax.set_ylabel('Target items / Top-15',fontsize=20,labelpad=8);tidy(ax,17 if len(ns)>8 else 19)
 v0=d[d.n_fake==0].set_index('model')['mean'];text(x+PAD,y+6.94,f"Baseline: SVD {v0['SVD']:.2f}; NMF {v0['NMF']:.2f}",17,MUTED)
text(x+PAD,31.52,'Full displayed ranges',22,DARK,'bold')
para(x+PAD,32.08,'Books: 0-3,000; mobile: 0-7,000; movies: 0-50. Injection counts are categorical. Rating budgets differ across datasets.',CW-2*PAD,20)
para(x+PAD,33.40,'† Hatched book NMF bars are raw outputs from checkpoints that failed the factor non-negativity check (1,000, 2,000, 2,500 and 3,000). The n=500 comparison passes this check.',CW-2*PAD,17,RED)

# COLUMN 3: define the two models and then hold the book dose fixed at 500.
x=X[2];bar(x,5.62,CW,'3. Understanding SVD and NMF',24)
text(x+PAD,6.45,'Both models predict ratings with:',23,DARK,'bold')
text(x+CW/2,7.13,r'$\hat r_{ui}=\mu+b_u+b_i+p_u^\top q_i$',32,DARK,ha='center')
definitions=[(r'$\mu$','Global mean of the training ratings.'),(r'$b_u$','Learned user bias: one offset per user.'),(r'$b_i$','Learned book bias: one offset per book.'),(r'$p_u^\top q_i$','Dot product of the learned user and book vectors.')]
for j,(symbol,meaning) in enumerate(definitions):
 yy=8.10+j*.77;text(x+.23,yy,symbol,25,BLUE,'bold');para(x+2.08,yy+.02,meaning,CW-2.32,21)
bar(x,11.65,CW,'SVD: Singular Value Decomposition',23)
para(x+PAD,12.45,'User and item vectors can contain positive and negative values. Additive gradient updates adjust the factors and biases to reduce rating prediction errors.',CW-2*PAD,23)
bar(x,14.55,CW,'NMF: Non-Negative Matrix Factorization',23)
para(x+PAD,15.35,'Factors are intended to remain non-negative and are updated multiplicatively. User and item biases are signed and updated additively.',CW-2*PAD,23)
bar(x,17.45,CW,'The Same Injection Rule',23)
para(x+PAD,18.21,'Each fake user rates all 1,606 Thriller books 5 and all 7,809 other books 1. No books are added.',CW-2*PAD,23)
para(x+PAD,19.63,'At n=500: 4,707,500 new ratings, equal to 80.9% of the original book rating count.',CW-2*PAD,22,DARK,'bold')
bar(x,21.25,CW,'Book Case Study: n = 500',24)
para(x+PAD,22.05,'Training on all users: 53,424 real + 500 fake. Below, each row is a mean over observed rating pairs.',CW-2*PAD,21)
text(x+PAD,23.20,r'Common global mean: $\mu=2.918$',23,DARK,'bold')
# Table preserves every score component without mixing training and prediction.
colx=[x+.18,x+2.15,x+4.40,x+6.25,x+8.15,x+10.20]
for cx,label in zip(colx,['Model /\nbooks',r'$b_u$',r'$b_i$',r'$p_u^\top q_i$',r'$\hat r$',r'$r$']):text(cx,24.10,label,17,DARK,'bold')
table=[]
for j,(model,group,label) in enumerate([('SVD','target','SVD / T'),('NMF','target','NMF / T'),('SVD','non_target','SVD / other'),('NMF','non_target','NMF / other')]):
 row=comp(model,'all_training',group);table.append(row.to_dict());yy=24.92+j*.77
 if j%2==0:base.add_patch(Rectangle((x+.10,yy-.10),CW-.20,.70,color='#F3F3F3',zorder=0))
 for cx,labelval in zip(colx,[label,f'{row.mean_bu:.3f}',f'{row.mean_bi:.3f}',f'{row.mean_interaction:.3f}',f'{row.mean_prediction:.3f}',f'{row.mean_recorded_rating:.3f}']):text(cx,yy,labelval,18,COL[model] if cx==colx[0] else DARK,'bold' if cx==colx[4] else 'normal')
text(x+PAD,28.18,'T = Thriller; other = non-Thriller.',18,MUTED)
bar(x,29.02,CW,'Different Components, Similar Score Trends',21)
para(x+PAD,29.84,'SVD: book bias and dot products both favor Thriller over other books. NMF: book bias creates a larger separation, while dot products partially offset it.',CW-2*PAD,23)
para(x+PAD,32.18,'The mean user bias is positive in SVD and negative in NMF. Adding all four components gives the final predicted rating.',CW-2*PAD,22)
text(x+PAD,34.55,'Shared settings: 8 factors; 180 epochs; seed 42.',18,MUTED)

# COLUMN 4: the same trained model, first synthetic fit and then real-user output.
x=X[3];bar(x,5.62,CW,'4. Why Do the Outputs Differ? (Books, n=500)',21)
para(x+PAD,6.40,'First, inspect only the fake users\' ratings from the same models trained on all real and fake data. Their desired ratings are exactly 5 and 1.',CW-2*PAD,22)
for j,(group,title,goal) in enumerate([('target','Thriller: recorded 5',5),('non_target','Other: recorded 1',1)]):
 ax=axis(x+.70+j*5.55,8.35,4.80,3.45);vals=[comp(m,'synthetic_training',group).mean_prediction for m in ['SVD','NMF']]
 ax.bar([0,1],vals,color=[BLUE,ORANGE],width=.58);ax.axhline(goal,color='#555555',lw=1.5,ls='--');ax.set_ylim(0,5.7);ax.set_xticks([0,1],['SVD','NMF']);ax.set_yticks([0,1,3,5]);ax.set_title(title,fontsize=19,pad=12,fontweight='bold');tidy(ax,18)
 for z,v in enumerate(vals):ax.text(z,v+.13,f'{v:.3f}',ha='center',fontsize=17,color=[BLUE,ORANGE][z],fontweight='bold')
para(x+PAD,12.35,'Training MAE (Thriller / other):\nSVD: 0.012 / 0.006     NMF: 1.136 / 0.244',CW-2*PAD,22,DARK,'bold')
para(x+PAD,13.67,'SVD uses positive and negative dot products. NMF\'s dot products are almost zero for these fake profiles; biases supply most of their score separation.',CW-2*PAD,22)
bar(x,15.71,CW,'Then Predict for Real Users',23)
para(x+PAD,16.47,'No parameters are updated. Rank unrated books for each real user. The global mean and that user\'s bias are the same for every candidate, so ranking depends on:',CW-2*PAD,22)
text(x+CW/2,18.32,r'$b_i+p_u^\top q_i$',30,DARK,ha='center')
text(x+PAD,19.15,'Thriller minus non-Thriller mean score:',21,DARK,'bold')
text(x+PAD,19.83,r'NMF: $3.096-2.635=+0.461$',25,ORANGE,'bold')
text(x+PAD,20.55,r'SVD: $0.296-0.396=-0.100$',25,BLUE,'bold')
para(x+PAD,21.37,'The first term is the book-bias advantage; the second is the opposing dot-product difference. The NMF bias advantage remains after this compensation.',CW-2*PAD,22)
bar(x,23.45,CW,'Observed Top-15 Recommendations',23)
ax=axis(x+1.10,24.45,CW-1.65,3.55);v=[float(ex[(ex.dataset=='Books')&(ex.model==m)&(ex.n_fake==500)].iloc[0]['mean']) for m in ['SVD','NMF']]
ax.bar([0,1],v,color=[BLUE,ORANGE],width=.55);ax.set_ylim(0,15);ax.set_yticks([0,5,10,15]);ax.set_xticks([0,1],['SVD','NMF']);ax.set_ylabel('Mean Thriller books',fontsize=20);tidy(ax,19)
for j,n in enumerate(v):ax.text(j,n+.20,f'{n:.2f} / 15',ha='center',fontsize=23,color=[BLUE,ORANGE][j],fontweight='bold')
bar(x,28.95,CW,'Conclusion',24)
para(x+PAD,29.70,'SVD fits the training ratings more closely. NMF gives Thriller stronger real-user exposure at n=500, alongside a larger book-bias separation that vectors only partly offset.',CW-2*PAD,22,DARK,'bold')
para(x+PAD,31.68,'Exposure is not held-out accuracy. These results describe the tested models and injection policy.',CW-2*PAD,18,MUTED)
# Author block follows the portrait placement in the supplied template.
authors=[(x+.15,'Raj Bhatnagar','poster_raj_reference.png'),(x+3.95,'Sara Moshtaghi','poster_sara_reference.png'),(x+7.75,'Vikram Ravindra','poster_vikram_reference.png')]
for xx,name,img in authors:
 text(xx+1.67,33.00,name,17,DARK,'bold',ha='center');text(xx+1.67,33.30,{'Raj Bhatnagar':'bhatnark@ucmail.uc.edu','Sara Moshtaghi':'moshtasa@mail.uc.edu','Vikram Ravindra':'ravindvm@ucmail.uc.edu'}[name],12,BLUE,ha='center');photo(xx+.83,33.52,1.70,2.18,img)

# Rendering validation and provenance.
fig.canvas.draw();renderer=fig.canvas.get_renderer();outside=[]
for t in texts:
 b=t.get_window_extent(renderer)
 if b.x0<0 or b.y0<0 or b.x1>W*fig.dpi or b.y1>H*fig.dpi:outside.append(t.get_text())
assert not outside,outside
pdf=OUT/'poster_svd_nmf_stage1.pdf';png=OUT/'poster_svd_nmf_stage1.png'
fig.savefig(pdf,facecolor='white');fig.savefig(png,dpi=150,facecolor='white');fig.savefig(OUT/'poster_svd_nmf_stage1_preview.png',dpi=55,facecolor='white');plt.close(fig)
manifest={'layout':'four columns, red section bars, original reference header and portraits','department':'Department of Computer Science, University of Cincinnati','results_scope':'all saved Top-15 doses','dose_ranges':{g:sorted(ex[ex.dataset==g].n_fake.unique().tolist()) for g in ex.dataset.unique()},'results':records,'n500_training_components':table,'flagged_book_nmf_doses':sorted(ex[(ex.dataset=='Books')&(ex.model=='NMF')&~ex.sign_ok].n_fake.tolist()),'n500_only_used_for_explanation':True,'synthetic_only_is_subset_of_same_models':True,'no_retraining':True,'pdf':str(pdf),'png':str(png)}
(NB/'poster_svd_nmf_stage1_validation.json').write_text(json.dumps(manifest,indent=2))
print('SAVED',pdf,png,flush=True)
