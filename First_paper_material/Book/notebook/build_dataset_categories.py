"""Rebuild the paper's dataset figure from unique-item metadata, without changing its mapping."""
from pathlib import Path
import ast,json,hashlib,textwrap
import pandas as pd,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator,StrMethodFormatter
ROOT=Path('/home/moshtasa/Research/phd-recsys');OUT=ROOT/'First_paper_material/Book';DATA=OUT/'data';DATA.mkdir(parents=True,exist_ok=True)
NB=ROOT/'state of the art/paper_figures/figures_0323.ipynb'
old=json.loads(NB.read_text());code=''.join(old['cells'][1]['source']);mapping=next(ast.literal_eval(n.value) for n in ast.parse(code).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='category_mapping' for t in n.targets))
paths={'books':ROOT/'Book/Archive/data/genre_clean.csv','movies':ROOT/'Movie /archive/preprocessing/df_final.csv','mobile':ROOT/'Mobile/archive/0316_Single/data/data/mobilerec_final.csv'}
def unique_meta(path,cols):
 pieces=[d.drop_duplicates() for d in pd.read_csv(path,usecols=cols,chunksize=300000,low_memory=False)]
 return pd.concat(pieces,ignore_index=True).drop_duplicates()
b=unique_meta(paths['books'],['book_id','genres']);assert not b.book_id.duplicated().any();Nbook=b.book_id.nunique()
g=b.assign(genre=b.genres.astype(str).str.split(',')).explode('genre');g.genre=g.genre.str.strip();g=g[~g.genre.str.lower().isin(['unknown','','nan'])].drop_duplicates(['book_id','genre'])
bcounts=g.groupby('genre').book_id.nunique().sort_values(ascending=False)
ref=pd.read_csv(ROOT/'Book/current/Data/domain_analysis/figures/genre_summary_table.csv').set_index('genre').total
assert bcounts.to_dict()==ref.to_dict(), 'Original genre counts differ from distinct-book reconstruction'
m=unique_meta(paths['movies'],['item_id','decade']);m['decade']=pd.to_numeric(m.decade,errors='coerce');m=m[m.decade.notna()&(m.decade>=0)];assert not m.item_id.duplicated().any();m['decade']=m.decade.astype(int);mcounts=m.groupby('decade').item_id.nunique().sort_index(ascending=False);Nmovie=m.item_id.nunique()
a=unique_meta(paths['mobile'],['app_package','app_category']);assert not a.app_package.duplicated().any();Napp=a.app_package.nunique();a['domain']=a.app_category.map(mapping)
acounts=a.groupby('domain').app_package.nunique().sort_values(ascending=False);unmapped=a[a.domain.isna()].groupby('app_category').app_package.nunique().to_dict()
ref=pd.read_csv(ROOT/'Mobile/archive/0316_Single/domain_analysis/figures/mapped_category_counts.csv').set_index('category_mapped')['count'];assert acounts.to_dict()==ref.to_dict()
assert len(bcounts)==13 and len(mcounts)==8 and len(acounts)==9
assert Nbook==9415 and Nmovie==1681 and Napp==10173
mapped=int(acounts.sum());assert mapped+sum(unmapped.values())==Napp
meta={'sources':{k:str(v) for k,v in paths.items()},'mapping_source':str(NB),'original_image_sha256':hashlib.sha256((ROOT/'state of the art/paper_figures/combined_3plots_A4.png').read_bytes()).hexdigest(),'books':{'n_unique_items':int(Nbook),'counts':{str(k):int(v) for k,v in bcounts.items()},'multi_label':True},'movies':{'n_unique_items':int(Nmovie),'counts':{str(k):int(v) for k,v in mcounts.items()}},'mobile':{'n_unique_items':int(Napp),'mapped_items':mapped,'counts':{str(k):int(v) for k,v in acounts.items()},'unmapped_categories':{str(k):int(v) for k,v in unmapped.items()}},'original_saved_summaries_matched':True,'duplicate_item_category_counts_removed':True,'original_nine_domain_mapping_unchanged':True}
(DATA/'dataset_category_counts.json').write_text(json.dumps(meta,indent=2))
rows=[{'dataset':name,'category':str(k),'unique_items':int(v)} for name,s in [('Books',bcounts),('Movies',mcounts),('Mobile apps',acounts)] for k,v in s.items()]
pd.DataFrame(rows).to_csv(DATA/'dataset_category_counts.csv',index=False)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'axes.titlesize':10,'axes.titleweight':'bold','axes.labelsize':8.5,'axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False,'axes.linewidth':.5,'xtick.labelsize':7.5,'ytick.labelsize':8.2,'savefig.facecolor':'white','figure.facecolor':'white'})
fig,axs=plt.subplots(3,1,figsize=(5.4,7.1),gridspec_kw={'height_ratios':[13,8,9]})
series=[bcounts,mcounts,acounts];titles=[f'(a) Books: genre (n = {Nbook:,})',f'(b) Movies: decade (n = {Nmovie:,})',f'(c) Mobile apps: domain (n = {mapped:,})'];colors=['#0072B2','#009E73','#8064A2'];units=['Unique books','Unique movies','Unique applications']
for ax,s,title,color,unit in zip(axs,series,titles,colors,units):
 y=np.arange(len(s));labels=[str(x)+'s' if unit=='Unique movies' else str(x) for x in s.index]
 labels=[x.replace('Science Fiction','Science fiction').replace('Nonfiction','Non-fiction') for x in labels]
 ax.barh(y,s.to_numpy(),height=.66,color=color,zorder=3)
 ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_ylim(len(s)-.45,-.65);ax.set_xlim(0,float(s.max())*1.18)
 for yy,v in zip(y,s):ax.text(v+float(s.max())*.018,yy,f'{int(v):,}',va='center',ha='left',fontsize=7.7,color='#202B33')
 ax.set_title(title,loc='left',pad=8,color='#172B3A');ax.set_xlabel(unit,labelpad=3,color='#35434D')
 ax.grid(axis='x',color='#DFE5E9',linewidth=.45,zorder=0);ax.set_axisbelow(True);ax.tick_params(axis='y',length=0,pad=5);ax.tick_params(axis='x',length=2.5,color='#84929B');ax.spines['bottom'].set_color('#AEBAC2');ax.xaxis.set_major_locator(MaxNLocator(nbins=4,integer=True));ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
fig.subplots_adjust(left=.325,right=.965,top=.95,bottom=.135,hspace=.65)
fig.text(.04,.071,'Books may have two genre labels and can therefore appear in two bars.',fontsize=7.1,color='#4E5B65')
fig.text(.04,.043,f'Mobile mapping covers {mapped:,} of {Napp:,} apps; {sum(unmapped.values()):,} medical apps have no mapped domain.',fontsize=7.1,color='#4E5B65')
file=OUT/'dataset_categories.png';fig.savefig(file,dpi=600);plt.close(fig)
(OUT/'dataset_categories_caption.md').write_text('Figure. Item-category distributions in the three datasets. (a) Unique books per assigned genre; multi-label books can contribute to two genres. (b) Unique MovieLens movies per release decade. (c) Unique mobile applications in the original nine aggregated domains. The mobile mapping covers 9,751 of 10,173 applications; the 422 applications labelled Medical in the original metadata are not assigned a domain by that mapping. Each bar counts distinct items rather than ratings. All axes use linear scales.\n\nLaTeX path: figures/dataset_categories.png. Upload dataset_categories.png into the project’s figures folder to replace the matching figure without changing its LaTeX filename.\n')
print(json.dumps(meta,indent=2));print('OUTPUT',file)
