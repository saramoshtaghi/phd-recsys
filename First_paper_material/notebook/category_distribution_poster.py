from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
P=Path(__file__).resolve().parent
mobile=json.loads((P/'mobile_original_category_counts.json').read_text())
books={'Drama':3006,'Mystery':2563,'Romance':2131,'Fantasy':2088,'Adventure':1789,'Thriller':1606,'Nonfiction':1071,'Classics':901,"Children's":863,'Historical':857,'Science Fiction':855,'Horror':769,'Adult':331}
movies={'1990s':1348,'1980s':107,'1970s':53,'1960s':43,'1950s':54,'1940s':45,'1930s':29,'1920s':2}
assert len(mobile)==48 and sum(mobile.values())==10173
BLUE='#0072B2';ORANGE='#D55E00';RED='#CF102D';DARK='#111111'
plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42})
W,H=11.4,9.1
fig=plt.figure(figsize=(W,H),facecolor='white')
def heading(y,title,size=18):
 ax=fig.add_axes([0,(H-y-.52)/H,1,.52/H]);ax.set_facecolor(RED);ax.set_xticks([]);ax.set_yticks([])
 for sp in ax.spines.values():sp.set_visible(False)
 ax.text(.5,.5,title,ha='center',va='center',fontsize=size,color='white',weight='bold')
def panel(x,y,w,h,data,target,title,fs=13,maxv=None):
 names=list(data);values=list(data.values());n=len(names)
 ax=fig.add_axes([x/W,(H-y-h)/H,w/W,h/H]);ax.set_xlim(0,1);ax.set_ylim(n-.5,-.9);ax.axis('off')
 ax.text(0,-1.15,title,fontsize=15,weight='bold',color=DARK)
 mx=maxv or max(values)
 label=.64 if title=='' else (.53 if len(title)>20 else .50)
 for k,(name,v) in enumerate(data.items()):
  c=ORANGE if name==target else BLUE
  ax.text(0,k,name,ha='left',va='center',fontsize=fs,color=c if name==target else DARK,weight='bold' if name==target else 'normal')
  ax.barh(k,(v/mx)*(.83-label),left=label,height=.55,color=c)
  ax.text(label+(v/mx)*(.83-label)+.015,k,f'{v:,}',ha='left',va='center',fontsize=fs-1,color=DARK)
heading(0,'Item Categories Across Three Datasets',18)
panel(.16,1.05,6.25,3.1,books,'Thriller','Books: 13 genres',13.8)
panel(6.75,1.05,4.45,3.1,movies,'1990s','Movies: 8 release decades',13.8)
fig.text(.02,1-4.48/H,'Mobile apps: 48 original categories',fontsize=15,weight='bold',va='top')
items=list(mobile.items())
for j in range(3):
 part=dict(items[j*16:(j+1)*16]);panel(.16+j*3.78,4.93,3.58,3.67,part,'Shopping','',10.9,maxv=max(mobile.values()))
fig.text(.015,.023,'Orange: target category     Blue: other categories',fontsize=11.8,color=DARK)
fig.text(.015,.005,'Counts are unique items. Book labels overlap; one movie has no decade label.',fontsize=9.2,color='#444444')
fig.savefig(P/'dataset_categories_poster.png',dpi=400,facecolor='white')
print(P/'dataset_categories_poster.png')
