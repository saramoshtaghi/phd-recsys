#!/usr/bin/env python3
"""Audit completion and persist an artifact index, without changing model results."""
from pathlib import Path
import json,hashlib,time
import numpy as np
from PIL import Image
R=Path('/home/moshtasa/Research/phd-recsys');NB=R/'First_paper_material/notebook';ROOT=R/'Book/current/PerturbRec/svd_nmf';L=ROOT/'layer2';COUNTS=[0,2,4,6,25,50,100,200,500,1000,2000,2500,3000,3500]
if __name__=='__main__':
 statuses={}
 for name in ['training','analysis','input_validation','numerical_validation','figure_manifest','book_figure_status','svd_umap_status','nmf_umap_status']:
  path=NB/f'adventure_{name}.json' if name not in ['training','analysis'] else NB/f'adventure_{name}_status.json';d=json.loads(path.read_text());assert d['status'] in ['trained','complete','passed'],(name,d);statuses[name]=d['status']
 t=json.loads((NB/'adventure_training_status.json').read_text());assert len(t['completed'])==26 and not t['failures'];assert t['counts']==COUNTS[1:]
 for m in ['svd','nmf']:
  hs=[v['initial_common_hashes'] for v in t['completed'] if v['model']==m];assert all(h==hs[0] for h in hs)
 images=[]
 for directory in [R/'First_paper_material/figures',R/'First_paper_material/figures/first_reviewer',ROOT/'layer1/result/final_results',L/'final_results']:
  for path in directory.glob('*.png'):
   if 'adventure' not in path.name.lower():continue
   with Image.open(path) as im:im.verify()
   with Image.open(path) as im:size=list(im.size)
   images.append(dict(path=str(path),pixels=size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
 assert [x['path'] for x in images]==json.loads((NB/'adventure_figure_manifest.json').read_text())['figures'] and len(images)==1
 models=[]
 for m in ['svd','nmf']:
  for n in COUNTS[1:]:
   p=L/'embeddings/controlled'/m/f'{m}_adventure_{n}.npz';assert p.exists();d=json.loads((NB/f'adventure_{m}_{n}_training.json').read_text());models.append(dict(model=m,n_fake=n,path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),nonnegative_factors=d['nonnegative_factors']))
 for m in ['svd','nmf']:
  with np.load(L/'embeddings/controlled'/m/f'{m}_adventure_3500_umap.npz') as z:assert z['before'].shape==z['after'].shape==(53424,2) and z['synthetic'].shape==(3500,2)
 out=dict(status='complete',genre='Adventure',models=['svd','nmf'],counts=COUNTS,trained_cases=26,analyzed_cases=28,original_users=53424,original_books=9415,target_books=1789,source_ratings=5815569,raw_scores_and_factors_preserved=True,clipped=False,new_directories_created=False,checks=statuses,nonnegative_constraint_failures=json.loads((NB/'adventure_analysis_status.json').read_text())['nonnegative_constraint_failures'],images=images,checkpoints=models,finished=time.time())
 out['image_policy']='Only one Adventure comparison retained per user request: Top-15, Top-25 and Top-35, all injection sizes through 3500.'
 (NB/'adventure_completion_manifest.json').write_text(json.dumps(out,indent=2));print('FINAL AUDIT PASSED:',len(models),'checkpoints;',len(images),'retained PNG; previous Adventure figures removed per user request.')
