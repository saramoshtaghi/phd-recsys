#!/usr/bin/env python3
"""Re-render the six displayed PNGs from independently audited numbers; do not edit outputs."""
from pathlib import Path
import importlib.util,json,pandas as pd,numpy as np,hashlib
from PIL import Image,ImageChops
R=Path('/home/moshtasa/Research/phd-recsys');NB=R/'First_paper_material/notebook'
if __name__=='__main__':
 sp=importlib.util.spec_from_file_location('plots',NB/'build_genre_reviewer_figures.py');p=importlib.util.module_from_spec(sp);sp.loader.exec_module(p);gold=pd.read_csv(NB/'three_genre_panel_independent_components.csv');checks=[]
 for genre,folder in [('Thriller','thriller'),('Adventure','advanture'),('Drama','drama')]:
  f=gold[gold.genre==genre];ns=[0,2,4,6,25,50,100,200,500]
  for fn in [p.eight,p.prediction]:
   tmp=fn(f,genre,Path('/tmp'),ns,[]);actual=R/'First_paper_material/figures/first_reviewer'/folder/tmp.name;before=hashlib.sha256(actual.read_bytes()).hexdigest();a=Image.open(actual).convert('RGB');b=Image.open(tmp).convert('RGB');assert a.size==b.size;diff=ImageChops.difference(a,b);bounds=diff.getbbox();changed=0 if bounds is None else int(np.count_nonzero(np.asarray(diff).max(2)));assert before==hashlib.sha256(actual.read_bytes()).hexdigest();checks.append(dict(genre=genre,file=str(actual),independently_rendered=str(tmp),same_dimensions=True,pixel_identical=bounds is None,changed_pixels=changed,total_pixels=a.size[0]*a.size[1]));print(genre,tmp.name,bounds is None,changed,flush=True)
 (NB/'three_genre_pixel_audit.json').write_text(json.dumps(dict(checks=checks,all_pixel_identical=all(v['pixel_identical'] for v in checks),outputs_modified=False),indent=2))
 assert all(v['pixel_identical'] for v in checks)
