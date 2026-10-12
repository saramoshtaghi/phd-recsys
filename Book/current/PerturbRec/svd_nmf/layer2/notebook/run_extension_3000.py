#!/usr/bin/env python3
"""Extend controlled training to 2500/3000, without replacing existing runs."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import json,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import numpy as np
from run_controlled import train_job,PARAMS,SRC,ROOT
from layer2_layout import pilot_file
P=ROOT.parent.parent
COUNTS=[2500,3000];MODELS=['nmf','svd'];GENRES=['Drama','Thriller']
MANIFEST=ROOT/'log/controlled/extension_2500_3000.json'

def main():
 MANIFEST.parent.mkdir(parents=True,exist_ok=True)
 for model in MODELS:
  with np.load(pilot_file(f'{model}_baseline_a.npz')) as a,np.load(pilot_file(f'{model}_baseline_b.npz')) as b:
   assert all(np.array_equal(a[k],b[k]) for k in a.files)
 for genre in GENRES:
  for n in COUNTS:
   meta=json.loads((P/'Data/manifests'/f'{genre.lower()}_{n}.json').read_text())
   assert meta['source']==str(SRC) and meta['books']==9415 and meta['real_ratings']==5815569 and meta['n_fake']==n
   assert (P/'Data'/f'f_{genre}_{n}u_pos5_neg1_books.csv').exists()
 manifest=dict(status='training',counts=COUNTS,models=MODELS,genres=GENRES,source=str(SRC),parameters=PARAMS,baseline_reused=True,paired_baselines_equal=True,started=time.time(),jobs=[])
 MANIFEST.write_text(json.dumps(manifest,indent=2));print('BASELINES VERIFIED; TRAINING 8 NEW CASES',flush=True)
 jobs=[(m,g,n) for n in COUNTS for m in MODELS for g in GENRES]
 with ProcessPoolExecutor(max_workers=8,mp_context=multiprocessing.get_context('spawn')) as pool:
  futures=[pool.submit(train_job,*job) for job in jobs]
  for f in futures:
   result=f.result();assert result['shared_initialization_verified'];manifest['jobs'].append(result);MANIFEST.write_text(json.dumps(manifest,indent=2))
 for m,g,n in jobs:
  p=ROOT/f'embeddings/controlled/{m}/{g.lower()}/{m}_{g.lower()}_{n}.npz'
  with np.load(p) as z:
   assert z['pu'].shape==(53424+n,8) and z['qi'].shape==(9415,8)
   assert all(np.isfinite(z[k]).all() for k in ['pu','qi','bu','bi','mu'])
 manifest.update(status='trained',finished=time.time());MANIFEST.write_text(json.dumps(manifest,indent=2));print('ALL 8 NEW MODELS COMPLETE',flush=True)
if __name__=='__main__':main()
