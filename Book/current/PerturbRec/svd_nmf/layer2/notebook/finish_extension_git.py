#!/usr/bin/env python3
"""Commit only the extension artifacts when the running pipeline completes."""
import time,json,subprocess
from pathlib import Path
import numpy as np,pandas as pd
R=Path('/home/moshtasa/Research/phd-recsys');P=R/'Book/current/PerturbRec';L=P/'svd_nmf/layer2/log/controlled'
for _ in range(720):
 status=json.loads((L/'extension_2500_3000_pipeline.json').read_text())
 if status['status']=='complete':break
 if status['status']=='failed':raise RuntimeError(str(status))
 time.sleep(20)
else:raise RuntimeError('Pipeline did not complete within four hours')
paths=[]
for genre in ['Drama','Thriller']:
 for n in [2500,3000]:
  paths.append(P/f'Data/manifests/{genre.lower()}_{n}.json')
  for model in ['nmf','svd']:
   paths.extend([P/f'svd_nmf/layer2/embeddings/controlled/{model}/{genre.lower()}/{model}_{genre.lower()}_{n}.npz',L/f'{model}/{genre.lower()}/{model}_{genre.lower()}_{n}.json'])
   metadata=json.loads(paths[-1].read_text());old=json.loads((L/f'{model}/{genre.lower()}/{model}_{genre.lower()}_2000.json').read_text())
   assert metadata['initial_common_hashes']==old['initial_common_hashes']
paths.extend((P/'svd_nmf/layer1/result/extension_2500_3000').rglob('*.png'))
paths.extend([P/'svd_nmf/layer1/tables/extension_2500_3000_recommendation_counts.csv',L/'extension_2500_3000.json',L/'extension_2500_3000_pipeline.json',L/'extension_2500_3000_pipeline.log'])
assert all(p.exists() for p in paths)
assert len(pd.read_csv(P/'svd_nmf/layer1/tables/extension_2500_3000_recommendation_counts.csv'))==64
relative=[str(p.relative_to(R)) for p in paths]
for args in [['git','add','-f','--',*relative],['git','commit','--only','-m','Save 2500 and 3000 controlled models and recommendation results','--',*relative],['git','push','origin','HEAD']]:
 result=subprocess.run(args,cwd=R,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT);print(result.stdout,flush=True)
 if result.returncode:raise RuntimeError(f'Git command failed: {args[1]}')
sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip()
(L/'extension_2500_3000_git_status.json').write_text(json.dumps(dict(status='pushed',commit=sha,files=len(relative)),indent=2))
print('RESULTS PUSHED',sha,flush=True)
