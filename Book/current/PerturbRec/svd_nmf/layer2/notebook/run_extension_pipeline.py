#!/usr/bin/env python3
"""Persistent generation -> controlled training -> recommendation evaluation."""
import subprocess,json,time,sys
from pathlib import Path
P=Path('/home/moshtasa/Research/phd-recsys/Book/current/PerturbRec')
LOG=P/'svd_nmf/layer2/log/controlled/extension_2500_3000_pipeline.json'
stages=[('generating',[sys.executable,str(P/'generate_injections.py'),'--counts','2500','3000','--genres','Drama','Thriller']),('training',[sys.executable,str(P/'svd_nmf/layer2/notebook/run_extension_3000.py')]),('recommendations',[sys.executable,str(P/'svd_nmf/layer1/notebook/analyze_extension_3000.py')])]
state=dict(status='starting',started=time.time(),completed_stages=[])
for stage,cmd in stages:
 state['status']=stage;LOG.write_text(json.dumps(state,indent=2));print('STAGE',stage,flush=True)
 result=subprocess.run(cmd)
 if result.returncode:
  state.update(status='failed',failed_stage=stage,exit_code=result.returncode);LOG.write_text(json.dumps(state,indent=2));raise SystemExit(result.returncode)
 state['completed_stages'].append(stage)
state.update(status='complete',finished=time.time());LOG.write_text(json.dumps(state,indent=2));print('PIPELINE COMPLETE',flush=True)
