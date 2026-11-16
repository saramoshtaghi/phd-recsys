from pathlib import Path
import json,subprocess
R=Path('/home/moshtasa/Research/phd-recsys/First_paper_material')
template=(R/'notebook/render_shopping_reviewer.py').read_text()
configs=[('Thriller','thriller','thriller','Book'),('Drama','drama','drama','Book'),('Adventure','adventure','advanture','Book'),('1990s','movies_1990','1990','Movie'),('1950s','movies_1950','1950','Movie'),('1920s','movies_1920','1920','Movie'),('Shopping','shopping','shopping','App'),('Casual','casual','casual','App'),('Beauty','beauty','beauty','App')]
outputs=[]
for label,stem,folder,item in configs:
 s=template.replace('shopping',stem).replace('Shopping',label).replace('App bias',item+' bias')
 s=s.replace('first_reviewer/'+stem,'first_reviewer/'+folder)
 if item=='Book':
  old="f=pd.read_csv(R/'notebook/"+stem+"_reviewer_components.csv')"
  new="f=pd.read_csv(R/'notebook/three_genre_panel_independent_components.csv'); f=f[(f.genre=="+repr(label)+") & (f.n_fake<=500)].copy()"
  assert old in s
  s=s.replace(old,new)
 exec(compile(s,'render_'+stem,'exec'),{'__name__':'__main__'})
 root=R/'figures/first_reviewer'/folder
 files=[root/('training_'+stem+'_all_users_overview_4panel.png'),root/('training_'+stem+'_synthetic_only_overview_5panel.png'),root/('prediction_real_users_'+stem+'_non_'+stem+'_4panel.png'),root/('training_'+stem+'_all_and_synthetic_8panel_paper.png')]
 files += [root/'paper_figures'/(prefix+'_5panel_paper.png') for prefix in ['training_'+stem+'_all_users','training_'+stem+'_synthetic_only','prediction_real_users_'+stem]]
 assert all(p.exists() and p.stat().st_size>10000 for p in files)
 manifest={'category':label,'source':'three_genre_panel_independent_components.csv' if item=='Book' else stem+'_reviewer_components.csv','figures':[str(p.relative_to(root)) for p in files],'training_panels':5,'prediction_panels':4,'combined_training_panels':8,'note':'Legacy filenames retained; panel counts documented here.'}
 mp=root/'paper_figures/figure_manifest.json';mp.write_text(json.dumps(manifest,indent=2));files.append(mp)
 if item=='Movie':
  alias=root/('prediction_real_users_'+folder+'_non_'+folder+'_4panel.png')
  if alias.exists(): alias.write_bytes(files[2].read_bytes());files.append(alias)
 outputs.extend(files)
 print('COMPLETE',label,flush=True)
repo=R.parent
paths=[str(p.relative_to(repo)) for p in outputs]
paths += ['First_paper_material/notebook/synchronize_category_figures.py']
subprocess.run(['git','add','--',*paths],cwd=repo,check=True)
subprocess.run(['git','commit','-m','Unify complete reviewer figure sets across nine categories','--',*paths],cwd=repo,check=True)
subprocess.run(['git','push'],cwd=repo,check=True)
