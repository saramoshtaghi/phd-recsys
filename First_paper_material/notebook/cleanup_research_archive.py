from pathlib import Path
import shutil,json,subprocess,time
R=Path('/home/moshtasa/Research/phd-recsys'); NB=R/'First_paper_material/notebook'; archive=R/'Book/Archive'
remove=[]
for p in archive.rglob('*'):
 if p.is_file() and not p.is_relative_to(archive/'data'):remove.append(p)
keep_ext={'.csv','.txt','.png','.jpg','.jpeg','.svg','.pdf','.eps','.tif','.tiff','.webp','.gif','.bmp'}
protected={R/'Mobile/current/0519/0519_SVD/notebook/SVD_0519.py',R/'Mobile/current/0519/0519_NMF/notebook/NMF_0519.py'}
for root in [R/'Mobile/archive',R/'Mobile/current/0519']:
 for p in root.rglob('*'):
  if p.is_file() and p.suffix.lower() not in keep_ext and p not in protected:remove.append(p)
entries=[dict(path=str(p.relative_to(R)),bytes=p.stat().st_size) for p in remove]
manifest=NB/'archive_cleanup_manifest.json';manifest.write_text(json.dumps(dict(time=time.strftime('%Y-%m-%d %H:%M:%S'),preserved_book_data='Book/Archive/data',mobile_keep_extensions=sorted(keep_ext),protected_sources=[str(p.relative_to(R)) for p in protected],deleted=entries,total_bytes=sum(x['bytes'] for x in entries)),indent=2))
tracked=set(subprocess.check_output(['git','ls-files','-z'],cwd=R).decode().split('\0'))
for p in remove:p.unlink()
for root in [archive,R/'Mobile/archive',R/'Mobile/current/0519']:
 for p in sorted(root.rglob('*'),key=lambda x:len(x.parts),reverse=True):
  if p.is_dir():
   try:p.rmdir()
   except OSError:pass
paths=[x['path'] for x in entries if x['path'] in tracked]+[str(manifest.relative_to(R)),'First_paper_material/notebook/cleanup_research_archive.py']
for start in range(0,len(paths),100):subprocess.run(['git','add','--',*paths[start:start+100]],cwd=R,check=True)
subprocess.run(['git','commit','-m','Remove obsolete book archives and retain mobile CSV text and figures','--',*paths],cwd=R,check=True)
subprocess.run(['git','push'],cwd=R,check=True)
print(json.dumps(dict(deleted_files=len(entries),deleted_GiB=sum(x['bytes'] for x in entries)/2**30,manifest=str(manifest))))
