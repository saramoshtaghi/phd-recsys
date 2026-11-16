"""Rebuild movie decade diagnostics with the original movie settings."""
from pathlib import Path
import sys
R=Path("/home/moshtasa/Research/phd-recsys");NB=R/"First_paper_material/notebook"
for decade in ["1990","1950"]:
 source=(NB/"run_shopping_reviewer.py").read_text()
 source=source.replace("Mobile/archive/0316_Single/data/data/mobilerec_final.csv","Movie /archive/preprocessing/df_final.csv")
 source=source.replace("NS=[0,30,60,300,500,2000,7000]","NS=[0,2,6,10,50]")
 source=source.replace("app_package","item_id").replace("app_category","decade").replace("uid","user_id")
 source="\n".join(" df.decade=pd.to_numeric(df.decade,errors=\"coerce\").fillna(0).astype(int).astype(str)" if line.startswith(" df.decade=") else line for line in source.splitlines())
 source=source.replace("to_raw_user_id","to_raw_uid")
 source=source.replace("Shopping",decade).replace("shopping","movies_"+decade)
 source=source.replace("first_reviewer/movies_"+decade,"first_reviewer/"+decade)
 source=source.replace("realusers=df.user_id.unique()", "pos=np.array(sorted(pos,key=int));neg=np.array(sorted(neg,key=int)); realusers=df.user_id.unique()")
 exec(compile(source,str(NB/"run_movie_reviewer.py"),"exec"),{"__name__":"__main__"})
 render=(NB/"render_shopping_reviewer.py").read_text().replace("Shopping",decade+"s").replace("shopping","movies_"+decade)
 render=render.replace("first_reviewer/movies_"+decade,"first_reviewer/"+decade).replace("App bias","Movie bias")
 exec(compile(render,str(NB/"render_shopping_reviewer.py"),"exec"),{"__name__":"__main__"})
