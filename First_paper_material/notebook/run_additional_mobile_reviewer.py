"""Run the Shopping diagnostic pipeline for Casual or Beauty."""
import sys
from pathlib import Path
category=sys.argv[1]
if category not in ("Casual","Beauty"):
    raise SystemExit("Expected Casual or Beauty")
root=Path("/home/moshtasa/Research/phd-recsys")
nb=root/"First_paper_material/notebook"
source=(nb/"run_shopping_reviewer.py").read_text()
source=source.replace("Shopping",category).replace("shopping",category.lower())
exec(compile(source,str(nb/"run_shopping_reviewer.py"),"exec"),{"__name__":"__main__"})
render=(nb/"render_shopping_reviewer.py").read_text()
render=render.replace("Shopping",category).replace("shopping",category.lower())
exec(compile(render,str(nb/"render_shopping_reviewer.py"),"exec"),{"__name__":"__main__"})
