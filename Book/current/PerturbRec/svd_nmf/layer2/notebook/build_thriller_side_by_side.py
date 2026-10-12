from pathlib import Path
import argparse, subprocess, sys
from PIL import Image, ImageOps
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);r=p.parse_args().root
# Rebuild the complete individual figures before composing, so a previous combined output is never reused as input.
subprocess.run([sys.executable,str(r/'notebook/build_final_layer2_figures.py'),'--root',str(r)],check=True)
paths=[r/f'result/final_results/{model}/Thriller/neighborhood_recommendations_up_to_500.png' for model in ['NMF','SVD']]
images=[Image.open(p).convert('RGB') for p in paths]
assert images[0].size==images[1].size
w,h=images[0].size;gap=80
canvas=Image.new('RGB',(2*w+gap,h),'white')
canvas.paste(images[0],(0,0));canvas.paste(images[1],(w+gap,0))
canvas.save(paths[1],dpi=(200,200))
print('Full figures side by side:',paths[1],canvas.size)
