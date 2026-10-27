from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT=Path('/Users/saramoshtaghi/Documents/Codex/2026-09-16/can')
REG='/System/Library/Fonts/Supplemental/Arial.ttf'
BOLD='/System/Library/Fonts/Supplemental/Arial Bold.ttf'
NAVY='#173243'; GREY='#536572'; RED='#CF102D'
def font(n,bold=False):return ImageFont.truetype(BOLD if bold else REG,n)
def build(model,color,vector,title):
 im=Image.new('RGB',(2200,740),'white');d=ImageDraw.Draw(im)
 def txt(x,y,s,n=36,fill=NAVY,bold=False,anchor=None):d.multiline_text((x,y),s,font=font(n,bold),fill=fill,spacing=12,anchor=anchor)
 def box(x,y,w,h,fill,outline=None):d.rounded_rectangle((x,y,x+w,y+h),radius=18,fill=fill,outline=outline,width=2)
 def arrow(a,b,y):
  d.line((a,y,b,y),fill=GREY,width=5);d.polygon([(b,y),(b-19,y-12),(b-19,y+12)],fill=GREY)
 d.rectangle((0,0,2200,12),fill=RED)
 txt(55,42,model,58,color,True);txt(270,51,title,43,NAVY,True)
 txt(55,134,'1  OBSERVED RATINGS',30,GREY,True)
 txt(650,134,'2  LEARN USER AND BOOK VECTORS',30,GREY,True)
 txt(1650,134,'3  PREDICT AND RANK',30,GREY,True)
 # An illustrative, sparse user-item matrix.
 box(55,205,440,320,'#F4F6F8')
 cw=102;xx=[70,183,285,387]
 for j,s in enumerate(['User','A','B','C']):txt(xx[j],227,s,31,NAVY,True)
 for row,y in [(['Sara','5','1','?'],310),(['Ali','4','?','5'],403)]:
  for j,s in enumerate(row):
   if s=='?':box(xx[j]-13,y-12,72,65,'#DDEFE8')
   txt(xx[j],y,s,34,'#007B5D' if s=='?' else NAVY,j==0)
 txt(73,477,'A, B, C = books',28,GREY)
 arrow(530,605,355)
 # Vectors are symbolic examples, not scores or experimental results.
 box(650,205,850,320,'#F6F8FA')
 txt(680,236,'User vector',32,NAVY,True);txt(680,352,'Book vector',32,NAVY,True)
 def cells(y,signs):
  for k,s in enumerate(signs):
   x=960+k*160;box(x,y,136,77,'#FCE3DC' if s=='−' else '#DDEBF3')
   txt(x+68,y+35,s,50,color,True,anchor='mm')
 cells(220,vector[0]);cells(336,vector[1])
 txt(680,463,'Compare the vectors using a dot product.',30,GREY)
 arrow(1535,1610,355)
 box(1650,205,495,320,'#F1F7F4')
 txt(1685,248,'Estimate the missing\nuser–book ratings.',36,NAVY,True)
 txt(1685,390,'Recommend the highest-\nscoring unrated books.',32,GREY)
 txt(55,586,('Vectors can contain negative, zero, or positive values.' if model=='SVD' else 'Vector values are constrained to be zero or positive.'),38,color,True)
 txt(55,657,'Both models also use the global rating mean and learned user and book biases. Teaching example; signs are illustrative.',27,GREY)
 im.save(ROOT/f'{model.lower()}_simple_explanation.png',dpi=(200,200))
build('SVD','#0072B2',(['+','−','+'],['−','+','+']),'Singular Value Decomposition')
build('NMF','#D55E00',(['+','0','+'],['0','+','+']),'Non-negative Matrix Factorization')
