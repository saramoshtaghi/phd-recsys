#!/usr/bin/env python3
"""Generate clean-book injections matching the controlled SVD/NMF protocol."""
import argparse, hashlib, json, re
from pathlib import Path
import numpy as np
import pandas as pd
INPUT_CSV=Path('/home/moshtasa/Research/phd-recsys/Book/Archive/data/genre_clean.csv')
OUT_DIR=Path('/home/moshtasa/Research/phd-recsys/Book/current/PerturbRec/Data')
RUNS=[2,4,6,25,50,100,200,500,1000,2000,2500,3000,3500]
GENRES=['Drama','Thriller','Adventure']
POS_RATING,NEG_RATING=5,1
COLS=['user_id','book_id','rating','genres']

def split_genres(value):
 return [g.strip() for g in str(value).split(',') if g.strip() and g.strip().lower()!='unknown']

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--counts',nargs='+',type=int,default=RUNS);ap.add_argument('--genres',nargs='+',choices=GENRES,default=GENRES);ap.add_argument('--output-dir',type=Path,default=OUT_DIR)
 args=ap.parse_args();assert all(n>0 for n in args.counts)
 out=args.output_dir;out.mkdir(parents=True,exist_ok=True);(out/'manifests').mkdir(exist_ok=True)
 df=pd.read_csv(INPUT_CSV,usecols=COLS)[COLS]
 assert not df[COLS].isna().any().any()
 df[['user_id','book_id','rating']]=df[['user_id','book_id','rating']].astype('int64')
 assert df.rating.between(1,5).all()
 books=df[['book_id','genres']].drop_duplicates('book_id');mapping=dict(books.itertuples(index=False,name=None))
 assert all(split_genres(g) for g in mapping.values())
 assert len(books)==9415 and df.user_id.nunique()==53424 and len(df)==5815569
 source_hash=hashlib.sha256(pd.util.hash_pandas_object(df,index=False).values.tobytes()).hexdigest()
 for genre in args.genres:
  pos=sorted(b for b,g in mapping.items() if genre in split_genres(g));neg=sorted(set(mapping)-set(pos))
  item_ids=np.array(pos+neg,dtype=np.int64);ratings=np.r_[np.full(len(pos),5,dtype=np.int64),np.ones(len(neg),dtype=np.int64)]
  item_genres=np.array([mapping[b] for b in item_ids],dtype=object)
  for n in args.counts:
   dest=out/f'f_{genre}_{n}u_pos5_neg1_books.csv';meta_path=out/'manifests'/f'{genre.lower()}_{n}.json'
   expected=len(df)+n*len(books)
   if dest.exists():
    if not meta_path.exists():raise RuntimeError(f'Refusing to replace unverified existing dataset: {dest}')
    meta=json.loads(meta_path.read_text());assert meta['source_hash']==source_hash and meta['rows']==expected
    print('SKIP verified',dest.name,flush=True);continue
   tmp=dest.with_suffix('.csv.partial')
   df.to_csv(tmp,index=False)
   # Exactly the controlled trainer's order: each user, target books, other books.
   start=int(df.user_id.max())+1
   users_per_chunk=max(1,250000//len(books))
   for a in range(0,n,users_per_chunk):
    ids=np.arange(start+a,start+min(a+users_per_chunk,n));cnt=len(ids)
    part=pd.DataFrame({'user_id':np.repeat(ids,len(books)),'book_id':np.tile(item_ids,cnt),'rating':np.tile(ratings,cnt),'genres':np.tile(item_genres,cnt)})
    assert len(part)==cnt*len(books)
    part.to_csv(tmp,index=False,header=False,mode='a')
   tmp.rename(dest)
   meta=dict(genre=genre,n_fake=n,source=str(INPUT_CSV),source_hash=source_hash,real_users=53424,books=len(books),real_ratings=len(df),target_books=len(pos),other_books=len(neg),synthetic_ratings=n*len(books),rows=expected,first_fake_id=start,last_fake_id=start+n-1,positive_rating=5,negative_rating=1,row_order='Original rows unchanged; each fake user: sorted target IDs then sorted other IDs',bytes=dest.stat().st_size)
   meta_path.write_text(json.dumps(meta,indent=2));print('GENERATED',dest.name,'rows',expected,flush=True)
if __name__=='__main__':main()
