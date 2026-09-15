"""Run regression cases or an externally labelled CSV against both pipelines."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.io import ROOT,read_reference,read_aliases
from src.hybrid import HybridAddressMatcher
from src.baseline import AddressMatcher
from src.evaluation import metrics


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--reference',default=str(ROOT/'data/reference_demo.csv'))
    p.add_argument('--queries',default=str(ROOT/'data/cases.csv'))
    p.add_argument('--output',default=str(ROOT/'reports'))
    p.add_argument('--threshold',type=float,default=.84)
    p.add_argument('--margin',type=float,default=.06)
    p.add_argument('--split',choices=['train','validation','test'])
    args=p.parse_args()
    ref=read_reference(args.reference)
    hybrid=HybridAddressMatcher(threshold=args.threshold,min_margin=args.margin).fit(ref,read_aliases())
    baseline=AddressMatcher().fit(hybrid.ref.address)
    queries=read_reference(args.queries)
    if not {'query','true_id','category'}.issubset(queries):
        raise ValueError('Queries need query,true_id,category; optional split,source_id')
    if args.split:
        if 'split' not in queries: raise ValueError('--split requires split column')
        queries=queries[queries.split==args.split]
    if queries.empty: raise ValueError('No queries to evaluate')
    if not set(queries.true_id)-{''} <= set(hybrid.ref.id):
        raise ValueError('Unknown true_id in queries')
    records=[]
    for row in queries.to_dict('records'):
        h=hybrid.match_one(row['query'])
        b=baseline.match_one(row['query'])
        for name,cid,accepted,score,status in [
            ('baseline',str(hybrid.ref.iloc[b.best_index].id),b.final_score>=.85,b.final_score,'threshold'),
            ('hybrid',h.candidate_id,h.status=='accepted',h.score,h.status)]:
            records.append(dict(row,model=name,candidate_id=cid,accepted=accepted,score=score,status=status))
    frame=pd.DataFrame(records)
    summaries=[]
    for name,df in frame.groupby('model'):
        summaries.append(dict(model=name,category='ALL',**metrics(df.to_dict('records'))))
        for category,group in df.groupby('category'):
            summaries.append(dict(model=name,category=category,**metrics(group.to_dict('records'))))
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    frame.to_csv(out/'predictions.csv',index=False)
    pd.DataFrame(summaries).to_csv(out/'metrics.csv',index=False)
    (out/'config.json').write_text(json.dumps(vars(args),ensure_ascii=False,indent=2),encoding='utf-8')
    print(pd.DataFrame(summaries).query("category=='ALL'").to_string(index=False))

if __name__=='__main__': main()
