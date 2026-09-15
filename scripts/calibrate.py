"""Select acceptance thresholds on validation only; never tune on test."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.hybrid import HybridAddressMatcher
from src.io import ROOT,read_reference,read_aliases
from src.evaluation import metrics


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--reference',required=True)
    p.add_argument('--queries',required=True)
    p.add_argument('--min-precision',type=float,default=.98)
    p.add_argument('--output',default=str(ROOT/'reports/calibration'))
    args=p.parse_args()
    if not 0<args.min_precision<=1: raise ValueError('min-precision must be in (0,1]')
    frame=read_reference(args.queries)
    if not {'split','source_id','query','true_id','category'}.issubset(frame):
        raise ValueError('Expected split,source_id,query,true_id,category')
    if frame.groupby('source_id').split.nunique().max()>1:
        raise ValueError('Source leakage across splits')
    validation=frame[frame.split=='validation']
    if validation.empty: raise ValueError('No validation rows')
    matcher=HybridAddressMatcher().fit(read_reference(args.reference),read_aliases())
    results=[]
    for threshold in (.70,.75,.80,.84,.88,.92,.96):
        for margin in (.02,.04,.06,.10):
            matcher.threshold,matcher.min_margin=threshold,margin
            records=[]
            for row in validation.to_dict('records'):
                d=matcher.match_one(row['query'])
                records.append(dict(row,candidate_id=d.candidate_id,accepted=d.status=='accepted'))
            results.append(dict(threshold=threshold,margin=margin,**metrics(records)))
    table=pd.DataFrame(results)
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    table.to_csv(out/'validation_grid.csv',index=False)
    eligible=table[(table.pair_precision>=args.min_precision)&(table.pair_tp>0)]
    if eligible.empty:
        choice={'status':'no_feasible_configuration','min_precision':args.min_precision}
    else:
        best=eligible.sort_values(['resolution_recall','pair_precision','threshold'],ascending=[False,False,False]).iloc[0]
        choice={'status':'selected_on_validation','threshold':float(best.threshold),'margin':float(best.margin),
                'validation_n':len(validation),'min_precision':args.min_precision,
                'note':'Point estimate only; confirm on untouched, sufficiently large real test data.'}
    (out/'selected.json').write_text(json.dumps(choice,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(choice,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
