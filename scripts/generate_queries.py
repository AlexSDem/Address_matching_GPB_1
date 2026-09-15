from dataclasses import replace
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.io import ROOT,read_reference,read_aliases
from src.hybrid import HybridAddressMatcher
from src.augmentation import augment,swap_house_building
from src.parser import FIELDS
from src.evaluation import group_split


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--reference',default=str(ROOT/'data/reference_demo.csv'))
    p.add_argument('--output',default=str(ROOT/'data/generated_queries.csv'))
    p.add_argument('--seed',type=int,default=42)
    args=p.parse_args()
    m=HybridAddressMatcher().fit(read_reference(args.reference),read_aliases())
    rows=[]
    def label_components(q, shortened=False):
        # Labels derive from source components, never from the parser under test.
        keys=[k for k in FIELDS if k!='district' and getattr(q,k)]
        def compatible(a):
            for k in keys:
                left,right=getattr(q,k),getattr(a,k)
                if k=='street' and shortened:
                    if len(left.split())!=len(right.split()) or not all(w.startswith(t[:4]) for t,w in zip(left.split(),right.split())):
                        return False
                elif left!=right:
                    return False
            return True
        ids=[str(m.ref.iloc[i].id) for i,a in enumerate(m.parts) if compatible(a)]
        return ids[0] if len(ids)==1 else ''
    for i,a in enumerate(m.parts):
        source=str(m.ref.iloc[i].id)
        for category,query in augment(a,args.seed+i):
            # All positive transformations retain the source identity except
            # omissions, which can remove information needed for unique matching.
            target=label_components(replace(a,region='',district='')) if category=='incomplete' else source
            if category=='street_abbreviation':
                target=label_components(a,shortened=True)
            rows.append({'source_id':source,'query':query,'true_id':target,'category':category})
        bad=swap_house_building(a)
        if bad:
            rows.append({'source_id':source,'query':bad,'true_id':label_components(replace(a,house=a.building,building=a.house)),'category':'invalid_permutation'})
    result=group_split(pd.DataFrame(rows),args.seed)
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    result.to_csv(args.output,index=False)
    print(result.groupby(['split','category']).size().to_string())

if __name__=='__main__': main()
