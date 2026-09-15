"""Separate top-1 pair classification from end-to-end address resolution."""
import numpy as np


def metrics(rows):
    n=len(rows)
    if not n: return {'n':0}
    positive=[r for r in rows if r['true_id']]
    tp=sum(r['accepted'] and r['candidate_id']==r['true_id'] and bool(r['true_id']) for r in rows)
    fp=sum(r['accepted'] and (not r['true_id'] or r['candidate_id']!=r['true_id']) for r in rows)
    # Pair label = is the top-1 candidate a true match, prediction = accepted.
    pair_fn=sum(not r['accepted'] and bool(r['true_id']) and r['candidate_id']==r['true_id'] for r in rows)
    pair_tn=n-tp-fp-pair_fn
    precision=tp/(tp+fp) if tp+fp else 0.
    pair_recall=tp/(tp+pair_fn) if tp+pair_fn else 0.
    resolution_recall=tp/len(positive) if positive else 0.
    hits=sum(r['candidate_id']==r['true_id'] for r in positive)
    # A wrong accepted entity is both a false link and a missed correct link.
    return {'n':n,'matchable_n':len(positive),'hitrate_at_1':hits/len(positive) if positive else None,
            'pair_tp':tp,'pair_fp':fp,'pair_fn':pair_fn,'pair_tn':pair_tn,
            'pair_accuracy':(tp+pair_tn)/n,'pair_precision':precision,'pair_recall':pair_recall,
            'pair_f1':2*precision*pair_recall/(precision+pair_recall) if precision+pair_recall else 0.,
            'resolution_recall':resolution_recall,
            'resolution_f1':2*precision*resolution_recall/(precision+resolution_recall) if precision+resolution_recall else 0.,
            'coverage':(tp+fp)/n,
            'decision_accuracy':sum((r['accepted'] and r['candidate_id']==r['true_id']) if r['true_id'] else not r['accepted'] for r in rows)/n}


def group_split(frame,seed=42):
    """Split source IDs before evaluating variants; reference remains searchable."""
    groups=sorted(frame['source_id'].unique())
    rng=np.random.default_rng(seed)
    rng.shuffle(groups)
    n=len(groups)
    if n<3: raise ValueError('At least 3 independent source IDs are required')
    a=max(1,int(n*.6)); b=max(a+1,int(n*.8))
    mapping={g:('train' if i<a else 'validation' if i<b else 'test') for i,g in enumerate(groups)}
    return frame.assign(split=frame.source_id.map(mapping))
