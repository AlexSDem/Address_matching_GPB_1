"""Component-aware hybrid retrieval and selective pair acceptance."""
from dataclasses import dataclass, asdict
import re
import numpy as np
import pandas as pd
from rapidfuzz.fuzz import ratio
from sklearn.feature_extraction.text import TfidfVectorizer
from .parser import AddressParser, parse_basic, FIELDS
from .address_normalize import normalize_ru_address, compact

WEIGHTS = {'region':2., 'city':3., 'district':.3, 'street_type':.5,
           'street':3., 'house':3., 'building':2., 'structure':2., 'apartment':1.}
IDENTITY = ('region','city','street_type','house','building','structure','apartment')

@dataclass
class Decision:
    query: str
    best: str | None
    best_id: str | None
    candidate: str | None
    candidate_id: str | None
    score: float
    margin: float
    status: str
    reasons: list
    parsed: dict
    reranked: bool = False

class HybridAddressMatcher:
    def __init__(self, top_k=30, threshold=.84, min_margin=.06,
                 component_weight=.85, reranker=None, rerank_enabled=False):
        if top_k < 2 or not 0 <= threshold <= 1 or not 0 <= min_margin <= 1 or not 0 <= component_weight <= 1:
            raise ValueError('Invalid top_k, threshold, min_margin or component_weight')
        if rerank_enabled and reranker is None:
            raise ValueError('Provide a validated reranker before enabling routing')
        self.top_k, self.threshold, self.min_margin = top_k, threshold, min_margin
        self.component_weight = component_weight
        self.reranker, self.rerank_enabled = reranker, rerank_enabled

    def fit(self, reference, aliases=()):
        if not isinstance(reference,pd.DataFrame):
            reference = pd.DataFrame({'address':list(reference)})
        ref = reference.copy().fillna('')
        if 'address' not in ref and 'united_addr' in ref:
            ref['address'] = ref['united_addr']
        if 'address' not in ref or ref.empty or ref['address'].astype(str).str.strip().eq('').any():
            raise ValueError('Reference needs nonempty address/united_addr values')
        if 'id' not in ref:
            # Stable within this file; production data should provide official IDs.
            ref['id'] = [str(i) for i in range(len(ref))]
        ref['id'] = ref['id'].astype(str)
        if ref['id'].duplicated().any() or ref['id'].str.strip().eq('').any():
            raise ValueError('Reference IDs must be unique and nonempty')
        self.ref = ref.reset_index(drop=True)
        self.parts = []
        osm = {'city':'addr:city','street':'addr:street','house':'addr:housenumber'}
        for row in self.ref.to_dict('records'):
            a = parse_basic(row['address'])
            for key in FIELDS:
                value = row.get(key) or row.get(osm.get(key,''))
                if value:
                    setattr(a,key,normalize_ru_address(value))
            # OSM addr:street often contains the type, unlike structured street.
            street_parts = parse_basic(a.street)
            if street_parts.street:
                a.street = street_parts.street
                a.street_type = a.street_type or street_parts.street_type
            number_parts = parse_basic('дом '+a.house)
            if number_parts.house:
                a.house = number_parts.house
                a.building = a.building or number_parts.building
                a.structure = a.structure or number_parts.structure
            self.parts.append(a)
        self.parser = AddressParser(self.parts, aliases)
        self.norm = [normalize_ru_address(x) for x in ref.address]
        self.vectorizer = TfidfVectorizer(analyzer='char',ngram_range=(2,4))
        self.matrix = self.vectorizer.fit_transform(self.norm)
        self.index = {key:{} for key in ('city','region','house','street')}
        for i,a in enumerate(self.parts):
            for key in self.index:
                self.index[key].setdefault(getattr(a,key),set()).add(i)
        return self

    def _pair(self,q,r):
        scores, conflicts = {}, []
        for key,weight in WEIGHTS.items():
            left,right = getattr(q,key),getattr(r,key)
            if not left:
                continue  # Missing query region is not a mismatch.
            if not right:
                scores[key]=0.
                conflicts.append('unverified_'+key)
                continue
            similarity = ratio(compact(left),compact(right))/100
            scores[key] = similarity
            if left != right and key in IDENTITY:
                # Only spelling variation of city names; digit suffixes are identity.
                typo = (key=='city' and similarity>=.88 and
                        re.findall(r'\d+',left)==re.findall(r'\d+',right))
                if not typo:
                    conflicts.append('conflict_'+key)
            if key=='street' and similarity < .68:
                conflicts.append('conflict_street')
        denom = sum(WEIGHTS[k] for k in scores)
        component = sum(WEIGHTS[k]*v for k,v in scores.items())/denom if denom else 0.
        return component, scores, conflicts

    def rank(self,query):
        if not hasattr(self,'matrix'):
            raise RuntimeError('Call fit first')
        q = self.parser.parse(query)
        sims = (self.matrix @ self.vectorizer.transform([q.normalized]).T).toarray().ravel()
        pool = set(np.argsort(-sims,kind='stable')[:self.top_k].tolist())
        # Union with component block prevents top-k from hiding a same-name city
        # or another корпус. No hard geography filter before typo resolution.
        blocks = [self.index[k].get(getattr(q,k),set()) for k in self.index if getattr(q,k)]
        blocks = [b for b in blocks if b]
        if blocks:
            pool.update(set.intersection(*blocks))
        rows = []
        for i in pool:
            component,details,conflicts = self._pair(q,self.parts[i])
            score = self.component_weight*component + (1-self.component_weight)*max(0.,float(sims[i]))
            rows.append({'candidate_id':self.ref.iloc[i]['id'],'candidate':self.ref.iloc[i]['address'],
                         'ref_index':i,'score':score,'retrieval_score':float(sims[i]),
                         'component_score':component,'components':details,'conflicts':conflicts})
        rows.sort(key=lambda r:(bool(r['conflicts']),-r['score'],r['candidate_id']))
        return q,rows

    def match_one(self,query):
        q,rows = self.rank(query)
        best = rows[0]
        viable = [r for r in rows if not r['conflicts']]
        margin = best['score']-viable[1]['score'] if len(viable)>1 else best['score']
        reasons = list(q.warnings) + best['conflicts']
        if not q.city or not q.street or not q.house:
            reasons.append('incomplete_identity')
        # Require clarification if query fits more than one canonical entity.
        # Text score must not resolve missing region/building/structure.
        if viable:
            identity_matches = [r for r in viable if r['component_score']>=.999]
            if len(identity_matches)>1:
                reasons.append('ambiguous_reference')
        reranked = False
        # Explicit extension point. Never override identity conflicts/incompleteness.
        if self.rerank_enabled and not reasons and len(viable)>1 and margin<self.min_margin:
            candidates = viable[:self.top_k]
            values = np.asarray(self.reranker(query,[r['candidate'] for r in candidates]),dtype=float)
            if values.shape != (len(candidates),) or not np.isfinite(values).all() or ((values<0)|(values>1)).any():
                raise ValueError('Reranker must return one finite [0,1] score per candidate')
            for row,value in zip(candidates,values):
                row['score'] = .8*row['score']+.2*float(value)
            candidates.sort(key=lambda r:-r['score'])
            best=candidates[0]
            margin=best['score']-candidates[1]['score']
            reranked=True
        if best['score']<self.threshold:
            reasons.append('low_score')
        if margin<self.min_margin:
            reasons.append('small_margin')
        if reasons:
            status = 'review' if not best['conflicts'] and best['score']>=self.threshold else 'no_match'
        else:
            status = 'accepted'
        accepted = status=='accepted'
        return Decision(str(query),best['candidate'] if accepted else None,
                        best['candidate_id'] if accepted else None,best['candidate'],best['candidate_id'],
                        float(best['score']),float(margin),status,list(dict.fromkeys(reasons)),q.to_dict(),reranked)

    def match_batch(self,queries):
        return pd.DataFrame([asdict(self.match_one(q)) for q in queries])
