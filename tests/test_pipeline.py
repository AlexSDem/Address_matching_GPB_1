import pandas as pd
import pytest
from src.hybrid import HybridAddressMatcher
from src.io import ROOT,read_reference,read_aliases
from src.parser import parse_basic
from src.augmentation import augment,swap_house_building
from src.evaluation import metrics,group_split

@pytest.fixture(scope='module')
def matcher():
    return HybridAddressMatcher().fit(read_reference(ROOT/'data/reference_demo.csv'),read_aliases())

CASES=read_reference(ROOT/'data/cases.csv').to_dict('records')
@pytest.mark.parametrize('case',CASES,ids=[c['category']+str(i) for i,c in enumerate(CASES)])
def test_supervisor_cases(matcher,case):
    result=matcher.match_one(case['query'])
    assert result.best_id == (case['true_id'] or None), result
    if case['true_id']:
        assert result.status=='accepted'
    else:
        assert result.status!='accepted'


def test_augmentation_keeps_numeric_bindings(matcher):
    a=matcher.parts[8]
    q=dict(augment(a))['component_permutation']
    parsed=matcher.parser.parse(q)
    assert (parsed.house,parsed.building)==('10','1')
    bad=swap_house_building(a)
    assert (matcher.parser.parse(bad).house,matcher.parser.parse(bad).building)==('1','10')
    assert matcher.match_one(bad).best_id is None


def test_metrics_distinguish_pair_recall_and_resolution_recall():
    rows=[dict(true_id='a',candidate_id='a',accepted=True),
          dict(true_id='b',candidate_id='c',accepted=True),
          dict(true_id='d',candidate_id='d',accepted=False),
          dict(true_id='',candidate_id='a',accepted=False)]
    m=metrics(rows)
    assert (m['pair_tp'],m['pair_fp'],m['pair_fn'],m['pair_tn'])==(1,1,1,1)
    assert m['pair_precision']==.5 and m['pair_recall']==.5
    assert m['resolution_recall']==pytest.approx(1/3)


def test_split_no_source_leakage():
    f=pd.DataFrame({'source_id':[str(i) for i in range(20) for j in range(3)]})
    result=group_split(f)
    assert result.groupby('source_id').split.nunique().max()==1
    assert set(result.split)=={'train','validation','test'}


def test_topk_does_not_hide_ambiguity():
    ref=pd.DataFrame({'id':[str(i) for i in range(40)],
                      'address':[f'Регион{i} область, г. Тестоград, ул. Ленина, д. 1' for i in range(40)],
                      'region':[f'Регион{i} область' for i in range(40)],'city':['Тестоград']*40,
                      'street':['Ленина']*40,'house':['1']*40})
    m=HybridAddressMatcher(top_k=2).fit(ref)
    assert m.match_one('Тестоград, ул. Ленина 1').status=='review'
    assert len(m.rank('Тестоград, ул. Ленина 1')[1])==40


def test_validation_and_fitted():
    with pytest.raises(ValueError): HybridAddressMatcher().fit([])
    with pytest.raises(RuntimeError): HybridAddressMatcher().match_one('Москва')
    with pytest.raises(ValueError): HybridAddressMatcher(rerank_enabled=True)


def test_reranker_cannot_override_conflict(matcher):
    calls=[]
    def rr(q,c): calls.append(q); return [1.]*len(c)
    m=HybridAddressMatcher(reranker=rr,rerank_enabled=True).fit(matcher.ref,read_aliases())
    assert m.match_one('Курская область, г. Красноярск, ул. Ленина 25').best_id is None
    assert not calls


def test_legacy_osm_input():
    ref=pd.DataFrame({'united_addr':['Невский проспект, 5, Санкт-Петербург'],
                      'addr:street':['Невский проспект'],'addr:housenumber':['5']})
    m=HybridAddressMatcher().fit(ref)
    assert m.match_one('Санкт-Петербург, Невский пр-т 5').status=='accepted'


def test_all_positive_space_and_permutation_variants(matcher):
    for i,a in enumerate(matcher.parts):
        variants=dict(augment(a))
        for category in ('clean','component_permutation','missing_spaces','abbreviations'):
            d=matcher.match_one(variants[category])
            assert d.best_id==str(matcher.ref.iloc[i].id), (category,d)


def test_contextual_abbreviation(matcher):
    assert matcher.match_one('Красноярск, ул. Лени. 25').best_id=='kr25'
    assert matcher.match_one('Москва, пр-т Лени. 7').best_id is None


def test_city_alias_is_not_street_replacement(matcher):
    parsed=matcher.parser.parse('г. Москва, ул. Горький, д. 1')
    assert parsed.city=='москва' and parsed.street=='горький'


def test_fully_glued_known_addresses(matcher):
    from src.augmentation import blocks
    for i,a in enumerate(matcher.parts):
        query=''.join(blocks(a)).replace(' ','')
        assert matcher.match_one(query).best_id==str(matcher.ref.iloc[i].id)
