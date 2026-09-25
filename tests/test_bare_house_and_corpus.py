import pandas as pd
import pytest
from src.address_normalize import normalize_ru_address
from src.hybrid import HybridAddressMatcher
from src.parser import parse_basic


@pytest.fixture
def matcher():
    # Reproduce BOTH formats of previously generated OSM caches:
    # address text and a house column still containing '25 к1'.
    ref = pd.DataFrame({
        'id': ['b1', 'b2', 's1', 'other', 'letter', 'fraction'],
        'address': ['г. Москва, Туристская улица, д. '+x
                    for x in ['25 к1', '25 к2', '25 стр. 1', '26 к1', '25к', '25/1']],
        'city': ['Москва'] * 6,
        'street': ['Туристская улица'] * 6,
        'house': ['25 к1', '25 к2', '25 стр. 1', '26 к1', '25к', '25/1'],
    })
    return HybridAddressMatcher().fit(ref)


@pytest.mark.parametrize('query', [
    'москва туристская 25 к1',
    'москва туристская 25 к 1',
    'москва туристская 25 к. 1',
    'москва туристская 25к1',
    'москва туристская 25 корпус 1',
    'Москва, Туристская улица, 25 к1',
    'г. Москва, ул. Туристская, д. 25, корп. 1',
])
def test_user_query(matcher, query):
    result = matcher.match_one(query)
    assert result.parsed['house'] == '25'
    assert result.parsed['building'] == '1'
    assert result.status == 'accepted', result
    assert result.best_id == 'b1'


def test_reference_reparsed_without_redownloading(matcher):
    assert matcher.parts[0].house == '25'
    assert matcher.parts[0].building == '1'


@pytest.mark.parametrize('query,expected', [
    ('москва туристская 25 к2', 'b2'),
    ('москва туристская 25 стр.1', 's1'),
    ('москва туристская 26 к1', 'other'),
    ('москва туристская 25к', 'letter'),
    ('москва туристская 25/1', 'fraction'),
])
def test_numeric_identities_stay_distinct(matcher, query, expected):
    result = matcher.match_one(query)
    assert result.status == 'accepted', result
    assert result.best_id == expected


@pytest.mark.parametrize('query', [
    'москва туристская 25',
    'москва туристская 25 к9',
    'москва туристская',
    'москва туристская неизвестная 25 к1',
    'москва туристская 25 к1 неизвестная',
    'москва туристская 25 к1 к2',
])
def test_incomplete_unknown_and_conflicting_not_accepted(matcher, query):
    assert matcher.match_one(query).status != 'accepted'


@pytest.mark.parametrize('street', ['8 Марта', '1905 года'])
def test_street_number_is_not_house(street):
    m = HybridAddressMatcher().fit([f'г. Москва, улица {street}, д. 7 к1'])
    assert not m.parser.parse('москва '+street).house
    result = m.match_one('москва '+street+' 7 к1')
    assert result.parsed['house'] == '7'
    assert result.parsed['building'] == '1'
    assert result.status == 'accepted'


@pytest.mark.parametrize('value', ['25 к1', '25к1', '25 к. 1', '25а к1', '25/2 к1'])
def test_normalization_idempotent(value):
    normalized = normalize_ru_address(value)
    assert normalize_ru_address(normalized) == normalized
    parsed = parse_basic('д. '+value)
    assert parsed.building == '1'
