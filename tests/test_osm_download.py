"""Network-free integration tests: paths, provenance, failures and matcher schema."""
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from scripts.download_reference_osm import ROOT, collect, main, normalize_features
from src.hybrid import HybridAddressMatcher

PLACE = {'query': 'Район, Москва, Россия', 'city': 'Москва', 'region': ''}


def features():
    return pd.DataFrame({
        'addr:street': ['улица Ленина', 'улица Ленина', 'улица Ленина', None, ' '],
        'addr:housenumber': ['10к2', '10к2', '10/2', '7', '3'],
        'addr:city': [None, None, None, None, None],
    }, index=pd.MultiIndex.from_tuples([('way', i) for i in range(5)]))


def setup_args(tmp_path):
    places = tmp_path / 'places.json'
    places.write_text(json.dumps([PLACE, dict(PLACE, query='Второй район')]), encoding='utf-8')
    output = tmp_path / 'nested' / 'reference.csv'
    return ['--places', str(places), '--output', str(output)], output


def test_clean_numeric_identity_provenance_and_matcher():
    df, report = collect([PLACE], lambda *a, **kw: features(), log=lambda x: None)
    assert len(df) == 2 and report[0]['rows'] == 3
    assert set(df.house) == {'10', '10/2'}
    assert df.loc[df.house == '10', 'building'].iloc[0] == '2'
    assert df.loc[df.house == '10', 'osm_ids'].iloc[0] == 'way:0 | way:1'
    m = HybridAddressMatcher().fit(df)
    assert m.match_one('Москва, ул. Ленина 10к2').status == 'accepted'


def test_ids_stable_on_order_and_actual_city_preserved():
    f = features()
    f.loc[('way', 2), 'addr:city'] = 'Другой город'
    original = normalize_features(f, PLACE)
    reordered = normalize_features(f.iloc[::-1], PLACE)
    assert set(original.id) == set(reordered.id)
    assert 'Другой город' in set(original.city)


def test_download_from_unrelated_cwd_and_reuse_without_network(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args, output = setup_args(tmp_path)
    assert main(args, fetch=lambda *a, **kw: features()) == 0
    assert output.is_file()
    assert json.loads(output.with_suffix('.meta.json').read_text())['complete']
    def no_network(*a, **kw):
        pytest.fail('Cached run must not call the network')
    assert main(args, fetch=no_network) == 0


def test_failed_refresh_does_not_replace_good_csv(tmp_path):
    args, output = setup_args(tmp_path)
    assert main(args, fetch=lambda *a, **kw: features()) == 0
    before = output.read_bytes()
    metadata = output.with_suffix('.meta.json').read_bytes()
    def sometimes(query, **kwargs):
        if query == 'Второй район':
            raise TimeoutError('service unavailable')
        return features()
    assert main(args + ['--refresh'], fetch=sometimes) == 1
    assert output.read_bytes() == before
    assert output.with_suffix('.meta.json').read_bytes() == metadata
    assert output.with_name('reference.partial.csv').is_file()
    assert not json.loads(output.with_suffix('.attempt.json').read_text())['complete']


def test_explicit_partial_and_later_strict_reuse(tmp_path):
    args, output = setup_args(tmp_path)
    def sometimes(query, **kwargs):
        return features() if query == PLACE['query'] else pd.DataFrame()
    assert main(args + ['--allow-partial'], fetch=sometimes) == 0
    assert not json.loads(output.with_suffix('.meta.json').read_text())['complete']
    assert main(args, fetch=sometimes) == 1
    assert main(args + ['--allow-partial'], fetch=sometimes) == 0


def test_all_failures_do_not_create_empty_reference(tmp_path):
    args, output = setup_args(tmp_path)
    assert main(args, fetch=lambda *a, **kw: pd.DataFrame()) == 1
    assert not output.exists()
    assert output.with_suffix('.attempt.json').exists()


def test_changed_geography_does_not_silently_reuse_cache(tmp_path):
    args, output = setup_args(tmp_path)
    assert main(args, fetch=lambda *a, **kw: features()) == 0
    Path(args[1]).write_text(json.dumps([PLACE]), encoding='utf-8')
    assert main(args, fetch=lambda *a, **kw: features()) == 1


def test_cli_help_outside_project_without_osmnx(tmp_path):
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/download_reference_osm.py'), '--help'],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0
    assert '--refresh' in result.stdout


def test_colab_notebook_code_compiles():
    notebook = json.loads((ROOT / 'notebooks/colab_quickstart.ipynb').read_text())
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']), f'cell_{i}', 'exec')
