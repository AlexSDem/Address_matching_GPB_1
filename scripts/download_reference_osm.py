"""Download a reusable OSM address reference. No hard-coded Colab path.

Run from project root: python scripts/download_reference_osm.py
OSMnx is optional: cached CSVs can be used without importing it.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from src.address_normalize import normalize_ru_address
from src.parser import parse_basic

IDENTITY = ('region', 'city', 'street_type', 'street', 'house', 'building', 'structure')
COLUMNS = ['id', 'address', 'united_addr', *IDENTITY, 'osm_ids', 'source_places']


def load_places(path):
    places = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(places, list) or not places:
        raise ValueError('Список районов должен быть непустым JSON-массивом.')
    for place in places:
        if not isinstance(place, dict) or any(
            not isinstance(place.get(k), str) or not place[k].strip() for k in ('query', 'city')
        ) or not isinstance(place.get('region', ''), str):
            raise ValueError('Для каждого района нужны query, city и необязательный region (строки).')
    return places


def normalize_features(features, place):
    """Keep real OSM strings; fallback city applies only when addr:city is absent."""
    required = ['addr:street', 'addr:housenumber']
    if not set(required).issubset(features.columns):
        raise ValueError('В ответе OSM нет addr:street или addr:housenumber.')
    frame = pd.DataFrame(features.drop(columns=['geometry'], errors='ignore'))
    rows = []
    for index, row in frame.iterrows():
        def value(key):
            raw = row.get(key, '')
            return '' if pd.isna(raw) else str(raw).strip()
        street_raw, house_raw = value('addr:street'), value('addr:housenumber')
        if not street_raw or not house_raw:
            continue
        street = parse_basic(street_raw)
        number = parse_basic('дом ' + house_raw)
        # Do not silently discard unsupported trailing text in a house number.
        # Preserve raw house unless the normalizer/parser can account for it fully.
        rebuilt = 'дом ' + number.house
        if number.building:
            rebuilt += ' корпус ' + number.building
        if number.structure:
            rebuilt += ' строение ' + number.structure
        can_split = bool(number.house) and normalize_ru_address('дом '+house_raw) == rebuilt
        city = value('addr:city') or place['city'].strip()
        region = value('addr:region') or place.get('region', '').strip()
        item = dict(region=region, city=city,
                    street_type=street.street_type,
                    street=street.street or street_raw,
                    house=number.house if can_split else house_raw,
                    building=number.building if can_split else '',
                    structure=number.structure if can_split else '')
        parts = [region, 'г. ' + city, street_raw, 'д. ' + house_raw]
        item['address'] = ', '.join(p for p in parts if p)
        item['united_addr'] = item['address']
        item['osm_ids'] = ':'.join(map(str, index)) if isinstance(index, tuple) else str(index)
        item['source_places'] = place['query']
        # Content-based identity, stable under row reorder, not an official address ID.
        key = json.dumps([normalize_ru_address(item[k]) for k in IDENTITY], ensure_ascii=False)
        item['id'] = 'osmaddr_' + hashlib.sha256(key.encode('utf-8')).hexdigest()
        rows.append(item)
    return pd.DataFrame(rows, columns=COLUMNS)


def collect(places, fetch, log=print):
    parts, outcomes = [], []
    for place in places:
        log('Загружаем: ' + place['query'])
        try:
            features = fetch(place['query'], tags={'building': True})
            frame = normalize_features(features, place)
            if frame.empty:
                raise ValueError('Нет непустых адресов с улицей и номером дома.')
            parts.append(frame)
            outcomes.append(dict(place=place, status='ok', rows=len(frame)))
            log(f'Получено адресов: {len(frame)}')
        except Exception as error:
            outcomes.append(dict(place=place, status='failed', error=f'{type(error).__name__}: {error}'))
            log(f'Не удалось загрузить {place["query"]}: {error}')
    if not parts:
        return pd.DataFrame(columns=COLUMNS), outcomes
    all_rows = pd.concat(parts, ignore_index=True)
    # Aggregate provenance for repeated buildings/overlapping district queries.
    aggregations = {c: 'first' for c in COLUMNS if c != 'id'}
    for key in ('osm_ids', 'source_places'):
        aggregations[key] = lambda values: ' | '.join(sorted(set(values)))
    result = all_rows.groupby('id', as_index=False, sort=True).agg(aggregations)
    return result[COLUMNS], outcomes


def validate_cache(path):
    data = pd.read_csv(path, dtype=str, keep_default_na=False)
    column = 'address' if 'address' in data else 'united_addr'
    if data.empty or column not in data or data[column].str.strip().eq('').any():
        raise ValueError('Существующий CSV пуст или не содержит непустые address/united_addr.')
    if 'id' in data and (data.id.duplicated().any() or data.id.str.strip().eq('').any()):
        raise ValueError('В существующем CSV пустые или повторяющиеся ID.')
    return data


def write_json(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def save_reference(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def main(argv=None, fetch=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--places', type=Path, default=ROOT / 'data/osm_places.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'reference_osm.csv')
    parser.add_argument('--refresh', action='store_true', help='Пересобрать CSV, минуя HTTP-кеш OSMnx.')
    parser.add_argument('--allow-partial', action='store_true', help='Явно разрешить сохранение неполной выгрузки.')
    parser.add_argument('--timeout', type=int, default=180, help='Таймаут одного HTTP-запроса, секунд.')
    args = parser.parse_args(argv)
    if args.timeout < 1:
        parser.error('--timeout должен быть положительным.')
    output = args.output.expanduser().resolve()
    metadata_path = output.with_suffix('.meta.json')
    try:
        places = load_places(args.places)
        if output.exists() and not args.refresh:
            data = validate_cache(output)
            if metadata_path.exists():
                metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
                if metadata.get('places') != places:
                    raise ValueError('Список районов изменился. Укажите другой --output или --refresh.')
                if not metadata.get('complete', False) and not args.allow_partial:
                    raise ValueError('Сохранённая выгрузка неполная. Нужен --refresh или --allow-partial.')
            else:
                print('У CSV нет отчёта выгрузки: его география и полнота не проверены.')
            print(f'Используется существующий справочник: {output} ({len(data)} адресов).')
            print('Чтобы собрать свежие данные, добавьте --refresh.')
            return 0
        if fetch is None:
            try:
                import osmnx as ox
            except ImportError as error:
                raise ValueError('Установите зависимости: python -m pip install -r requirements-osm.txt') from error
            ox.settings.use_cache = not args.refresh
            ox.settings.cache_folder = ROOT / 'cache/osm'
            ox.settings.requests_timeout = args.timeout
            fetch = ox.features_from_place
        frame, outcomes = collect(places, fetch)
        complete = all(item['status'] == 'ok' for item in outcomes)
        metadata = dict(created_at=datetime.now(timezone.utc).isoformat(),
                        source='OpenStreetMap contributors',
                        attribution='https://www.openstreetmap.org/copyright',
                        complete=complete, rows=len(frame), places=places, outcomes=outcomes)
        # A failed refresh never overwrites a previously usable CSV or its metadata.
        if frame.empty or (not complete and not args.allow_partial):
            attempt = output.with_suffix('.attempt.json')
            write_json(attempt, metadata)
            if not frame.empty:
                partial = output.with_name(output.stem + '.partial.csv')
                save_reference(frame, partial)
                print(f'Доступная часть сохранена отдельно: {partial}')
            print(f'Выгрузка не завершена. Основной CSV не изменён. Отчёт: {attempt}', file=sys.stderr)
            print('Повторите запуск позже; для неполной выгрузки используйте --allow-partial.', file=sys.stderr)
            return 1
        save_reference(frame, output)
        write_json(metadata_path, metadata)
        if not complete:
            print('ВНИМАНИЕ: сохранён неполный справочник; проверьте отчёт выгрузки.')
        print(f'Сохранено: {output}\nУникальных адресов: {len(frame)}\nОтчёт: {metadata_path}')
        return 0
    except (ValueError, OSError, TypeError) as error:
        print(f'Ошибка: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
