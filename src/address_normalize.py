"""Conservative normalization: preserve numeric suffixes, slashes and hyphens."""
import re

ABBREVIATIONS = {
    'пр-т': 'проспект', 'просп': 'проспект', 'ул': 'улица',
    'пер': 'переулок', 'ш': 'шоссе', 'наб': 'набережная', 'пл': 'площадь',
    'обл': 'область', 'респ': 'республика', 'р-н': 'район',
    'г': 'город', 'пос': 'поселок', 'д': 'дом', 'корп': 'корпус',
    'стр': 'строение', 'кв': 'квартира',
}

def normalize_ru_address(text):
    if text is None:
        return ''
    s = str(text).lower().replace('ё', 'е').strip()
    s = re.sub(r'\s*[-–—]\s*', '-', s)
    # Dotted markers have a reliable boundary even when the next space is missing.
    for short, full in sorted(ABBREVIATIONS.items(), key=lambda x: -len(x[0])):
        s = re.sub(r'(?<![а-яa-z])' + re.escape(short) + r'\.', full + ' ', s)
        s = re.sub(r'(?<![а-яa-z])' + re.escape(short) + r'(?![а-яa-z])', full + ' ', s)
    s = re.sub(r'(?<=\d)(?=корпус|строение|квартира)', ' ', s)
    # OSM also writes a separated short marker: '25 к1', '25 к. 1'.
    # Bind it to a preceding numeric house token; do not expand arbitrary 'к'.
    house_token = r'(?<![а-яa-z0-9])(\d+[а-яa-z]?(?:[/\-]\d+[а-яa-z]?)?)'
    s = re.sub(house_token + r'\s+к\.?\s*(?=\d)', r'\1 корпус ', s)
    s = re.sub(house_token + r'\s+с\.?\s*(?=\d)', r'\1 строение ', s)
    s = re.sub(r'(?<=\d)к(?=\d)', ' корпус ', s)
    s = re.sub(r'(?<=\d)с(?=\d)', ' строение ', s)
    s = re.sub(r'[.;:]', ' ', s)
    return re.sub(r'\s+', ' ', s).strip(' ,')


def compact(text):
    return re.sub(r'\s+', '', normalize_ru_address(text))
