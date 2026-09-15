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
    s = re.sub(r'(?<=\d)к(?=\d)', ' корпус ', s)
    s = re.sub(r'(?<=\d)с(?=\d)', ' строение ', s)
    s = re.sub(r'[.;:]', ' ', s)
    return re.sub(r'\s+', ' ', s).strip(' ,')


def compact(text):
    return re.sub(r'\s+', '', normalize_ru_address(text))
