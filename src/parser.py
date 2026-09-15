"""Rule-based component parser with reference-derived gazetteer.

No global replacement of historic names. Aliases apply to the city field only.
Unknown/ambiguous components remain visible and block automatic acceptance.
"""
from dataclasses import dataclass, asdict, field
import re
from .address_normalize import normalize_ru_address, compact

FIELDS = ('region', 'city', 'district', 'street_type', 'street', 'house', 'building', 'structure', 'apartment')
STREET_TYPES = r'улица|проспект|переулок|шоссе|набережная|площадь|бульвар|проезд|тупик'
MARKERS = rf'область|край|республика|город|поселок|район|{STREET_TYPES}|дом|корпус|строение|квартира'
NUMBER = r'\d+[а-яa-z]?(?:[/\-]\d+[а-яa-z]?)?'

@dataclass
class Address:
    region: str = ''
    city: str = ''
    district: str = ''
    street_type: str = ''
    street: str = ''
    house: str = ''
    building: str = ''
    structure: str = ''
    apartment: str = ''
    normalized: str = ''
    warnings: list = field(default_factory=list)
    def components(self):
        return {k: getattr(self, k) for k in FIELDS}
    def to_dict(self):
        return asdict(self)


def parse_basic(text):
    s = normalize_ru_address(text)
    a = Address(normalized=s)
    # Add boundaries around full numeric markers without splitting city suffixes.
    s = re.sub(r'(дом|корпус|строение|квартира)(?=\d)', r'\1 ', s)
    for key, marker in [('house','дом'),('building','корпус'),('structure','строение'),('apartment','квартира')]:
        values = re.findall(r'\b' + marker + r'\s*(' + NUMBER + r')(?![а-яa-z0-9])', s)
        if len(set(values)) > 1:
            a.warnings.append('conflicting_' + key)
        if values:
            setattr(a, key, values[0])
    chunks = [x.strip() for x in s.split(',') if x.strip()]
    for chunk in chunks:
        # Geography before city/street. Region/district types can follow the name.
        m = re.search(r'^(.+?\s+(?:область|край|республика))\b|^(республика\s+[^,]+?)(?=\s+(?:город|поселок|'+STREET_TYPES+r')\b|$)', chunk)
        if m:
            a.region = m.group(0).strip()
        m = re.search(r'^(.+?\s+район)\b', chunk)
        if m:
            a.district = m.group(1)
        m = re.search(r'\b(?:город|поселок)\s+(.+?)(?=\s+(?:'+MARKERS+r')\b|,|$)', chunk)
        if m:
            a.city = m.group(1).strip()
        m = re.search(r'\b('+STREET_TYPES+r')\s+(.+?)(?=\s+(?:дом|корпус|строение|квартира)\b|$)', chunk)
        if m and not re.fullmatch(NUMBER,m.group(2).strip()):
            a.street_type, a.street = m.group(1), m.group(2).strip()
        else:
            m = re.search(r'^(.+?)\s+('+STREET_TYPES+r')\b', chunk)
            if m:
                a.street, a.street_type = m.group(1).strip(), m.group(2)
                tail=chunk[m.end():].strip()
                if not a.house and re.fullmatch(NUMBER,tail):
                    a.house=tail
        if a.street:
            # Bare final number after a street: ул. Ленина 25. Keep "улица 8 марта".
            m = re.search(r'\s+('+NUMBER+r')$', a.street)
            if m and not a.house:
                a.house = m.group(1)
                a.street = a.street[:m.start()].strip()
    # Legacy OSM serialization: "Невский проспект, 5, Санкт-Петербург".
    if not a.house:
        for chunk in chunks:
            if re.fullmatch(NUMBER, chunk):
                a.house = chunk
                break
    if not a.city and len(chunks) >= 3 and re.fullmatch(NUMBER, chunks[-2]):
        if not re.search(r'\b('+MARKERS+r')\b', chunks[-1]):
            a.city = chunks[-1]
    return a


def _pattern(value):
    # Optional spaces inside every name; numeric/hyphen boundaries remain strict.
    chars = list(compact(value))
    return re.compile(r'(?<![а-яa-z0-9-])' + r'\s*'.join(map(re.escape, chars)) + r'(?![а-яa-z0-9-])')

class AddressParser:
    def __init__(self, references, aliases=()):
        self.aliases = list(aliases)
        self.references = list(references)
        self.lexicon = {}
        self.glued = {}
        for key in ('region','city','district','street'):
            values = {getattr(a, key) for a in references if getattr(a, key)}
            ordered = sorted(values, key=lambda x: (-len(x), x))
            self.lexicon[key] = [(v, _pattern(v)) for v in ordered]
            lookup = {compact(v): v for v in ordered}
            if lookup:
                names = '|'.join(re.escape(v) for v in lookup)
                prefix = r'(?<![а-яa-z0-9-])(?:город\s*|поселок\s*|улица\s*|проспект\s*|переулок\s*)?'
                self.glued[key] = (re.compile('('+prefix+')('+names+r')(?=(?:'+MARKERS+r')|\d|,|\s|$)'), lookup)

        dense_names=sorted({compact(v) for entries in self.lexicon.values() for v,_ in entries},key=lambda x:(-len(x),x))
        alternatives='|'.join(map(re.escape,dense_names)) or r'(?!)'
        prefix=r'(?<![а-яa-z0-9-])(?:город|поселок|улица|проспект|переулок)?'
        suffix=r'(?=г\.|д\.|корп\.|стр\.|'+MARKERS+'|'+alternatives+r'|,|$)'
        self.raw_boundaries=re.compile('('+prefix+')('+alternatives+')'+suffix)

    def parse(self, text):
        # Recover marker boundaries after dictionary-known names before dots are removed.
        raw = '' if text is None else str(text).lower().replace('ё','е')
        for _ in range(2):
            raw = self.raw_boundaries.sub(lambda m: m.group(1)+' '+m.group(2)+' ',raw)
        a = parse_basic(raw)
        s = a.normalized
        # Expand full markers attached to text: городКрасноярскулицаЛенинадом25.
        for marker in sorted(MARKERS.split('|'), key=len, reverse=True):
            s = re.sub(r'(?<![а-я])' + marker + r'(?=[а-я0-9])', marker+' ', s)
        # Place dictionary name boundaries only at the beginning or after markers.
        # Repeated passes support multiple glued components.
        for _ in range(2):
            for pattern, lookup in self.glued.values():
                s = pattern.sub(lambda m: m.group(1)+' '+lookup[m.group(2)]+' ', s)
        b = parse_basic(s)
        for key in FIELDS:
            if getattr(b,key):
                setattr(a,key,getattr(b,key))
        a.warnings = list(dict.fromkeys(a.warnings+b.warnings))
        # Exact gazetteer lookup, longest overlapping name wins.
        for key in ('region','city','district','street'):
            search_space = getattr(a,key) or s
            if key == 'city' and not a.city:
                search_space = re.split(r'\b('+STREET_TYPES+r')\b', s)[0]
            found = []
            occupied = []
            for value, pattern in self.lexicon[key]:
                for m in pattern.finditer(search_space):
                    if not any(m.start()<hi and m.end()>lo for lo,hi in occupied):
                        found.append(value)
                        occupied.append(m.span())
            found = list(dict.fromkeys(found))
            if len(found) == 1:
                # Never turn explicit "Красноярск-45" into "Красноярск".
                setattr(a,key,found[0])
            elif len(found) > 1:
                a.warnings.append('ambiguous_' + key)
        # Contextual historical city aliases, optionally constrained by region.
        city_space = a.city or re.split(r'\b('+STREET_TYPES+r')\b',s)[0]
        targets = []
        for alias in self.aliases:
            old, new = normalize_ru_address(alias['alias']), normalize_ru_address(alias['city'])
            region = normalize_ru_address(alias.get('region',''))
            if _pattern(old).search(city_space) and (not a.region or not region or a.region == region):
                targets.append(new)
        if len(set(targets)) == 1:
            a.city = targets[0]
        elif len(set(targets)) > 1:
            a.warnings.append('ambiguous_city_alias')
        # Detect unique shortened street names only against the reference dictionary.
        if a.street and a.street not in {v for v,_ in self.lexicon['street']}:
            tokens = a.street.split()
            context_names = {r.street for r in self.references
                             if all(not getattr(a,k) or getattr(a,k)==getattr(r,k)
                                    for k in ('region','city','street_type'))}
            candidates = [v for v in sorted(context_names) if len(v.split())==len(tokens)
                          and all(len(t)>=3 and w.startswith(t) for t,w in zip(tokens,v.split()))]
            if len(candidates)==1:
                a.street = candidates[0]
            elif len(candidates)>1:
                a.warnings.append('ambiguous_street_abbreviation')
        # Unknown bare city before an explicit street must not silently disappear.
        if not a.city:
            prefix = re.split(r'\b('+STREET_TYPES+r')\b',s)[0].strip(' ,')
            for key in ('region','district'):
                if getattr(a,key):
                    prefix = prefix.replace(getattr(a,key),'').strip(' ,')
            if prefix and re.search('[а-я]',prefix) and not re.search(r'\b('+MARKERS+r')\b',prefix):
                a.city = prefix
        return a
