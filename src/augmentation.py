"""Component-preserving positives; identity-changing mutations are relabelled."""
import random
from .parser import parse_basic


def blocks(a):
    out=[]
    if a.region: out.append(a.region)
    if a.city: out.append('г. '+a.city)
    if a.district: out.append(a.district)
    if a.street: out.append((a.street_type or 'улица')+' '+a.street)
    if a.house:
        building='д. '+a.house
        if a.building: building+=' корп. '+a.building
        if a.structure: building+=' стр. '+a.structure
        if a.apartment: building+=' кв. '+a.apartment
        out.append(building)
    return out


def augment(a, seed=42):
    """Yield (category, query); preserve multiword names and numeric groups."""
    rng=random.Random(seed)
    parts=blocks(a)
    canonical=', '.join(parts)
    yield 'clean',canonical
    perm=parts.copy()
    rng.shuffle(perm)
    if perm==parts and len(perm)>1: perm=perm[1:]+perm[:1]
    yield 'component_permutation',', '.join(perm)
    yield 'missing_spaces',canonical.replace(' ','')
    short=canonical
    for full,abbr in [('улица','ул.'),('проспект','пр-т'),('область','обл.'),('переулок','пер.')]:
        short=short.replace(full,abbr)
    yield 'abbreviations',short
    if a.street and all(len(t)>=5 for t in a.street.split()):
        yield 'street_abbreviation',canonical.replace(a.street,' '.join(t[:4]+'.' for t in a.street.split()))
    yield 'incomplete',', '.join(x for x in parts if x not in (a.region,a.district))
    # Corrupt letters within the street only, not numeric identity.
    if a.street and len(a.street)>4:
        name=a.street
        i=next((i for i in range(1,len(name)-1) if name[i].isalpha() and name[i+1].isalpha() and name[i]!=name[i+1]),None)
        if i is not None:
            typo=name[:i]+name[i+1]+name[i]+name[i+2:]
            yield 'typo',canonical.replace(name,typo)


def swap_house_building(a):
    """NOT a positive augmentation: bindings change, even if tokens are identical."""
    if not a.house or not a.building or a.house==a.building:
        return None
    from dataclasses import replace
    return ', '.join(blocks(replace(a,house=a.building,building=a.house)))
