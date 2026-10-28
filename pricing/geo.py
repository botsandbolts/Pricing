"""Local GeoLite2 City / DB-IP City Lite lookup; no remote IP lookup calls."""
import ipaddress
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=2)
def _reader(path, modified):
    import maxminddb
    return maxminddb.open_database(path)


def approximate_location(headers, ip_header='', database='', provider=''):
    empty = {'country': None, 'city': None, 'region': None, 'geo_provider': None}
    if not ip_header or not database or provider not in {'MaxMind GeoLite2', 'DB-IP Lite'}:
        return empty
    try:
        # Accept only a single IP from an explicitly trusted proxy-overwritten header.
        # X-Forwarded-For chains are deliberately rejected rather than guessed.
        ip = ipaddress.ip_address(str(headers.get(ip_header, '')).strip())
        if not ip.is_global:
            return empty
        file = Path(database)
        row = _reader(str(file), file.stat().st_mtime_ns).get(str(ip)) or {}
        country = row.get('country', {}).get('iso_code')
        if not isinstance(country, str) or len(country) != 2:
            country = None
        regions = row.get('subdivisions', [])
        def name(item):
            value = item.get('names', {}).get('en')
            return value[:100] if isinstance(value, str) else None
        return {'country': country, 'city': name(row.get('city', {})),
                'region': name(regions[0]) if regions else None, 'geo_provider': provider}
    except Exception:
        return empty
