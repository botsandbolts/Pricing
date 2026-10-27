"""Verify and materialize repository-stored MiniLM weights without a network call."""
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import tempfile


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def backbone_directory(path, cache=None):
    """Return a normal Hugging Face directory, unpacking verified parts if needed."""
    path = Path(path)
    registry = path / 'weights.json'
    if not registry.exists():
        return path
    spec = json.loads(registry.read_text())
    config = (path / 'config.json').read_bytes()
    if hashlib.sha256(config).hexdigest() != spec['config_sha256']:
        raise ValueError('Transformer configuration checksum mismatch')
    root = Path(cache or os.environ.get('PRICING_CACHE_DIR', Path.home() / '.cache' / 'pricing'))
    destination = root / (spec['sha256'] + '-' + spec['config_sha256'][:16])
    weights = destination / 'model.safetensors'
    if weights.exists() and weights.stat().st_size == spec['bytes'] and sha256(weights) == spec['sha256']:
        if (destination / 'config.json').exists() and (destination / 'config.json').read_bytes() == config:
            return destination
    payload = bytearray()
    for part in spec['parts']:
        name = part['file']
        if Path(name).name != name:
            raise ValueError('Invalid transformer part path')
        data = (path / name).read_bytes()
        if len(data) != part['bytes'] or hashlib.sha256(data).hexdigest() != part['sha256']:
            raise ValueError('Transformer weight part checksum mismatch')
        payload.extend(data)
    if hashlib.sha256(payload).hexdigest() != spec['compressed_sha256']:
        raise ValueError('Transformer compressed checksum mismatch')
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='minilm-', dir=root) as temporary:
        stage = Path(temporary)
        (stage / 'config.json').write_bytes(config)
        with gzip.GzipFile(fileobj=io.BytesIO(payload)) as source, (stage / 'model.safetensors').open('wb') as target:
            shutil.copyfileobj(source, target, length=1024 * 1024)
        candidate = stage / 'model.safetensors'
        if candidate.stat().st_size != spec['bytes'] or sha256(candidate) != spec['sha256']:
            raise ValueError('Transformer decompressed checksum mismatch')
        destination.mkdir(parents=True, exist_ok=True)
        # Publish complete files only; concurrent readers never see partial weights.
        os.replace(stage / 'config.json', destination / 'config.json')
        os.replace(candidate, weights)
    return destination
