"""Download new GUISyn mobile screens without changing the original dataset."""
import argparse
import hashlib
import io
import json
from pathlib import Path

import requests
from PIL import Image
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parents[1]


def fingerprint(image):
    rgb = image.convert('RGB')
    return hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--count', type=int, default=256)
    args = parser.parse_args()
    dest = ROOT / 'data/guisyn-additional'
    dest.mkdir(parents=True, exist_ok=True)
    manifest = dest / 'source_annotations.json'
    original = json.loads((ROOT / 'data/guisyn/source_annotations.json').read_text())
    sources = json.loads(manifest.read_text()) if manifest.exists() else []
    ids, hashes = set(), set()
    for source in original + sources:
        path = Path(source['image']) if 'image' in source else ROOT / 'data/guisyn' / f"{source['row_idx']:05d}.png"
        with Image.open(path) as image:
            digest = fingerprint(image)
        if source['id'] in ids or (source in sources and digest in hashes):
            raise RuntimeError('Duplicate saved screenshot')
        ids.add(source['id'])
        hashes.add(digest)
    session = requests.Session()
    retry = Retry(total=5, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retry))
    offset = max(s['row_idx'] for s in original + sources) + 1
    skipped = 0
    while len(sources) < args.count:
        response = session.get('https://datasets-server.huggingface.co/rows', params={
            'dataset': 'allenai/MolmoPoint-GUISyn', 'config': 'mobile',
            'split': 'train', 'offset': offset, 'length': min(32, args.count - len(sources))}, timeout=120)
        response.raise_for_status()
        rows = response.json()['rows']
        if not rows:
            raise RuntimeError('Dataset exhausted before requested count')
        for entry in rows:
            offset = entry['row_idx'] + 1
            if entry.get('truncated_cells'):
                raise RuntimeError('Truncated annotation')
            row = entry['row']
            if row['id'] in ids:
                skipped += 1
                continue
            annotation = row['annotation']
            if isinstance(annotation, str):
                annotation = json.loads(annotation)
            image_response = session.get(row['image']['src'], timeout=120)
            image_response.raise_for_status()
            with Image.open(io.BytesIO(image_response.content)) as image:
                digest = fingerprint(image)
                w, h = image.size
                if digest in hashes or not any(
                    any(t.strip() for t in a.get('intent', [])) and
                    0 <= a['x_center'] < w and 0 <= a['y_center'] < h and
                    a.get('width', 0) > 0 and a.get('height', 0) > 0 for a in annotation):
                    skipped += 1
                    continue
                path = dest / f"{entry['row_idx']:05d}.png"
                image.convert('RGB').save(path)
            sources.append({'row_idx': entry['row_idx'], 'id': row['id'],
                            'annotation': annotation, 'image': str(path), 'pixel_sha256': digest})
            ids.add(row['id'])
            hashes.add(digest)
            temporary = manifest.with_suffix('.tmp')
            temporary.write_text(json.dumps(sources, ensure_ascii=False))
            temporary.replace(manifest)
            print(f'prepared {len(sources)}/{args.count}; skipped={skipped}; row={entry["row_idx"]}', flush=True)
    print(f'Complete: {manifest}', flush=True)


if __name__ == '__main__':
    main()
