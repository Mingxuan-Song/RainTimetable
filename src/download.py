from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import time
import urllib.request

REVISION = '18f4efa2a0fcab0a2bd74203896e8c21d9f95ebb'
BASE = 'https://huggingface.co/datasets/TaoDerong/GreatPlains-Multisource-2000-2024'
FILES = {
    'GreatPlains_CHIRPS_DailyPrecip_2000_2024.csv': ['date', 'precip_daily'],
    'GreatPlains_MOD09A1_NDVI_8day_2000_2024.csv': ['date', 'ndvi'],
    'GreatPlains_ERA5L_SoilTempPET_Daily_2000_2024.csv': ['date', 'temp2m_daily_K'],
    'README.md': [],
}

def download_data(root, refresh=False):
    """Read response bytes, check content, and record URLs and SHA-256 hashes.

    Existing snapshots are checked against their recorded hashes. A refresh
    issues new GET requests; cached runs never claim a new HTTP verification.
    """
    import pandas as pd
    root = Path(root)
    raw = root / 'data' / 'raw'
    raw.mkdir(parents=True, exist_ok=True)
    manifest_path = raw / 'download_manifest.json'
    old = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    records = []
    for name, required in FILES.items():
        path = raw / name
        url = f'{BASE}/resolve/{REVISION}/{name}'
        cached = path.exists() and not refresh
        if not cached:
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(url, timeout=90) as response:
                        status = response.status
                        content_type = response.headers.get('Content-Type', '')
                        content = response.read()
                    if status != 200 or not content or content.lstrip().startswith(b'<'):
                        raise ValueError(f'Unexpected response for {name}: {status}')
                    path.write_bytes(content)
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(2 ** attempt)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        previous = next((x for x in old.get('files', []) if x['file'] == name), None)
        if cached and previous and previous['sha256'] != digest:
            raise ValueError(f'Cached file has changed: {name}')
        rows = None
        if required:
            frame = pd.read_csv(path)
            if not set(required).issubset(frame.columns) or frame.empty:
                raise ValueError(f'Invalid CSV schema: {name}')
            rows = len(frame)
        records.append(dict(file=name, url=url, bytes=path.stat().st_size,
                            sha256=digest, rows=rows,
                            http_status=previous.get('http_status') if cached and previous else (None if cached else status),
                            verified_at=previous.get('verified_at') if cached and previous else (None if cached else datetime.now(timezone.utc).isoformat()),
                            access='cached' if cached else 'downloaded'))
    result = dict(dataset=BASE, revision=REVISION, license='CC-BY-4.0', files=records)
    manifest_path.write_text(json.dumps(result, indent=2))
    return pd.DataFrame(records)

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--refresh', action='store_true')
    args = p.parse_args()
    print(download_data(Path(__file__).resolve().parents[1], args.refresh).to_string(index=False))
