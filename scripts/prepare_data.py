"""
Check a copy of the Banana Quality dataset against the file used in the study, then
save it as data/banana_quality.csv for the analysis scripts.

Usage:
    python scripts/prepare_data.py path/to/banana_quality.csv   (or .xlsx)

Two checks are reported:
  1. File SHA-256. Matches only if you have the identical file used in the study
     (an .xlsx export of the Kaggle file).
  2. Content fingerprint. Values rounded to 4 decimals and hashed, so a CSV or
     XLSX copy of the same data passes even if the file format differs.
The content fingerprint is the check that matters for reproducing the results.
"""
import sys, os, hashlib
import pandas as pd

FILE_SHA256 = '9579b0e209572d62c8dc3657f884bec2ae17be85e121e12c4e5180ddb45c868d'
CONTENT_SHA256 = 'e37ccdfe38232aa5bd6acd6bdbe9b474ab7229da6f90d426a7cdf83fd7a135a6'
FEATURES = ['Size', 'Weight', 'Sweetness', 'Softness', 'HarvestTime', 'Ripeness', 'Acidity']


def main(path):
    raw = open(path, 'rb').read()
    file_hash = hashlib.sha256(raw).hexdigest()
    df = pd.read_excel(path) if path.lower().endswith(('.xlsx', '.xls')) else pd.read_csv(path)
    missing = [c for c in FEATURES + ['Quality'] if c not in df.columns]
    if missing:
        sys.exit(f'Missing columns: {missing}')
    df['Quality'] = df['Quality'].astype(str).str.strip()
    s = df[FEATURES].round(4).to_csv(index=False, float_format='%.4f') + ''.join(df['Quality'])
    content_hash = hashlib.sha256(s.encode()).hexdigest()

    print(f'Rows: {len(df)} (expected 8000)')
    print(f'Classes: {df["Quality"].value_counts().to_dict()} (expected Good 4006, Bad 3994)')
    print(f'File SHA-256 match:        {file_hash == FILE_SHA256}')
    print(f'Content fingerprint match: {content_hash == CONTENT_SHA256}')
    if content_hash != CONTENT_SHA256:
        print('WARNING: this copy differs from the data used in the study; results will not match exactly.')

    os.makedirs('data', exist_ok=True)
    df[FEATURES + ['Quality']].to_csv('data/banana_quality.csv', index=False)
    print('Saved data/banana_quality.csv')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
