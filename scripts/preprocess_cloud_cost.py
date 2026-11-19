"""Preprocess FinOps-Copilot cloud billing data without removing cost anomalies.
Run from repository root: python scripts/preprocess_cloud_cost.py
"""
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data' / 'raw' / 'cloud_cost_usage.csv'
OUTPUT = ROOT / 'data' / 'processed' / 'cloud_cost_usage_cleaned.csv'
REPORT = ROOT / 'reports' / 'preprocessing_report.json'
REQUIRED = ['date', 'team', 'service', 'region', 'resource_id', 'environment',
            'instance_type', 'usage_amount', 'unit_cost', 'cost', 'month']
NUMERIC = ['usage_amount', 'unit_cost', 'cost']


def main():
    if not SOURCE.exists():
        raise FileNotFoundError(f'Missing input file: {SOURCE}')
    df = pd.read_csv(SOURCE, low_memory=False)
    missing_columns = sorted(set(REQUIRED) - set(df.columns))
    if missing_columns:
        raise ValueError(f'Missing required columns: {missing_columns}')

    report = {
        'source': str(SOURCE.relative_to(ROOT)),
        'original_rows': int(len(df)),
        'original_columns': int(len(df.columns)),
        'missing_values_before': {k: int(v) for k, v in df.isna().sum().items()},
        'duplicate_rows_before': int(df.duplicated().sum()),
    }

    # Normalize textual fields; missing instance type may be legitimate for non-VM services.
    text_cols = ['team', 'service', 'region', 'resource_id', 'environment', 'instance_type']
    for col in text_cols:
        df[col] = df[col].astype('string').str.strip().replace('', pd.NA)
    report['missing_instance_type_before'] = int(df['instance_type'].isna().sum())
    df['instance_type'] = df['instance_type'].fillna('Not Applicable')

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Do not treat legitimate zero/negative costs as invalid automatically:
    # discounts, refunds and credits can be meaningful in billing exports.
    report['zero_cost_rows_before'] = int(df['cost'].eq(0).sum())
    report['negative_cost_rows_before'] = int(df['cost'].lt(0).sum())
    report['negative_usage_rows_before'] = int(df['usage_amount'].lt(0).sum())
    report['negative_unit_cost_rows_before'] = int(df['unit_cost'].lt(0).sum())
    report['invalid_date_rows'] = int(df['date'].isna().sum())
    report['invalid_numeric_rows'] = int(df[NUMERIC].isna().any(axis=1).sum())

    # Rows without core identifiers or valid date/numbers cannot be used reliably.
    required_non_null = ['date', 'team', 'service', 'region', 'resource_id',
                         'environment'] + NUMERIC
    bad = df[required_non_null].isna().any(axis=1)
    report['rows_removed_invalid_required_fields'] = int(bad.sum())
    df = df.loc[~bad].copy()

    report['duplicates_removed'] = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()

    # Derive month from parsed date; avoid inconsistent month/date values.
    df['month'] = df['date'].dt.strftime('%Y-%m')
    df = df.sort_values(['date', 'service', 'resource_id'], kind='stable').reset_index(drop=True)
    df['date'] = df['date'].dt.strftime('%Y-%m-%d')

    report['cleaned_rows'] = int(len(df))
    report['missing_values_after'] = {k: int(v) for k, v in df.isna().sum().items()}
    report['zero_cost_rows_after'] = int(df['cost'].eq(0).sum())
    report['negative_cost_rows_after'] = int(df['cost'].lt(0).sum())
    report['date_range'] = [df['date'].min(), df['date'].max()] if len(df) else []
    report['cost_sum_before_cleaning'] = float(pd.to_numeric(pd.read_csv(SOURCE)['cost'], errors='coerce').sum())
    report['cost_sum_after_cleaning'] = float(df['cost'].sum())
    report['note'] = ('Unusual high costs and zero/negative costs are not automatically removed; '
                      'anomalies and possible credits require separate investigation.')

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False)
    REPORT.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    print(f'\nCleaned CSV: {OUTPUT}\nReport: {REPORT}')


if __name__ == '__main__':
    main()
