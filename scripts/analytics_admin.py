"""Private Turso maintenance. Credentials come from environment, never CLI flags."""
import argparse
import os
from pathlib import Path
from pricing.analytics import TursoStore, valid_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['init', 'check', 'purge', 'export', 'delete-visitor'])
    parser.add_argument('--visitor-id')
    parser.add_argument('--output', default='private-analytics')
    args = parser.parse_args()
    store = TursoStore(os.environ['TURSO_DATABASE_URL'], os.environ['TURSO_AUTH_TOKEN'])
    store.initialize()
    if args.command == 'purge':
        store.purge()
    elif args.command == 'check':
        # Counts only. Do not print row-level visitor/query records or credentials.
        for table in ('visitors', 'pricing_queries'):
            result = store.execute('SELECT COUNT(*) FROM ' + table)
            print(table, result['rows'][0][0]['value'])
    elif args.command == 'delete-visitor':
        if not valid_id(args.visitor_id):
            parser.error('A valid --visitor-id is required')
        store.execute('DELETE FROM pricing_queries WHERE visitor_id=?', [args.visitor_id])
        store.execute('DELETE FROM visitors WHERE visitor_id=?', [args.visitor_id])
    elif args.command == 'export':
        import pandas as pd
        import pyarrow as pa
        import pyarrow.parquet as pq
        folder = Path(args.output)
        folder.mkdir(parents=True, exist_ok=True, mode=0o700)
        for table, key in [('visitors', 'visitor_id'), ('pricing_queries', 'query_id')]:
            last = ''; writer = None
            target = folder / (table + '.parquet')
            try:
                while True:
                    result = store.execute(f'SELECT * FROM {table} WHERE {key}>? ORDER BY {key} LIMIT 1000', [last])
                    columns = [col['name'] for col in result['cols']]
                    rows = [[cell.get('value') for cell in row] for row in result['rows']]
                    if not rows:
                        break
                    # Stable string schema handles nullable city/country across chunks.
                    frame = pd.DataFrame(rows, columns=columns).astype('string')
                    arrow = pa.Table.from_pandas(frame, preserve_index=False)
                    if writer is None:
                        writer = pq.ParquetWriter(str(target), arrow.schema, compression='zstd')
                        os.chmod(target, 0o600)
                    writer.write_table(arrow)
                    last = frame[key].iloc[-1]
            finally:
                if writer:
                    writer.close()
            if writer is None:
                pd.DataFrame(columns=columns).astype('string').to_parquet(target, index=False)
                os.chmod(target, 0o600)
    print('Analytics maintenance complete.')


if __name__ == '__main__':
    main()
