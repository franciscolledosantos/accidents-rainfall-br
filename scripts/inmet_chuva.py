"""Lê os zips anuais do INMET (estações automáticas) e extrai a chuva horária (UTC) por estação."""
import zipfile, io, sys, os, re
import pandas as pd, numpy as np
D = os.path.join(os.path.dirname(__file__), '..', 'dados')
zips = sys.argv[1:]
meta, series = [], []
for z in zips:
    with zipfile.ZipFile(z) as zf:
        for n in zf.namelist():
            if not n.upper().endswith('.CSV'):
                continue
            raw = zf.read(n).decode('latin1')
            lines = raw.splitlines()
            h = {}
            for l in lines[:8]:
                k, _, v = l.partition(';')
                h[k.strip().rstrip(':').upper()] = v.strip()
            def f(x):
                try: return float(x.replace(',', '.'))
                except: return np.nan
            cod = h.get('CODIGO (WMO)', h.get('CÓDIGO (WMO)', ''))
            lat = f(h.get('LATITUDE', 'nan')); lon = f(h.get('LONGITUDE', 'nan'))
            meta.append(dict(cod=cod, uf=h.get('UF', ''), estacao=h.get('ESTACAO', h.get('ESTAÇÃO', '')), lat=lat, lon=lon,
                             alt=f(h.get('ALTITUDE', 'nan'))))
            d = pd.read_csv(io.StringIO('\n'.join(lines[8:])), sep=';', dtype=str, usecols=[0, 1, 2])
            d.columns = ['data', 'hora', 'p']
            data = d['data'].str.replace('/', '-', regex=False)
            hora = d['hora'].str.extract(r'(\d{2}):?(\d{2})')
            t = pd.to_datetime(data + ' ' + hora[0] + ':' + hora[1], format='%Y-%m-%d %H:%M', errors='coerce')
            p = pd.to_numeric(d['p'].str.replace(',', '.', regex=False), errors='coerce')
            p[(p < 0) | (p > 200)] = np.nan
            s = pd.DataFrame({'cod': cod, 't_utc': t, 'p': p.astype('float32')}).dropna(subset=['t_utc'])
            series.append(s)
    print(z, len(series), flush=True)
m = pd.DataFrame(meta).drop_duplicates('cod')
s = pd.concat(series, ignore_index=True).drop_duplicates(['cod', 't_utc'])
s['cod'] = s['cod'].astype('category')
tag = '_'.join(re.findall(r'(\d{4})', ' '.join(os.path.basename(z) for z in zips)))
m.to_csv(os.path.join(D, f'inmet_estacoes_{tag}.csv'), index=False)
s.to_parquet(os.path.join(D, f'inmet_chuva_horaria_{tag}.parquet'), index=False)
print('estacoes', len(m), 'registros', len(s), 'faltantes %.1f%%' % (100 * s['p'].isna().mean()))
