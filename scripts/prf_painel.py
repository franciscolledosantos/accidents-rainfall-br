"""Monta a base de acidentes da PRF (datatran, por ocorrência), 2017-2025."""
import pandas as pd, numpy as np, glob, os
D = os.path.join(os.path.dirname(__file__), '..', 'dados')
fs = sorted(glob.glob(os.path.join(D, 'prf', 'datatran20*.csv')))
dfs = []
for f in fs:
    d = pd.read_csv(f, sep=';', encoding='latin1', dtype=str, na_values=['NA', '(null)', ''])
    d['arquivo'] = os.path.basename(f)
    dfs.append(d)
df = pd.concat(dfs, ignore_index=True)
num = lambda s: pd.to_numeric(s.str.replace(',', '.', regex=False), errors='coerce')
for c in ['latitude', 'longitude', 'km']:
    df[c] = num(df[c])
for c in ['pessoas', 'mortos', 'feridos_leves', 'feridos_graves', 'ilesos', 'feridos', 'veiculos']:
    df[c] = pd.to_numeric(df[c], errors='coerce')
# data: 2017-2019 arquivos usam dd/mm/aaaa em alguns anos
dt = pd.to_datetime(df['data_inversa'], format='%Y-%m-%d', errors='coerce')
m = dt.isna()
dt[m] = pd.to_datetime(df.loc[m, 'data_inversa'], format='%d/%m/%Y', errors='coerce')
hh = pd.to_timedelta(df['horario'], errors='coerce')
df['datahora'] = dt + hh
df['hora_utc'] = (df['datahora'] + pd.Timedelta(hours=3)).dt.floor('h')  # horário de Brasília (UTC-3); INMET usa UTC
ok = df['datahora'].notna() & df['latitude'].between(-34, 6) & df['longitude'].between(-74, -34)
print('total', len(df), 'validos', ok.sum())
df = df[ok].copy()
df['ano'] = df['datahora'].dt.year
df['grave'] = ((df['mortos'] > 0) | (df['feridos_graves'] > 0)).astype(int)
df['fatal'] = (df['mortos'] > 0).astype(int)
df['chuva_prf'] = df['condicao_metereologica'].str.contains('Chuva|Garoa', case=False, na=False).astype(int)
cols = ['id', 'datahora', 'hora_utc', 'ano', 'uf', 'br', 'km', 'municipio', 'causa_acidente', 'tipo_acidente',
        'classificacao_acidente', 'fase_dia', 'condicao_metereologica', 'tipo_pista', 'tracado_via', 'uso_solo',
        'pessoas', 'mortos', 'feridos_graves', 'feridos_leves', 'veiculos', 'latitude', 'longitude', 'grave', 'fatal', 'chuva_prf']
df = df[cols].drop_duplicates('id')
df.to_parquet(os.path.join(D, 'prf_acidentes_2017_2025.parquet'), index=False)
print(df.groupby('ano').size())
print(df['condicao_metereologica'].value_counts().head(10))
