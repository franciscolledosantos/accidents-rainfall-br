"""Monta o conjunto case-crossover (estratificado no tempo) acidente PRF x chuva horária INMET.

Para cada acidente (caso): estação automática mais próxima (<= DMAX km) com dado válido;
controles = mesmas hora e dia da semana, no mesmo mês-calendário (hora local), na mesma estação.
Exposição: chuva acumulada na hora que contém o acidente (registro INMET das HH+1 UTC, que acumula HH..HH+1),
chuva na hora anterior e acumulados de 3 h e 24 h; dias secos antecedentes.
"""
import sys, os, glob
import numpy as np, pandas as pd
from sklearn.neighbors import BallTree

D = os.path.join(os.path.dirname(__file__), '..', 'dados')
DMAX = float(os.environ.get('DMAX', 25))
anos = [int(a) for a in sys.argv[1:]]

STD = {'AC': -5, 'AM': -4, 'RO': -4, 'RR': -4, 'MT': -4, 'MS': -4}
DST_UF = set('RS SC PR SP RJ ES MG GO DF MT MS'.split())
DST = [('2016-10-16', '2017-02-19'), ('2017-10-15', '2018-02-18'), ('2018-11-04', '2019-02-17')]


def to_utc(local, uf):
    off = uf.map(STD).fillna(-3).astype(float)
    dst = pd.Series(False, index=local.index)
    for a, b in DST:
        dst |= (local >= pd.Timestamp(a)) & (local < pd.Timestamp(b))
    off = off + (dst & uf.isin(DST_UF)).astype(float)
    return local - pd.to_timedelta(off, unit='h')


acc = pd.read_parquet(os.path.join(D, 'prf_acidentes_2017_2025.parquet'))
for ano in anos:
    a = acc[acc['ano'] == ano].copy()
    est = pd.read_csv(os.path.join(D, f'inmet_estacoes_{ano}.csv'), dtype={'cod': str}).dropna(subset=['lat', 'lon'])
    rain = pd.read_parquet(os.path.join(D, f'inmet_chuva_horaria_{ano}.parquet'))
    # inclui a última semana do ano anterior para os acumulados de 24 h e dias secos (se houver)
    prev = os.path.join(D, f'inmet_chuva_horaria_{ano-1}.parquet')
    if os.path.exists(prev):
        rp = pd.read_parquet(prev)
        rp = rp[rp['t_utc'] >= pd.Timestamp(f'{ano-1}-12-15')]
        rain = pd.concat([rp, rain], ignore_index=True)
    nxt = os.path.join(D, f'inmet_chuva_horaria_{ano+1}.parquet')
    if os.path.exists(nxt):   # primeiras horas do ano seguinte (acidentes na noite de 31/12)
        rn = pd.read_parquet(nxt)
        rn = rn[rn['t_utc'] < pd.Timestamp(f'{ano+1}-01-02')]
        rain = pd.concat([rain, rn], ignore_index=True)
    rain['cod'] = rain['cod'].astype(str)
    # grade horária completa por estação
    R = rain.pivot_table(index='t_utc', columns='cod', values='p', aggfunc='first')
    full = pd.date_range(R.index.min().floor('D'), R.index.max().ceil('D'), freq='h')
    R = R.reindex(full)
    R3 = R.rolling(3, min_periods=3).sum()
    R24 = R.rolling(24, min_periods=20).sum()
    daily = R.resample('D').sum(min_count=20)
    wet = (daily >= 1.0)
    # dias secos antecedentes (até o dia anterior): contagem de dias consecutivos sem chuva >= 1 mm
    dry = pd.DataFrame(index=daily.index, columns=daily.columns, dtype=float)
    # dias secos consecutivos até o dia anterior; dia sem dado torna a contagem desconhecida (NaN)
    # até o próximo dia chuvoso, que a zera
    cnt = np.zeros(daily.shape[1]); vals = wet.values; nanm = daily.isna().values
    out = np.full(daily.shape, np.nan)
    for i in range(daily.shape[0]):
        out[i] = cnt
        cnt = np.where(nanm[i], np.nan, np.where(vals[i], 0.0, cnt + 1))
    dry[:] = out
    cols = {c: i for i, c in enumerate(R.columns)}
    # estação mais próxima COM dados no ano
    est = est[est['cod'].isin(cols)]
    tree = BallTree(np.radians(est[['lat', 'lon']].values), metric='haversine')
    dist, idx = tree.query(np.radians(a[['latitude', 'longitude']].values), k=1)
    a['cod'] = est['cod'].values[idx[:, 0]]
    a['dist_km'] = dist[:, 0] * 6371.0
    a = a[a['dist_km'] <= DMAX].copy()
    a['utc'] = to_utc(a['datahora'], a['uf'])
    loc = a['datahora']
    rows = []
    # datas de controle: mesmo dia da semana e mesma hora no mesmo mês (hora local)
    first = loc.dt.to_period('M').dt.start_time
    for k in range(6):
        cand = first + pd.to_timedelta(((loc.dt.dayofweek - first.dt.dayofweek) % 7) + 7 * k, unit='D') \
               + pd.to_timedelta(loc.dt.hour, unit='h') + pd.to_timedelta(loc.dt.minute, unit='m')
        ok = cand.dt.month == loc.dt.month
        c = a.loc[ok, ['id', 'cod', 'uf', 'dist_km']].copy()
        c['local'] = cand[ok]
        c['caso'] = (cand[ok].dt.normalize() == loc[ok].dt.normalize()).astype(int)
        rows.append(c)
    cc = pd.concat(rows, ignore_index=True)
    cc['utc'] = to_utc(cc['local'], cc['uf'])
    hend = cc['utc'].dt.floor('h') + pd.Timedelta(hours=1)  # registro INMET que cobre a hora do evento
    ci = cc['cod'].map(cols).values
    ti = R.index.get_indexer(hend)
    okt = ti >= 0
    def take(M, shift=0):
        v = np.full(len(cc), np.nan)
        tt = ti - shift
        m = okt & (tt >= 0)
        v[m] = M.values[tt[m], ci[m]]
        return v
    cc['p0'] = take(R)
    cc['p1'] = take(R, 1)
    cc['p3'] = take(R3)
    cc['p24'] = take(R24, 1)
    for L in range(2, 7):
        cc[f'p_l{L}'] = take(R, L)
    di = daily.index.get_indexer(cc['utc'].dt.floor('D'))
    v = np.full(len(cc), np.nan); m = di >= 0
    v[m] = dry.values[di[m], ci[m]]
    cc['dias_secos'] = v
    # mantém estratos com caso válido e >= 1 controle válido
    cc = cc.dropna(subset=['p0'])
    g = cc.groupby('id')['caso'].agg(['sum', 'count'])
    keep = g.index[(g['sum'] == 1) & (g['count'] >= 2)]
    cc = cc[cc['id'].isin(keep)]
    cc['ano'] = ano
    cc.to_parquet(os.path.join(D, f'cc_{ano}.parquet'), index=False)
    print(ano, 'acidentes ok', len(keep), 'linhas', len(cc), 'dist mediana', round(a['dist_km'].median(), 1), flush=True)
