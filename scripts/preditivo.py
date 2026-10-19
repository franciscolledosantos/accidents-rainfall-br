"""Componente preditivo: contagem diária de acidentes na área de cada estação (<= 25 km) x chuva diária.

Validação: ano fora (leave-one-year-out). Modelos:
 B0 GLM Poisson: offset log(média da estação no treino) + dia da semana + mês + feriado
 B1 B0 + variáveis de chuva
 L0 LightGBM Poisson sem chuva; L1 LightGBM Poisson com chuva
Métrica: desvio de Poisson médio no ano de teste e ganho relativo (skill) sobre B0;
razão observado/previsto em dias de chuva forte (>= 20 mm) para checar calibração.
"""
import os, glob
import numpy as np, pandas as pd, holidays
from sklearn.neighbors import BallTree
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_poisson_deviance
import lightgbm as lgb

D = os.path.join(os.path.dirname(__file__), '..', 'dados')
OUT = os.path.join(os.path.dirname(__file__), '..', 'resultados')
acc = pd.read_parquet(os.path.join(D, 'prf_acidentes_2017_2025.parquet'))
br_h = holidays.Brazil(years=range(2017, 2026))
panels = []
for ano in range(2017, 2026):
    est = pd.read_csv(os.path.join(D, f'inmet_estacoes_{ano}.csv'), dtype={'cod': str}).dropna(subset=['lat', 'lon'])
    r = pd.read_parquet(os.path.join(D, f'inmet_chuva_horaria_{ano}.parquet'))
    r['cod'] = r['cod'].astype(str)
    r['dia'] = (r['t_utc'] - pd.Timedelta(hours=3)).dt.floor('D')   # dia local aproximado (UTC-3)
    r['molhado'] = (r['p'] >= 0.2).astype(float).where(r['p'].notna())
    g = r.groupby(['cod', 'dia']).agg(P=('p', 'sum'), Pmax=('p', 'max'), horas_chuva=('molhado', 'sum'), nval=('p', 'count')).reset_index()
    g = g[(g['nval'] >= 20) & (g['dia'].dt.year == ano)]
    est = est[est['cod'].isin(g['cod'].unique())]
    a = acc[acc['ano'] == ano]
    tree = BallTree(np.radians(est[['lat', 'lon']].values), metric='haversine')
    dist, idx = tree.query(np.radians(a[['latitude', 'longitude']].values), k=1)
    a = a.assign(cod=est['cod'].values[idx[:, 0]], dkm=dist[:, 0] * 6371)
    a = a[a['dkm'] <= 25]
    cnt = a.groupby(['cod', a['datahora'].dt.floor('D')]).size().rename('y').reset_index().rename(columns={'datahora': 'dia'})
    cntf = a[a['fatal'] == 1].groupby(['cod', a['datahora'].dt.floor('D')]).size().rename('y_fatal').reset_index().rename(columns={'datahora': 'dia'})
    # todas as estações do ano; a seleção das que têm rodovia federal no raio é feita depois (período inteiro)
    g = g.merge(cnt, on=['cod', 'dia'], how='left').merge(cntf, on=['cod', 'dia'], how='left').fillna({'y': 0, 'y_fatal': 0})
    g['ano'] = ano
    panels.append(g)
P = pd.concat(panels, ignore_index=True).sort_values(['cod', 'dia'])
tem_rod = P.groupby('cod')['y'].sum()
P = P[P['cod'].isin(tem_rod.index[tem_rod > 0])]          # estação com rodovia federal no raio (2017-2025)
# calendário completo por estação antes das defasagens (dias sem dado ficam NaN)
full = []
for c, g in P.groupby('cod'):
    idx = pd.date_range(g['dia'].min(), g['dia'].max(), freq='D')
    gg = g.set_index('dia').reindex(idx)
    gg['cod'] = c
    gg['P_lag1'] = gg['P'].shift(1)
    # dias secos consecutivos até o dia anterior; dia sem dado -> desconhecido até o próximo dia chuvoso
    cnt, out = 0.0, []
    for v in gg['P'].values:
        out.append(cnt)
        cnt = np.nan if np.isnan(v) else (0.0 if v >= 1 else cnt + 1)
    gg['dias_secos'] = out
    full.append(gg[gg['y'].notna()].rename_axis('dia').reset_index())
P = pd.concat(full, ignore_index=True)
P['dow'] = P['dia'].dt.dayofweek
P['mes'] = P['dia'].dt.month
P['feriado'] = P['dia'].dt.date.map(lambda d: d in br_h).astype(int)
P['logP'] = np.log1p(P['P']); P['logPmax'] = np.log1p(P['Pmax']); P['logPlag'] = np.log1p(P['P_lag1'])
P['P20'] = (P['P'] >= 20).astype(int)
P['primeira'] = ((P['P'] >= 1) & (P['dias_secos'] >= 7)).astype(float).where(P['dias_secos'].notna())
P.to_parquet(os.path.join(D, 'painel_estacao_dia.parquet'), index=False)
print('painel', P.shape, 'estacoes', P['cod'].nunique(), 'acidentes', int(P['y'].sum()))

cal = ['dow', 'mes', 'feriado']
rain = ['logP', 'logPmax', 'horas_chuva', 'logPlag', 'P20', 'primeira', 'dias_secos']
res = []
for alvo in ['y', 'y_fatal']:
    for ano in range(2017, 2026):
        tr, te = P[P['ano'] != ano], P[P['ano'] == ano]
        te = te[te['cod'].isin(tr['cod'].unique())]
        mu = tr.groupby('cod')[alvo].mean().clip(lower=1e-4)
        off_tr, off_te = np.log(tr['cod'].map(mu)), np.log(te['cod'].map(mu))
        def dmat(df, cols):
            X = pd.get_dummies(df[['dow', 'mes']].astype(str), drop_first=False).astype(float)
            X['feriado'] = df['feriado']
            for c in cols:
                X[c] = df[c].fillna(0)
                if df[c].isna().any() or c in ('logPlag', 'dias_secos', 'primeira'):
                    X[c + '_na'] = df[c].isna().astype(float)
            return X
        Xtr0, Xte0 = dmat(tr, []), dmat(te, [])
        Xte0 = Xte0.reindex(columns=Xtr0.columns, fill_value=0)
        Xtr1, Xte1 = dmat(tr, rain), dmat(te, rain)
        Xte1 = Xte1.reindex(columns=Xtr1.columns, fill_value=0)
        preds = {}
        # GLM com offset: ajusta y/exp(off) com peso exp(off) (equivalente ao offset em Poisson)
        for nm, (A, B) in {'B0': (Xtr0, Xte0), 'B1': (Xtr1, Xte1)}.items():
            w = np.exp(off_tr)
            m = PoissonRegressor(alpha=1e-6, max_iter=300).fit(A, tr[alvo] / w, sample_weight=w)
            preds[nm] = m.predict(B) * np.exp(off_te)
        for nm, cols in {'L0': cal, 'L1': cal + rain}.items():
            Xa = tr[cols].copy(); Xb = te[cols].copy()
            mdl = lgb.LGBMRegressor(objective='poisson', n_estimators=600, learning_rate=0.03, num_leaves=31,
                                    min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
            mdl.fit(Xa, tr[alvo], init_score=off_tr.values)
            preds[nm] = np.exp(mdl.predict(Xb, raw_score=True) + off_te.values)
        forte = te['P'] >= 20
        for nm, pr in preds.items():
            pr = np.clip(pr, 1e-6, None)
            res.append(dict(alvo=alvo, ano_teste=ano, modelo=nm, desvio=mean_poisson_deviance(te[alvo], pr),
                            obs_forte=te.loc[forte, alvo].sum(), prev_forte=pr[forte.values].sum(),
                            obs_seco=te.loc[te['P'] < 0.2, alvo].sum(), prev_seco=pr[(te['P'] < 0.2).values].sum(),
                            obs_prev_chuva_forte=te.loc[forte, alvo].sum() / pr[forte.values].sum(),
                            obs_prev_seco=te.loc[te['P'] < 0.2, alvo].sum() / pr[(te['P'] < 0.2).values].sum(),
                            n=len(te), n_forte=int(forte.sum())))
        print(alvo, ano, flush=True)
R = pd.DataFrame(res)
b0 = R[R.modelo == 'B0'].set_index(['alvo', 'ano_teste'])['desvio']
R['skill_vs_B0_%'] = 100 * (1 - R['desvio'] / R.set_index(['alvo', 'ano_teste']).index.map(b0).values)
R.to_csv(os.path.join(OUT, 'preditivo_loyo.csv'), index=False)
S = R.groupby(['alvo', 'modelo']).agg(desvio=('desvio', 'mean'), skill=('skill_vs_B0_%', 'mean'),
        skill_min=('skill_vs_B0_%', 'min'), obs_forte=('obs_forte', 'sum'), prev_forte=('prev_forte', 'sum'),
        obs_seco=('obs_seco', 'sum'), prev_seco=('prev_seco', 'sum'),
        razao_forte_min=('obs_prev_chuva_forte', 'min'), razao_forte_max=('obs_prev_chuva_forte', 'max'),
        n_forte=('n_forte', 'sum'))
S['razao_forte_pooled'] = S['obs_forte'] / S['prev_forte']; S['razao_seco_pooled'] = S['obs_seco'] / S['prev_seco']
S.to_csv(os.path.join(OUT, 'preditivo_resumo.csv'))
print(S.round(4))
