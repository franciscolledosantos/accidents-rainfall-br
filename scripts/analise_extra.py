"""Análises complementares: defasagens horárias, mortes atribuíveis, tendência anual, heterogeneidade."""
import sys, os, glob
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(__file__))
from clogit import clogit

D = os.path.join(os.path.dirname(__file__), '..', 'dados')
OUT = os.path.join(os.path.dirname(__file__), '..', 'resultados')
cc = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(D, 'cc_20*.parquet')))], ignore_index=True)
acc = pd.read_parquet(os.path.join(D, 'prf_acidentes_2017_2025.parquet')).set_index('id')
for c in ['fatal', 'mortos', 'classificacao_acidente', 'grave']:
    cc[c] = cc['id'].map(acc[c])
out = {}
# 1) defasagens 0..6 h (indicadores de chuva >= 0.2 mm), conjunto
lagc = ['p0', 'p1'] + [f'p_l{L}' for L in range(2, 7)]
d = cc.dropna(subset=lagc).copy()
g = d.groupby('id')['caso'].agg(['sum', 'count']); d = d[d['id'].isin(g.index[(g['sum'] == 1) & (g['count'] >= 2)])]
X = np.column_stack([(d[c] >= 0.2).astype(float) for c in lagc])
b, se, ll, sec = clogit(d['caso'].values, X, d['id'].values, cluster=d['cod'].values)
lag = pd.DataFrame({'lag_h': range(7), 'OR': np.exp(b), 'lo': np.exp(b - 1.96 * sec), 'hi': np.exp(b + 1.96 * sec),
                    'n_casos': int(d['caso'].sum())})
lag.to_csv(os.path.join(OUT, 'defasagens.csv'), index=False)
print(lag.round(3))
# 2) mortes atribuíveis (estratos de acidentes fatais)
f = cc[cc['fatal'] == 1].copy(); f['chuva'] = (f['p0'] >= 0.2).astype(float)
b, se, ll, sec = clogit(f['caso'].values, f[['chuva']].values, f['id'].values, cluster=f['cod'].values)
ORf = np.exp(b[0]); lo, hi = np.exp(b[0] - 1.96 * sec[0]), np.exp(b[0] + 1.96 * sec[0])
cf = f[f['caso'] == 1]
mortos_exp = cf.loc[cf['chuva'] == 1, 'mortos'].sum()
att = mortos_exp * (ORf - 1) / ORf
att_lo, att_hi = mortos_exp * (lo - 1) / lo, mortos_exp * (hi - 1) / hi
anos = cc['ano'].nunique()
print('fatais pareados', len(cf), 'mortes', cf['mortos'].sum(), 'mortes em hora chuvosa', mortes_exp if False else mortos_exp,
      'OR', round(ORf, 3), 'mortes atribuiveis', round(att), (round(att_lo), round(att_hi)), 'por ano', round(att / anos))
# extrapolação simples para todas as rodovias federais (supõe a mesma exposição e OR fora da área de 25 km)
tot_mortos = acc.loc[acc['ano'].between(2017, 2025), 'mortos'].sum()
frac_pareada = cf['mortos'].sum() / tot_mortos
out.update(dict(OR_fatal=ORf, OR_fatal_lo=lo, OR_fatal_hi=hi, mortos_pareados=cf['mortos'].sum(), mortos_hora_chuvosa=mortos_exp,
                mortes_atrib=att, mortes_atrib_lo=att_lo, mortes_atrib_hi=att_hi, mortes_atrib_ano=att / anos,
                mortos_total_PRF=tot_mortos, fracao_pareada=frac_pareada, mortes_atrib_nacional_ano=att / anos / frac_pareada))
# 3) tendência anual do OR (meta-regressão ponderada) e heterogeneidade regional
r = pd.read_csv(os.path.join(OUT, 'cc_resultados.csv'))
ya = r[r['modelo'].str.startswith('Ano ')].copy(); ya['ano'] = ya['modelo'].str[-4:].astype(int)
w = 1 / ya['se_rob'] ** 2
X = np.column_stack([np.ones(len(ya)), ya['ano'] - 2021])
beta = np.linalg.solve(X.T @ (w.values[:, None] * X), X.T @ (w * ya['b']).values)
cov = np.linalg.inv(X.T @ (w.values[:, None] * X))
z = beta[1] / np.sqrt(cov[1, 1])
out.update(dict(tendencia_logOR_ano=beta[1], tendencia_p=2 * (1 - stats.norm.cdf(abs(z)))))
ya2 = ya[ya['ano'] >= 2018]; w2 = 1 / ya2['se_rob'] ** 2
X2 = np.column_stack([np.ones(len(ya2)), ya2['ano'] - 2021])
beta2 = np.linalg.solve(X2.T @ (w2.values[:, None] * X2), X2.T @ (w2 * ya2['b']).values)
cov2 = np.linalg.inv(X2.T @ (w2.values[:, None] * X2))
out.update(dict(tendencia_sem2017=beta2[1], tendencia_sem2017_p=2 * (1 - stats.norm.cdf(abs(beta2[1] / np.sqrt(cov2[1, 1]))))))
# efeitos aleatórios (DerSimonian-Laird) na meta-regressão da tendência
v = ya['se_rob'].values ** 2; bb = ya['b'].values; Xr = np.column_stack([np.ones(len(ya)), ya['ano'].values - 2021])
wf = 1 / v; Wf = np.diag(wf)
bf = np.linalg.solve(Xr.T @ Wf @ Xr, Xr.T @ Wf @ bb)
Qr = np.sum(wf * (bb - Xr @ bf) ** 2); P_ = Wf - Wf @ Xr @ np.linalg.inv(Xr.T @ Wf @ Xr) @ Xr.T @ Wf
tau2 = max(0.0, (Qr - (len(bb) - 2)) / np.trace(P_))
wr = 1 / (v + tau2); Wr = np.diag(wr)
br = np.linalg.solve(Xr.T @ Wr @ Xr, Xr.T @ Wr @ bb); cr = np.linalg.inv(Xr.T @ Wr @ Xr)
out.update(dict(tendencia_RE=br[1], tendencia_RE_p=2 * (1 - stats.norm.cdf(abs(br[1] / np.sqrt(cr[1, 1])))), tau2=tau2))
rg = r[r['modelo'].str.startswith('S regiao')]
wq = 1 / rg['se_rob'] ** 2; bq = (wq * rg['b']).sum() / wq.sum(); Q = (wq * (rg['b'] - bq) ** 2).sum()
out.update(dict(Q_regiao=Q, Q_p=1 - stats.chi2.cdf(Q, len(rg) - 1)))
# 4) composição por ano (explica 2017): proporção de acidentes sem vítimas
comp = acc.groupby('ano')['classificacao_acidente'].value_counts(normalize=True).unstack().round(3)
comp.to_csv(os.path.join(OUT, 'composicao_classificacao_ano.csv'))
print(comp)
pd.Series(out).to_csv(os.path.join(OUT, 'extras.csv'))
print(pd.Series(out).round(4))
