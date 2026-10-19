"""Complemento da etapa 1 (missingness) de revisao_17-10.py:
(a) recalcula a probabilidade condicional de falta (hora anterior / hora seguinte) com dtype explícito;
(b) substitui a sensibilidade 'sem 2021 e 2025' por 'sem 2021 e 2022' (os dois anos com >30% de horas-estação
    faltantes na grade das estações presentes no arquivo anual, Tabela S7)."""
import sys, os, glob
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(__file__))
from clogit import clogit
D = os.path.join(os.path.dirname(__file__), '..', 'dados'); OUT = os.path.join(os.path.dirname(__file__), '..', 'resultados')
cond = []
for ano in range(2017, 2026):
    r = pd.read_parquet(os.path.join(D, f'inmet_chuva_horaria_{ano}.parquet')); r['cod'] = r['cod'].astype(str)
    Rg = r.pivot_table(index='t_utc', columns='cod', values='p', aggfunc='first').astype('float32'); del r
    Rg = Rg.reindex(pd.date_range(f'{ano}-01-01', f'{ano}-12-31 23:00', freq='h'))
    V = Rg.values; na = np.isnan(V)
    nxt = np.vstack([na[1:], np.zeros((1, na.shape[1]), bool)]); prv = np.vstack([np.zeros((1, na.shape[1]), bool), na[:-1]])
    for nome, m in [('seca (<0.2)', V < 0.2), ('chuva 0.2-10', (V >= 0.2) & (V < 10)), ('chuva >=10', V >= 10)]:
        m = m & ~na
        cond.append(dict(ano=ano, condicao='hora valida ' + nome, n_horas=int(m.sum()), p_falta_seguinte=float(nxt[m].mean()), p_falta_anterior=float(prv[m].mean())))
    print(ano, flush=True)
cd = pd.DataFrame(cond); cd.to_csv(os.path.join(OUT, 'rev17_missing_condicional_ano.csv'), index=False)
g = cd.groupby('condicao')
pooled = pd.DataFrame({'n_horas': g['n_horas'].sum(),
                       'p_falta_seguinte': g.apply(lambda x: (x['n_horas'] * x['p_falta_seguinte']).sum() / x['n_horas'].sum()),
                       'p_falta_anterior': g.apply(lambda x: (x['n_horas'] * x['p_falta_anterior']).sum() / x['n_horas'].sum())}).reset_index()
pooled.to_csv(os.path.join(OUT, 'rev17_missing_condicional.csv'), index=False); print(pooled.round(4).to_string())
# (b)
cc = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(D, 'cc_20*.parquet')))], ignore_index=True)
LAB = ['0', '0.2-2', '2-5', '5-10', '>=10']; cc['cat'] = pd.cut(cc['p0'], [-0.01, 0.19, 2, 5, 10, 1e9], labels=LAB)
X = pd.get_dummies(cc['cat'], prefix='p0').astype(float)[[f'p0_{l}' for l in LAB[1:]]]
df = cc[~cc['ano'].isin([2021, 2022])]
b, se, ll, sec = clogit(df['caso'].values, X.loc[df.index].values, df['id'].values, cluster=df['cod'].values)[:4]
rows = [dict(modelo='Sens sem 2021 e 2022 (anos com >30% de falta)', termo=c, OR=np.exp(b[j]), lo=np.exp(b[j] - 1.96 * sec[j]), hi=np.exp(b[j] + 1.96 * sec[j]), b=b[j], se_rob=sec[j],
             p=2 * (1 - stats.norm.cdf(abs(b[j] / sec[j]))), n_casos=int(df['caso'].sum()), n_linhas=len(df)) for j, c in enumerate(X.columns)]
sm = pd.read_csv(os.path.join(OUT, 'rev17_sens_missing.csv'))
sm = pd.concat([pd.DataFrame(rows), sm[~sm['modelo'].str.startswith('Sens sem 2021 e 2025')]], ignore_index=True)
sm.to_csv(os.path.join(OUT, 'rev17_sens_missing.csv'), index=False); print(sm[['modelo', 'termo', 'OR', 'lo', 'hi', 'n_casos']].round(3).to_string())
