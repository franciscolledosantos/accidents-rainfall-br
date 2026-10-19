"""Análises adicionais pedidas pelo revisor independente (17/10/2026).

1. Perda de dados das estações (missingness informativa): fração de horas-estação faltantes por ano;
   probabilidade de falta na hora seguinte a uma hora chuvosa vs seca; referentes faltantes por exposição
   do caso; sensibilidades (sem 2021/2025; sem estação-anos com >30% de falta; sem estratos com hora
   vizinha a lacuna; só estratos completos).
2. Dose-resposta: termo contínuo log(1+p0), tendência ordinal, teste de Wald de platô (2-5 = 5-10 = >=10).
3. Estratificação distância (<=10 / >10 km) x uso do solo (rural / urbano).
4. OR anual dentro de cada classe da PRF (composição vs tendência) com meta-regressão.
5. IC dos acidentes atribuíveis por simulação paramétrica (cov robusta por estação).
6. n de casos expostos por categoria; correlação entre indicadores de lag 0 e 1.
7. Composição por classe na amostra pareada (Fig. 4) e descritivas.csv regenerado.
Saídas em resultados/rev17_*.csv e rev17_log.txt.
"""
import sys, os, glob, time
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(__file__))
from clogit import clogit

D = os.path.join(os.path.dirname(__file__), '..', 'dados')
OUT = os.path.join(os.path.dirname(__file__), '..', 'resultados')
RESUME = os.environ.get('RESUME') == '1'   # retoma após as etapas 2, 3 e 5 já gravadas
LOG = open(os.path.join(OUT, 'rev17_log.txt'), 'a' if RESUME else 'w', encoding='utf-8')


def log(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); LOG.write(s + '\n'); LOG.flush()


t0 = time.time()
cc = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(D, 'cc_20*.parquet')))], ignore_index=True)
acc = pd.read_parquet(os.path.join(D, 'prf_acidentes_2017_2025.parquet')).set_index('id')
for c in ['uso_solo', 'classificacao_acidente', 'fatal', 'mortos', 'chuva_prf']:
    cc[c] = cc['id'].map(acc[c])
cc['area'] = cc['uso_solo'].map({'Sim': 'urbana', 'Não': 'rural', 'Urbano': 'urbana', 'Rural': 'rural'})
cc['classe'] = cc['classificacao_acidente'].map({'Sem Vítimas': 'sem vitimas', 'Com Vítimas Feridas': 'com feridos', 'Com Vítimas Fatais': 'com mortos'})
CUTS = [-0.01, 0.19, 2, 5, 10, 1e9]; LAB = ['0', '0.2-2', '2-5', '5-10', '>=10']
cc['cat'] = pd.cut(cc['p0'], CUTS, labels=LAB)
cc['chuva'] = (cc['p0'] >= 0.2).astype(float)
dums = pd.get_dummies(cc['cat'], prefix='p0').astype(float)
cc = pd.concat([cc, dums], axis=1)
catcols = [f'p0_{l}' for l in LAB[1:]]
log('cc', cc.shape, 'casos', int(cc['caso'].sum()), f'{time.time()-t0:.0f}s')


def fit(df, cols, label, full=False):
    X = df[cols].astype(float).values
    r = clogit(df['caso'].values, X, df['id'].values, cluster=df['cod'].values, full=full)
    b, se, ll, sec = r[:4]
    rows = [dict(modelo=label, termo=c, OR=np.exp(b[j]), lo=np.exp(b[j] - 1.96 * sec[j]), hi=np.exp(b[j] + 1.96 * sec[j]),
                 b=b[j], se_rob=sec[j], p=2 * (1 - stats.norm.cdf(abs(b[j] / sec[j]))),
                 n_casos=int(df['caso'].sum()), n_linhas=len(df)) for j, c in enumerate(cols)]
    return (rows, r[4]) if full else rows


# ---------------------------------------------------------------- 6. n expostos e correlação entre lags
cs = cc[cc['caso'] == 1]
nexp = cs['cat'].value_counts().reindex(LAB)
cs_ant = int(((cs['p0'] < 0.2) & (cs['p1'] >= 0.2)).sum())
log('n casos por categoria:', nexp.to_dict(), 'casos secos após hora chuvosa:', cs_ant)
pd.DataFrame({'categoria': LAB, 'n_casos': nexp.values}).to_csv(os.path.join(OUT, 'rev17_n_expostos.csv'), index=False)
lagc = ['p0', 'p1'] + [f'p_l{L}' for L in range(2, 7)]
dl = cc.dropna(subset=lagc)
I = (dl[lagc] >= 0.2).astype(float)
corr01 = I['p0'].corr(I['p1']); corr12 = I['p1'].corr(I['p_l2'])
log(f'correlação (phi) lag0-lag1 = {corr01:.3f}; lag1-lag2 = {corr12:.3f}')
pd.DataFrame({'par': ['lag0-lag1', 'lag1-lag2'], 'phi': [corr01, corr12]}).to_csv(os.path.join(OUT, 'rev17_corr_lags.csv'), index=False)

# ---------------------------------------------------------------- 2. dose-resposta
R = []
if RESUME:
    log('RESUME: etapas 2, 3 e 5 já gravadas; pulando')
if not RESUME:
    cc['logp'] = np.log1p(cc['p0'])
    R += fit(cc, ['logp'], 'DR log(1+p0) contínuo')
    cc['score'] = cc['cat'].cat.codes.astype(float)          # 0,1,2,3,4
    R += fit(cc, ['score'], 'DR escore ordinal (por categoria)')
    rows, V = fit(cc, catcols, 'M1 intensidade', full=True)
    R += rows
    b = np.array([r['b'] for r in rows])
    # platô: H0 b(2-5) = b(5-10) = b(>=10)
    C = np.array([[0, 1, -1, 0], [0, 0, 1, -1]])
    W = (C @ b) @ np.linalg.solve(C @ V @ C.T, C @ b); p_plato = 1 - stats.chi2.cdf(W, 2)
    # degrau 0.2-2 vs 2-5
    C1 = np.array([[1, -1, 0, 0]]); W1 = (C1 @ b) @ np.linalg.solve(C1 @ V @ C1.T, C1 @ b); p_step = 1 - stats.chi2.cdf(W1, 1)
    # tendência linear entre as três classes superiores (escore 2,3,4 só entre expostos >= 2 mm)
    d2 = cc.copy(); d2['score_hi'] = d2['score'].where(d2['score'] >= 2, 0.0)
    d2['hi_any'] = (d2['score'] >= 2).astype(float); d2['s_hi'] = (d2['score'] - 2).clip(lower=0) * d2['hi_any']
    R += fit(d2, ['p0_0.2-2', 'hi_any', 's_hi'], 'DR tendência acima de 2 mm (escore 0,1,2 em 2-5,5-10,>=10)')
    log(f'Wald platô (2-5 = 5-10 = >=10): chi2 = {W:.2f}, p = {p_plato:.3f}; degrau 0.2-2 vs 2-5: chi2 = {W1:.2f}, p = {p_step:.2e}')
    dr = pd.DataFrame(R); dr.to_csv(os.path.join(OUT, 'rev17_dose_resposta.csv'), index=False)
    pd.DataFrame({'teste': ['plato_2-5_5-10_>=10', 'degrau_0.2-2_vs_2-5'], 'chi2': [W, W1], 'gl': [2, 1], 'p': [p_plato, p_step]}).to_csv(
        os.path.join(OUT, 'rev17_dose_resposta_testes.csv'), index=False)
    log(dr[['modelo', 'termo', 'OR', 'lo', 'hi', 'p']].round(4).to_string())

    # ---------------------------------------------------------------- 5. IC dos acidentes atribuíveis (simulação paramétrica)
    rng = np.random.default_rng(17)
    n_c = np.array([nexp[l] for l in LAB[1:]], float)
    B = rng.multivariate_normal(b, V, size=20000)
    ORs = np.exp(B)
    att = (n_c * (ORs - 1) / ORs).sum(1)
    att_pt = (n_c * (np.exp(b) - 1) / np.exp(b)).sum()
    lo, hi = np.percentile(att, [2.5, 97.5])
    log(f'acidentes atribuíveis = {att_pt:.0f} (IC95% {lo:.0f}-{hi:.0f}); FA = {100*att_pt/len(cs):.2f}% ({100*lo/len(cs):.2f}-{100*hi/len(cs):.2f})')
    pd.DataFrame({'atribuiveis': [att_pt], 'lo': [lo], 'hi': [hi], 'FA_pct': [100 * att_pt / len(cs)],
                  'FA_lo': [100 * lo / len(cs)], 'FA_hi': [100 * hi / len(cs)], 'casos': [len(cs)], 'sim': [20000]}).to_csv(
        os.path.join(OUT, 'rev17_atribuiveis_IC.csv'), index=False)

    # ---------------------------------------------------------------- 3. distância x uso do solo
    R = []
    cc['d10'] = np.where(cc['dist_km'] <= 10, '<=10 km', '>10 km')
    for a in ['rural', 'urbana']:
        for dd in ['<=10 km', '>10 km']:
            df = cc[(cc['area'] == a) & (cc['d10'] == dd)]
            R += fit(df, ['chuva'], f'{a} {dd}')
    dx = pd.DataFrame(R); dx.to_csv(os.path.join(OUT, 'rev17_distancia_x_uso_solo.csv'), index=False)
    log(dx[['modelo', 'OR', 'lo', 'hi', 'n_casos']].round(3).to_string())
    sh = cc[cc['caso'] == 1].groupby('d10')['area'].value_counts(normalize=True).round(3)
    log('proporção urbana por distância:', sh.to_dict())

# ---------------------------------------------------------------- 4. OR anual dentro de cada classe PRF
if not RESUME:
    R = []
    for cl in ['com feridos', 'sem vitimas', 'com mortos']:
        for y in sorted(cc['ano'].unique()):
            df = cc[(cc['classe'] == cl) & (cc['ano'] == y)]
            if df['caso'].sum() < 300:
                continue
            R += fit(df, ['chuva'], f'{cl} {y}')
    ya = pd.DataFrame(R); ya['classe'] = ya['modelo'].str[:-5]; ya['ano'] = ya['modelo'].str[-4:].astype(int)
    tr = []
    for cl, g in ya.groupby('classe'):
        for nome, gg in [('todos', g), ('sem 2017', g[g['ano'] >= 2018])]:
            w = 1 / gg['se_rob'] ** 2
            X = np.column_stack([np.ones(len(gg)), gg['ano'] - 2021])
            beta = np.linalg.solve(X.T @ (w.values[:, None] * X), X.T @ (w * gg['b']).values)
            cov = np.linalg.inv(X.T @ (w.values[:, None] * X))
            z = beta[1] / np.sqrt(cov[1, 1])
            tr.append(dict(classe=cl, anos=nome, tendencia_logOR_ano=beta[1], se=np.sqrt(cov[1, 1]), p=2 * (1 - stats.norm.cdf(abs(z)))))
    tr = pd.DataFrame(tr)
    ya.to_csv(os.path.join(OUT, 'rev17_OR_anual_por_classe.csv'), index=False)
    tr.to_csv(os.path.join(OUT, 'rev17_tendencia_por_classe.csv'), index=False)
    log(ya[['modelo', 'OR', 'lo', 'hi', 'n_casos']].round(3).to_string()); log(tr.round(5).to_string())

    # ---------------------------------------------------------------- 7. composição por classe na amostra pareada; descritivas
    comp = cs.groupby('ano')['classificacao_acidente'].value_counts(normalize=True).unstack().round(4)
    comp.to_csv(os.path.join(OUT, 'composicao_classificacao_ano_pareada.csv'))
    log('composição pareada:\n' + comp.to_string())
    ctrl = cc[cc['caso'] == 0]
    pd.Series({'linhas': len(cc), 'casos': len(cs), 'controles': len(ctrl), 'ctrl_por_caso': len(ctrl) / len(cs), 'estacoes': cc['cod'].nunique(),
               'dist_med': cs['dist_km'].median(), 'dist_q1': cs['dist_km'].quantile(.25), 'dist_q3': cs['dist_km'].quantile(.75),
               'prev_chuva_caso': cs['chuva'].mean(), 'prev_chuva_ctrl': ctrl['chuva'].mean(),
               'prev10_caso': (cs['p0'] >= 10).mean(), 'prev10_ctrl': (ctrl['p0'] >= 10).mean()}).to_csv(os.path.join(OUT, 'descritivas.csv'), header=False)

# ---------------------------------------------------------------- 1. missingness
# 1a. fração de horas-estação faltantes por ano (grade completa) e por estação-ano; 1b. falta após hora chuvosa; 1c. hora vizinha a lacuna
miss_year, miss_sy, cond = [], [], []
adj = pd.Series(False, index=cc.index)
cc['hend'] = cc['utc'].dt.floor('h') + pd.Timedelta(hours=1)
for ano in range(2017, 2026):
    r = pd.read_parquet(os.path.join(D, f'inmet_chuva_horaria_{ano}.parquet'))
    r['cod'] = r['cod'].astype(str)
    Rg = r.pivot_table(index='t_utc', columns='cod', values='p', aggfunc='first').astype('float32'); del r
    full = pd.date_range(f'{ano}-01-01', f'{ano}-12-31 23:00', freq='h')
    Rg = Rg.reindex(full)
    na = Rg.isna()
    miss_year.append(dict(ano=ano, estacoes=Rg.shape[1], frac_faltante=na.values.mean()))
    f_sy = na.mean(0); miss_sy.append(pd.DataFrame({'cod': f_sy.index, 'ano': ano, 'frac_faltante': f_sy.values}))
    # falta na hora h dada a chuva na hora h-1 (dentro da grade)
    prev = Rg.shift(1); nxt = na.shift(-1)
    for nome, m in [('seca (<0.2)', prev < 0.2), ('chuva 0.2-10', (prev >= 0.2) & (prev < 10)), ('chuva >=10', prev >= 10)]:
        cond.append(dict(ano=ano, hora_anterior=nome, n_horas=int(m.values.sum()), p_falta_hora_seguinte=float(na.values[m.values].mean())))
    # hora anterior faltante dado chuva na hora atual
    pna = na.shift(1)
    for nome, m in [('seca (<0.2)', Rg < 0.2), ('chuva 0.2-10', (Rg >= 0.2) & (Rg < 10)), ('chuva >=10', Rg >= 10)]:
        cond.append(dict(ano=ano, hora_anterior='[hora anterior faltante | hora atual] ' + nome, n_horas=int(m.values.sum()),
                         p_falta_hora_seguinte=float(pna.values[m.values].mean())))
    # linhas do cc deste ano: hora vizinha (h-1 ou h+1) faltante
    m = cc['ano'] == ano
    sub = cc.loc[m]
    ci = sub['cod'].map({c: i for i, c in enumerate(Rg.columns)})
    ti = Rg.index.get_indexer(sub['hend'])
    ok = (ti >= 0) & ci.notna().values
    cii = ci.fillna(0).astype(int).values
    before = np.ones(len(sub), bool); after = np.ones(len(sub), bool)
    tb = ti - 1; ta = ti + 1
    mb = ok & (tb >= 0); ma = ok & (ta < len(Rg))
    before[mb] = na.values[tb[mb], cii[mb]]; after[ma] = na.values[ta[ma], cii[ma]]
    before[~mb] = False; after[~ma] = False        # fora da grade anual (virada de ano): não marca
    adj.loc[sub.index] = before | after
    log(ano, 'miss', round(miss_year[-1]['frac_faltante'], 3), f'{time.time()-t0:.0f}s'); del Rg, na, prev, nxt, pna; import gc; gc.collect()
cc['adj_gap'] = adj.values
my = pd.DataFrame(miss_year); my.to_csv(os.path.join(OUT, 'rev17_missing_por_ano.csv'), index=False)
msy = pd.concat(miss_sy); msy.to_csv(os.path.join(OUT, 'rev17_missing_por_estacao_ano.csv'), index=False)
cd = pd.DataFrame(cond)
pooled = cd.groupby('hora_anterior').apply(lambda g: pd.Series({'n_horas': g['n_horas'].sum(),
                                                                'p_falta': (g['n_horas'] * g['p_falta_hora_seguinte']).sum() / g['n_horas'].sum()})).reset_index()
cd.to_csv(os.path.join(OUT, 'rev17_missing_condicional_ano.csv'), index=False)
pooled.to_csv(os.path.join(OUT, 'rev17_missing_condicional.csv'), index=False)
log(my.round(3).to_string()); log(pooled.round(4).to_string())

# 1d. referentes faltantes por exposição do caso
cs = cc[cc['caso'] == 1].copy()
loc = cs['local']; first = loc.dt.to_period('M').dt.start_time
fd = first + pd.to_timedelta(((loc.dt.dayofweek - first.dt.dayofweek) % 7), unit='D')
ncand = ((loc.dt.days_in_month - fd.dt.day) // 7 + 1) - 1      # referentes candidatos = mesmas semanas no mês - 1
nobs = cc[cc['caso'] == 0].groupby('id').size()
cs['ncand'] = ncand.values; cs['nobs'] = cs['id'].map(nobs).fillna(0).values
cs['nmiss'] = cs['ncand'] - cs['nobs']
ref = cs.groupby('cat', observed=True).agg(casos=('id', 'size'), ref_candidatos=('ncand', 'sum'), ref_validos=('nobs', 'sum'))
ref.index = ref.index.astype(str)
ref['frac_ref_faltante'] = 1 - ref['ref_validos'] / ref['ref_candidatos']
ref.loc['chuva (>=0.2)'] = [cs.loc[cs['chuva'] == 1, 'id'].size, cs.loc[cs['chuva'] == 1, 'ncand'].sum(), cs.loc[cs['chuva'] == 1, 'nobs'].sum(), np.nan]
ref.loc['chuva (>=0.2)', 'frac_ref_faltante'] = 1 - ref.loc['chuva (>=0.2)', 'ref_validos'] / ref.loc['chuva (>=0.2)', 'ref_candidatos']
ref.to_csv(os.path.join(OUT, 'rev17_referentes_faltantes_por_exposicao.csv'))
log(ref.round(4).to_string())
# estratos completos (nenhum referente faltante)
ids_completos = cs.loc[cs['nmiss'] <= 0, 'id']

# 1e. sensibilidades
R = []
R += fit(cc[~cc['ano'].isin([2021, 2025])], catcols, 'Sens sem 2021 e 2025 (anos com >30% de falta)')
msy['key'] = msy['cod'] + '_' + msy['ano'].astype(str)
bad = set(msy.loc[msy['frac_faltante'] > 0.30, 'key'])
cc['key'] = cc['cod'] + '_' + cc['ano'].astype(str)
R += fit(cc[~cc['key'].isin(bad)], catcols, 'Sens sem estação-anos com >30% de falta')
ids_adj = set(cc.loc[cc['adj_gap'], 'id'])
R += fit(cc[~cc['id'].isin(ids_adj)], catcols, 'Sens sem estratos com hora vizinha a lacuna')
R += fit(cc[cc['id'].isin(ids_completos)], catcols, 'Sens só estratos com todos os referentes válidos')
sens = pd.DataFrame(R); sens.to_csv(os.path.join(OUT, 'rev17_sens_missing.csv'), index=False)
log(sens[['modelo', 'termo', 'OR', 'lo', 'hi', 'n_casos']].round(3).to_string())
log('estação-anos excluídos (>30%):', len(bad), 'de', len(msy), '; estratos com hora vizinha a lacuna:', len(ids_adj),
    '; estratos completos:', len(ids_completos))
log(f'fim {time.time()-t0:.0f}s')
LOG.close()
