"""Análise case-crossover: chuva horária (INMET) x acidentes nas rodovias federais (PRF), 2017-2025."""
import sys, os, glob
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from clogit import clogit
from statsmodels.stats.multitest import multipletests

D = os.path.join(os.path.dirname(__file__), '..', 'dados')
OUT = os.path.join(os.path.dirname(__file__), '..', 'resultados')
os.makedirs(OUT, exist_ok=True)
cc = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(D, 'cc_20*.parquet')))], ignore_index=True)
acc = pd.read_parquet(os.path.join(D, 'prf_acidentes_2017_2025.parquet'))
acc = acc.set_index('id')
REG = {'AC': 'N', 'AM': 'N', 'AP': 'N', 'PA': 'N', 'RO': 'N', 'RR': 'N', 'TO': 'N',
       'AL': 'NE', 'BA': 'NE', 'CE': 'NE', 'MA': 'NE', 'PB': 'NE', 'PE': 'NE', 'PI': 'NE', 'RN': 'NE', 'SE': 'NE',
       'DF': 'CO', 'GO': 'CO', 'MT': 'CO', 'MS': 'CO', 'ES': 'SE', 'MG': 'SE', 'RJ': 'SE', 'SP': 'SE',
       'PR': 'S', 'RS': 'S', 'SC': 'S'}
for c in ['fase_dia', 'tipo_pista', 'tracado_via', 'tipo_acidente', 'grave', 'fatal', 'chuva_prf', 'uso_solo', 'mortos']:
    cc[c] = cc['id'].map(acc[c])
cc['regiao'] = cc['uf'].map(REG)
TIPO = {'Saída de leito carroçável': 'Saída de pista', 'Capotamento': 'Capotamento/tombamento', 'Tombamento': 'Capotamento/tombamento',
        'Colisão com objeto estático': 'Colisão com objeto', 'Colisão com objeto': 'Colisão com objeto', 'Colisão com objeto em movimento': 'Colisão com objeto',
        'Colisão frontal': 'Colisão frontal', 'Colisão lateral sentido oposto': 'Colisão frontal',
        'Colisão traseira': 'Colisão traseira', 'Engavetamento': 'Colisão traseira',
        'Colisão lateral': 'Lateral/transversal', 'Colisão lateral mesmo sentido': 'Lateral/transversal', 'Colisão transversal': 'Lateral/transversal',
        'Atropelamento de Pedestre': 'Atropelamento de pedestre', 'Atropelamento de Animal': 'Atropelamento de animal',
        'Queda de ocupante de veículo': 'Queda de ocupante', 'Incêndio': 'Incêndio'}
CUTS = [-0.01, 0.19, 2, 5, 10, 1e9]
LAB = ['0', '0.2-2', '2-5', '5-10', '>=10']
cc['cat'] = pd.cut(cc['p0'], CUTS, labels=LAB)
cc['chuva'] = (cc['p0'] >= 0.2).astype(float)
cc['chuva_ant'] = ((cc['p0'] < 0.2) & (cc['p1'] >= 0.2)).astype(float)  # parou de chover na hora anterior


def fit(df, cols, label, cl='cod'):
    X = df[cols].astype(float).values
    b, se, ll, sec = clogit(df['caso'].values, X, df['id'].values, cluster=df[cl].values)
    res = []
    for j, c in enumerate(cols):
        res.append(dict(modelo=label, termo=c, OR=np.exp(b[j]), lo=np.exp(b[j] - 1.96 * sec[j]), hi=np.exp(b[j] + 1.96 * sec[j]),
                        b=b[j], se_rob=sec[j], p=2 * (1 - __import__('scipy').stats.norm.cdf(abs(b[j] / sec[j]))),
                        n_casos=int(df['caso'].sum()), n_linhas=len(df)))
    return res


dums = pd.get_dummies(cc['cat'], prefix='p0').astype(float)
cc = pd.concat([cc, dums], axis=1)
catcols = [f'p0_{l}' for l in LAB[1:]]
R = []
# 1) principal: categorias de intensidade
R += fit(cc, catcols, 'M1 intensidade')
# 2) qualquer chuva + chuva só na hora anterior (pista molhada)
R += fit(cc, ['chuva', 'chuva_ant'], 'M2 chuva e pós-chuva')
# 3) primeira chuva após estiagem: interação chuva x (dias secos >= 7)
d = cc.dropna(subset=['dias_secos']).copy()
d['seco7'] = (d['dias_secos'] >= 7).astype(float)
d['chuva_x_seco7'] = d['chuva'] * d['seco7']
R += fit(d, ['chuva', 'seco7', 'chuva_x_seco7'], 'M3 estiagem')
# 4) subgrupos (modelo separado por subgrupo: OR de qualquer chuva)
sub = {
    'gravidade': cc['fatal'].map({1: 'fatal'}).fillna(cc['grave'].map({1: 'grave nao fatal', 0: 'sem vitima grave'})),
    'classificacao': cc['id'].map(acc['classificacao_acidente']).map({'Sem Vítimas': 'sem vitimas', 'Com Vítimas Feridas': 'com feridos', 'Com Vítimas Fatais': 'com mortos'}),
    'regiao': cc['regiao'],
    'periodo': cc['fase_dia'].replace({'Pleno dia': 'dia', 'Plena Noite': 'noite', 'Plena noite': 'noite',
                                       'Amanhecer': 'transicao', 'Anoitecer': 'transicao'}),
    'pista': cc['tipo_pista'],
    'tipo': cc['tipo_acidente'].map(TIPO).fillna('Outros'),
    'area': cc['uso_solo'].map({'Sim': 'urbana', 'Não': 'rural', 'Urbano': 'urbana', 'Rural': 'rural'}),
}
for k, s in sub.items():
    for g in s.dropna().unique():
        df = cc[s == g]
        if df['caso'].sum() < 300:
            continue
        R += fit(df, ['chuva'], f'S {k}: {g}')
res = pd.DataFrame(R)
m = res['modelo'].str.startswith('S ')
res.loc[m, 'q_BH'] = multipletests(res.loc[m, 'p'], method='fdr_bh')[1]
# 5) sensibilidades
for dm in [10]:
    df = cc[cc['dist_km'] <= dm]
    R2 = fit(df, catcols, f'Sens dist<={dm}km')
    res = pd.concat([res, pd.DataFrame(R2)])
df = cc[~cc['ano'].isin([2020, 2021])]
res = pd.concat([res, pd.DataFrame(fit(df, catcols, 'Sens sem 2020-21'))])
for y in sorted(cc['ano'].unique()):
    res = pd.concat([res, pd.DataFrame(fit(cc[cc['ano'] == y], ['chuva'], f'Ano {y}'))])
res.to_csv(os.path.join(OUT, 'cc_resultados.csv'), index=False)
pd.set_option('display.width', 200)
print(res[['modelo', 'termo', 'OR', 'lo', 'hi', 'p', 'n_casos']].round(3).to_string())

# fração atribuível (casos expostos): AF_e = sum_exp (OR_cat-1)/OR_cat
cs = cc[cc['caso'] == 1]
orc = dict(zip(res[(res.modelo == 'M1 intensidade')]['termo'], res[(res.modelo == 'M1 intensidade')]['OR']))
af = sum(((cs[c] == 1).sum() * (orc[c] - 1) / orc[c]) for c in catcols)
print('casos', len(cs), 'expostos', int(cs['chuva'].sum()), 'atribuiveis', round(af), 'FA pop %', round(100 * af / len(cs), 2))
# validação do pareamento: concordância chuva INMET x registro PRF
print(pd.crosstab(cs['chuva'], cs['chuva_prf'], normalize='index').round(3))
pd.DataFrame({'casos': [len(cs)], 'expostos': [int(cs['chuva'].sum())], 'atribuiveis': [af],
              'FA_pct': [100 * af / len(cs)]}).to_csv(os.path.join(OUT, 'fracao_atribuivel.csv'), index=False)
pd.crosstab(cs['chuva'], cs['chuva_prf']).to_csv(os.path.join(OUT, 'concordancia_inmet_prf.csv'))
