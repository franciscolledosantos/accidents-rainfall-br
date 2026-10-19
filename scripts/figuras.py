import os, glob
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

B = os.path.join(os.path.dirname(__file__), '..')
D, R, F = [os.path.join(B, x) for x in ('dados', 'resultados', 'figuras')]
os.makedirs(F, exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9})
res = pd.read_csv(os.path.join(R, 'cc_resultados.csv'))


def save(fig, name):
    fig.savefig(os.path.join(F, name + '.png'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(F, name + '.tif'), dpi=300, bbox_inches='tight')
    plt.close(fig)


# Figura 1: crashes e estações
acc = pd.read_parquet(os.path.join(D, 'prf_acidentes_2017_2025.parquet'), columns=['latitude', 'longitude'])
cc = pd.concat([pd.read_parquet(f, columns=['id', 'cod', 'caso']) for f in glob.glob(os.path.join(D, 'cc_20*.parquet'))])
used = cc['cod'].astype(str).unique()
est = pd.concat([pd.read_csv(f, dtype={'cod': str}) for f in glob.glob(os.path.join(D, 'inmet_estacoes_*.csv'))]).drop_duplicates('cod')
fig, ax = plt.subplots(figsize=(6.3, 6.3))
ax.hexbin(acc['longitude'], acc['latitude'], gridsize=260, bins='log', cmap='Greys', mincnt=1, linewidths=0)
e1 = est[est['cod'].isin(used)]
ax.scatter(e1['lon'], e1['lat'], s=7, c='#1f78b4', edgecolor='white', lw=0.3, label=f'INMET automatic stations used (n = {len(e1)})')
ax.set_xlim(-74, -34); ax.set_ylim(-34, 6); ax.set_aspect('equal')
ax.set_xlabel('Longitude (°)'); ax.set_ylabel('Latitude (°)')
ax.legend(loc='lower left', fontsize=8, frameon=True)
ax.text(0.98, 0.98, 'Grey: density of 632,659 crashes on\nfederal highways (PRF), 2017–2025', transform=ax.transAxes,
        ha='right', va='top', fontsize=8)
save(fig, 'Fig1_map')

# Figura 2: dose-resposta e defasagens
m1 = res[res['modelo'] == 'M1 intensidade']
lag = pd.read_csv(os.path.join(R, 'defasagens.csv'))
fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.8))
lab = ['0.2–2', '2–5', '5–10', '≥10']
x = np.arange(4)
axs[0].errorbar(x, m1['OR'], yerr=[m1['OR'] - m1['lo'], m1['hi'] - m1['OR']], fmt='o', color='#1f78b4', capsize=3)
axs[0].axhline(1, color='k', lw=0.6, ls='--'); axs[0].set_xticks(x); axs[0].set_xticklabels(lab)
axs[0].set_xlabel('Rainfall in the crash hour (mm h$^{-1}$)'); axs[0].set_ylabel('Odds ratio (95% CI)')
axs[0].set_title('(a) Intensity', loc='left', fontsize=9)
axs[1].errorbar(lag['lag_h'], lag['OR'], yerr=[lag['OR'] - lag['lo'], lag['hi'] - lag['OR']], fmt='o-', color='#e6550d', capsize=3)
axs[1].axhline(1, color='k', lw=0.6, ls='--'); axs[1].set_xlabel('Hours before the crash hour (lag)'); axs[1].set_ylabel('Odds ratio (95% CI)')
axs[1].set_title('(b) Lag structure (wet hour ≥ 0.2 mm)', loc='left', fontsize=9)
for a in axs: a.spines[['top', 'right']].set_visible(False)
fig.tight_layout(); save(fig, 'Fig2_dose_lag')

# Figura 3: subgrupos (forest)
s = res[res['modelo'].str.startswith('S ') & (res['modelo'] != 'S classificacao: com mortos')].copy()  # classe PRF 'com mortos' = gravidade 'fatal' (mesmo modelo)
TR = {'classificacao': 'PRF class', 'gravidade': 'Severity', 'regiao': 'Region', 'periodo': 'Light', 'pista': 'Carriageway', 'tipo': 'Crash type', 'area': 'Land use'}
VAL = {'sem vitimas': 'No victims', 'com feridos': 'Injury', 'sem vitima grave': 'No serious injury', 'grave nao fatal': 'Serious, non-fatal', 'fatal': 'Fatal', 'noite': 'Night', 'dia': 'Day',
       'transicao': 'Dawn/dusk', 'Simples': 'Single', 'Dupla': 'Dual', 'Múltipla': 'Multilane', 'rural': 'Rural', 'urbana': 'Urban',
       'Colisão traseira': 'Rear-end (incl. pile-up)', 'Saída de pista': 'Run-off-road', 'Capotamento/tombamento': 'Rollover/overturn', 'Colisão com objeto': 'Collision with object', 'Lateral/transversal': 'Sideswipe/angle', 'Atropelamento de pedestre': 'Pedestrian struck', 'Atropelamento de animal': 'Animal struck', 'Queda de ocupante': 'Occupant fall (mostly motorcycles)', 'Incêndio': 'Vehicle fire', 'Outros': 'Other (n = 1,929)', 'Colisão lateral': 'Sideswipe (opposite)', 'Colisão lateral mesmo sentido': 'Sideswipe (same direction)',
       'Colisão transversal': 'Angle (transverse)', 'Atropelamento de Pedestre': 'Pedestrian struck', 'Tombamento': 'Overturn',
       'Saída de leito carroçável': 'Run-off-road', 'Queda de ocupante de veículo': 'Occupant fall (motorcycle)', 'outros': 'Other types', 'Colisão frontal': 'Head-on (incl. opposite sideswipe)'}
s['grp'] = s['modelo'].str.extract(r'S (\w+):')[0].map(TR)
s['val'] = s['modelo'].str.split(': ').str[1].map(lambda v: VAL.get(v, v))
order = ['PRF class', 'Severity', 'Light', 'Land use', 'Carriageway', 'Region', 'Crash type']
s['o'] = s['grp'].map({g: i for i, g in enumerate(order)})
s = s.sort_values(['o', 'OR'])
fig, ax = plt.subplots(figsize=(6.3, 7.2))
y = np.arange(len(s))[::-1]
col = np.where(s['q_BH'] < 0.05, '#1f78b4', '#999999')
for yi, (o, l, h, c) in zip(y, zip(s['OR'], s['lo'], s['hi'], col)):
    ax.plot([l, h], [yi, yi], color=c, lw=1)
ax.scatter(s['OR'], y, c=col, s=14, zorder=3)
ax.set_yticks(y); ax.set_yticklabels([f"{g}: {v}" for g, v in zip(s['grp'], s['val'])], fontsize=7.5)
ax.axvline(1, color='k', lw=0.6, ls='--'); ax.set_xscale('log'); ax.set_xticks([0.6, 0.8, 1, 1.5, 2, 2.5]); ax.set_xticklabels(['0.6', '0.8', '1', '1.5', '2', '2.5'])
from matplotlib.ticker import NullFormatter, NullLocator
ax.xaxis.set_minor_formatter(NullFormatter()); ax.xaxis.set_minor_locator(NullLocator())
ax.set_xlabel('Odds ratio for rain in the crash hour (≥ 0.2 mm), 95% CI')
ax.spines[['top', 'right']].set_visible(False)
fig.tight_layout(); save(fig, 'Fig3_subgroups')

# Figura 4: OR anual e proporção sem vítimas
ya = res[res['modelo'].str.startswith('Ano ')].copy(); ya['ano'] = ya['modelo'].str[-4:].astype(int)
comp = pd.read_csv(os.path.join(R, 'composicao_classificacao_ano_pareada.csv'), index_col=0)  # amostra pareada (mesma população dos OR anuais)
fig, ax = plt.subplots(figsize=(6.3, 2.8))
ax.errorbar(ya['ano'], ya['OR'], yerr=[ya['OR'] - ya['lo'], ya['hi'] - ya['OR']], fmt='o-', color='#1f78b4', capsize=3, label='OR (rain in crash hour)')
ax.set_ylabel('Odds ratio (95% CI)'); ax.set_xlabel('Year'); ax.set_xticks(ya['ano'].values)
ax2 = ax.twinx(); ax2.bar(comp.index, 100 * comp['Sem Vítimas'], color='#fdae6b', alpha=0.5, width=0.6, label='Share of linked crashes without victims')
ax2.set_ylabel('Crashes without victims (%)'); ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=7.5, loc='upper right', frameon=False)
fig.tight_layout(); save(fig, 'Fig4_annual')
print('ok')
