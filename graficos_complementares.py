#!/usr/bin/env python3
"""
=======================================================================
ME623 - Graficos Complementares para o Relatório PI
Nome: Arthur Agostinho Furlan Teixeira
RA: 164363
Gera:
  1. Heatmap de correlacao entre os niveis dos fatores
  2. Grafico de interacao entre fatores
  3. Tabela de empresas analisadas (CSV)

INSTRUCOES:
  pip install pandas numpy matplotlib seaborn
  python graficos_complementares.py
=======================================================================
"""

import numpy as np
import pandas as pd
import os
import sys

# ── Verificar dependencias ──
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
except ImportError:
    print("ERRO: pip install matplotlib")
    sys.exit(1)

try:
    import seaborn as sns
except ImportError:
    print("ERRO: pip install seaborn")
    sys.exit(1)

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# =====================================================================
# 1. TABELA DE EMPRESAS ANALISADAS
# =====================================================================

EMPRESAS = [
    ('ABEV3',  'Ambev S.A.',                        'Consumo n\u00e3o c\u00edclico'),
    ('BBAS3',  'Banco do Brasil S.A.',               'Financeiro'),
    ('BBDC4',  'Banco Bradesco S.A.',                'Financeiro'),
    ('BBSE3',  'BB Seguridade Participa\u00e7\u00f5es S.A.', 'Financeiro'),
    ('BPAC11', 'BTG Pactual S.A.',                   'Financeiro'),
    ('BRKM5',  'Braskem S.A.',                       'Materiais b\u00e1sicos'),
    ('CMIG4',  'CEMIG S.A.',                         'Utilidade p\u00fablica'),
    ('CSAN3',  'Cosan S.A.',                         'Petr\u00f3leo, g\u00e1s e biocombust\u00edveis'),
    ('CSNA3',  'Companhia Sider\u00fargica Nacional', 'Materiais b\u00e1sicos'),
    ('ENEV3',  'Eneva S.A.',                         'Utilidade p\u00fablica'),
    ('EQTL3',  'Equatorial Energia S.A.',            'Utilidade p\u00fablica'),
    ('GGBR4',  'Gerdau S.A.',                        'Materiais b\u00e1sicos'),
    ('ITUB4',  'Ita\u00fa Unibanco S.A.',            'Financeiro'),
    ('KLBN11', 'Klabin S.A.',                        'Materiais b\u00e1sicos'),
    ('LREN3',  'Lojas Renner S.A.',                  'Consumo c\u00edclico'),
    ('PETR4',  'Petrobras S.A.',                     'Petr\u00f3leo, g\u00e1s e biocombust\u00edveis'),
    ('PRIO3',  'PRIO S.A.',                          'Petr\u00f3leo, g\u00e1s e biocombust\u00edveis'),
    ('RADL3',  'Raia Drogasil S.A.',                 'Sa\u00fade'),
    ('RAIL3',  'Rumo S.A.',                          'Bens industriais'),
    ('RENT3',  'Localiza S.A.',                      'Consumo c\u00edclico'),
    ('SBSP3',  'Sabesp S.A.',                        'Utilidade p\u00fablica'),
    ('SUZB3',  'Suzano S.A.',                        'Materiais b\u00e1sicos'),
    ('TOTS3',  'TOTVS S.A.',                         'Tecnologia da informa\u00e7\u00e3o'),
    ('UGPA3',  'Ultrapar Participa\u00e7\u00f5es S.A.', 'Petr\u00f3leo, g\u00e1s e biocombust\u00edveis'),
    ('VALE3',  'Vale S.A.',                          'Materiais b\u00e1sicos'),
    ('VIVT3',  'Telef\u00f4nica Brasil S.A.',        'Comunica\u00e7\u00f5es'),
    ('WEGE3',  'WEG S.A.',                           'Bens industriais'),
]

df_empresas = pd.DataFrame(EMPRESAS, columns=['Ticker', 'Empresa', 'Setor'])
df_empresas.to_csv(os.path.join(OUTPUT_DIR, 'tabela_empresas.csv'), index=False, encoding='utf-8-sig')
print(f"  -> tabela_empresas.csv ({len(df_empresas)} empresas)")

# Contagem por setor
print("\n  Distribuicao por setor:")
for setor, count in df_empresas['Setor'].value_counts().items():
    print(f"    {setor:40s}: {count}")

# =====================================================================
# 2. DADOS DO EXPERIMENTO L9
# =====================================================================

# Resultados do experimento
dados_L9 = pd.DataFrame({
    'Exp':     [1,    2,    3,    4,    5,    6,    7,    8,    9],
    'A':       [5,    5,    5,    10,   10,   10,   15,   15,   15],
    'B':       [126,  252,  504,  126,  252,  504,  126,  252,  504],
    'C':       [0.25, 0.50, 0.75, 0.50, 0.75, 0.25, 0.75, 0.25, 0.50],
    'D_cod':   [1,    2,    3,    3,    1,    2,    2,    3,    1],  # 1=Igual, 2=MinVar, 3=MaxSharpe
    'D_label': ['Igual','Min.Var.','Max.Sharpe','Max.Sharpe','Igual','Min.Var.','Min.Var.','Max.Sharpe','Igual'],
    'Retorno': [0.1486, 0.2748, 0.2219, 0.2225, 0.1818, 0.2214, 0.1778, 0.2401, 0.1942],
    'Risco':   [0.1916, 0.1533, 0.2141, 0.2169, 0.1638, 0.1336, 0.1208, 0.2122, 0.1577],
    'Sharpe':  [0.1494, 0.9424, 0.3978, 0.4583, 0.3459, 0.6996, 0.4131, 0.5122, 0.4360],
})

# =====================================================================
# 3. HEATMAP DE CORRELAÇÃO ENTRE NÍVEIS E RESPOSTAS
# =====================================================================

print("\n  Gerando gráficos...")

# 3a. Matriz de correlação: fatores numéricos + respostas
df_corr_base = pd.DataFrame({
    'Num. Ativos (A)':     dados_L9['A'].astype(float),
    'Janela Hist. (B)':    dados_L9['B'].astype(float),
    'Lambda (C)':          dados_L9['C'].astype(float),
    'Metodo Pond. (D)':    dados_L9['D_cod'].astype(float),
    'Retorno':             dados_L9['Retorno'],
    'Risco':               dados_L9['Risco'],
    'Indice Sharpe':       dados_L9['Sharpe'],
})

corr_matrix = df_corr_base.corr()

fig, ax = plt.subplots(figsize=(9, 7))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

cmap = sns.diverging_palette(220, 20, as_cmap=True)
sns.heatmap(
    corr_matrix,
    mask=mask,
    annot=True,
    fmt='.3f',
    cmap=cmap,
    center=0,
    vmin=-1, vmax=1,
    square=True,
    linewidths=0.8,
    linecolor='white',
    cbar_kws={'label': 'Coeficiente de Correlação de Pearson', 'shrink': 0.8},
    annot_kws={'size': 11, 'fontweight': 'bold'},
    ax=ax,
)

ax.set_title('Matriz de Correlação entre Fatores e Respostas\n(Experimento Taguchi $L_9$)',
             fontsize=14, fontweight='bold', pad=15)
ax.tick_params(axis='both', labelsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'grafico_correlacao_fatores.png'), dpi=200, bbox_inches='tight')
plt.close()
print("  -> grafico_correlacao_fatores.png")

# =====================================================================
# 4. HEATMAP: SHARPE MÉDIO POR COMBINAÇÃO DE NÍVEIS (PARES DE FATORES)
# =====================================================================

# Níveis codificados
dados_L9['A_nivel'] = dados_L9['A'].map({5: 1, 10: 2, 15: 3})
dados_L9['B_nivel'] = dados_L9['B'].map({126: 1, 252: 2, 504: 3})
dados_L9['C_nivel'] = dados_L9['C'].map({0.25: 1, 0.50: 2, 0.75: 3})
dados_L9['D_nivel'] = dados_L9['D_cod']

pares = [
    ('A_nivel', 'D_nivel', 'Num. Ativos (A)', 'Metodo Pond. (D)'),
    ('B_nivel', 'D_nivel', 'Janela Hist. (B)', 'Metodo Pond. (D)'),
    ('C_nivel', 'D_nivel', 'Lambda (C)',       'Metodo Pond. (D)'),
    ('A_nivel', 'B_nivel', 'Num. Ativos (A)', 'Janela Hist. (B)'),
    ('B_nivel', 'C_nivel', 'Janela Hist. (B)', 'Lambda (C)'),
    ('A_nivel', 'C_nivel', 'Num. Ativos (A)', 'Lambda (C)'),
]

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Índice Sharpe por Combinação de Níveis dos Fatores\n(Interações observadas no $L_9$)',
             fontsize=14, fontweight='bold')

for idx, (f1, f2, lab1, lab2) in enumerate(pares):
    ax = axes[idx // 3, idx % 3]
    pivot = dados_L9.pivot_table(values='Sharpe', index=f1, columns=f2, aggfunc='mean')

    sns.heatmap(
        pivot,
        annot=True,
        fmt='.3f',
        cmap='YlOrRd',
        linewidths=0.5,
        linecolor='white',
        cbar_kws={'shrink': 0.7},
        annot_kws={'size': 12, 'fontweight': 'bold'},
        ax=ax,
    )
    ax.set_xlabel(lab2, fontsize=10, fontweight='bold')
    ax.set_ylabel(lab1, fontsize=10, fontweight='bold')
    ax.set_title(f'{lab1} vs {lab2}', fontsize=11)

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(os.path.join(OUTPUT_DIR, 'grafico_interacao_niveis.png'), dpi=200, bbox_inches='tight')
plt.close()
print("  -> grafico_interacao_niveis.png")

# =====================================================================
# 5. GRÁFICO DE INTERAÇÃO CLÁSSICO (LINHAS)
# =====================================================================

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Gráficos de Interação: Método de Ponderação (D) vs demais fatores',
             fontsize=13, fontweight='bold')

cores_metodo = {'Igual': '#2196F3', 'Min.Var.': '#4CAF50', 'Max.Sharpe': '#FF5722'}
marcadores = {'Igual': 'o', 'Min.Var.': 's', 'Max.Sharpe': '^'}

# D vs A
ax = axes[0]
for metodo in ['Igual', 'Min.Var.', 'Max.Sharpe']:
    subset = dados_L9[dados_L9['D_label'] == metodo]
    ax.plot(subset['A'], subset['Sharpe'],
            marker=marcadores[metodo], color=cores_metodo[metodo],
            linewidth=2, markersize=8, label=metodo)
ax.set_xlabel('Num. Ativos (A)', fontsize=10)
ax.set_ylabel('Índice Sharpe', fontsize=10)
ax.set_title('D vs A', fontsize=11, fontweight='bold')
ax.set_xticks([5, 10, 15])
ax.legend(title='Método (D)', fontsize=8, title_fontsize=9)
ax.grid(True, alpha=0.3)

# D vs B
ax = axes[1]
for metodo in ['Igual', 'Min.Var.', 'Max.Sharpe']:
    subset = dados_L9[dados_L9['D_label'] == metodo]
    ax.plot(subset['B'], subset['Sharpe'],
            marker=marcadores[metodo], color=cores_metodo[metodo],
            linewidth=2, markersize=8, label=metodo)
ax.set_xlabel('Janela Histórica - dias úteis (B)', fontsize=10)
ax.set_ylabel('Índice Sharpe', fontsize=10)
ax.set_title('D vs B', fontsize=11, fontweight='bold')
ax.set_xticks([126, 252, 504])
ax.set_xticklabels(['126\n(6m)', '252\n(12m)', '504\n(24m)'])
ax.legend(title='Método (D)', fontsize=8, title_fontsize=9)
ax.grid(True, alpha=0.3)

# D vs C
ax = axes[2]
for metodo in ['Igual', 'Min.Var.', 'Max.Sharpe']:
    subset = dados_L9[dados_L9['D_label'] == metodo]
    ax.plot(subset['C'], subset['Sharpe'],
            marker=marcadores[metodo], color=cores_metodo[metodo],
            linewidth=2, markersize=8, label=metodo)
ax.set_xlabel('Lambda - aversão ao risco (C)', fontsize=10)
ax.set_ylabel('Índice Sharpe', fontsize=10)
ax.set_title('D vs C', fontsize=11, fontweight='bold')
ax.set_xticks([0.25, 0.50, 0.75])
ax.legend(title='Método (D)', fontsize=8, title_fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'grafico_interacao_linhas.png'), dpi=200, bbox_inches='tight')
plt.close()
print("  -> grafico_interacao_linhas.png")

# =====================================================================
# 6. GRÁFICO CONTRIBUIÇÃO ANOVA (BARRAS)
# =====================================================================

fig, ax = plt.subplots(figsize=(8, 5))

fatores_nomes = ['Método Pond.\n(D)', 'Janela Hist.\n(B)', 'Lambda\n(C)', 'Num. Ativos\n(A)']
contribs = [52.98, 25.92, 20.09, 1.02]
cores_barra = ['#E53935', '#FB8C00', '#FDD835', '#90CAF9']

bars = ax.barh(fatores_nomes, contribs, color=cores_barra, edgecolor='white', height=0.6)

for bar, val in zip(bars, contribs):
    ax.text(bar.get_width() + 0.8, bar.get_y() + bar.get_height()/2,
            f'{val:.2f}%', va='center', fontsize=12, fontweight='bold')

ax.set_xlabel('Contribuição Percentual (%)', fontsize=11)
ax.set_title('ANOVA: Contribuição de cada Fator no Índice Sharpe',
             fontsize=13, fontweight='bold')
ax.set_xlim(0, 62)
ax.grid(axis='x', alpha=0.3)
ax.invert_yaxis()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'grafico_anova_barras.png'), dpi=200, bbox_inches='tight')
plt.close()
print("  -> grafico_anova_barras.png")

# =====================================================================
print(f"\n{'='*60}")
print(f"  CONCLUÍDO! Gráficos gerados em: {OUTPUT_DIR}")
print(f"{'='*60}\n")
