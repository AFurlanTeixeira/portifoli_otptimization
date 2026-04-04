#!/usr/bin/env python3
"""
=======================================================================
ME623 - Seleção e Validação da Base de Dados para o  Projeto I
Nome: Arthur Agostinho Furlan Teixeira
RA: 164363

Critérios formais de seleção de ativos:
  C1. Composição oficial da carteira teorica do Ibovespa (B3)
  C2. Exigência de dados completos no periodo inteiro (0% faltantes)

INSTRUÇÕESS:
  pip install yfinance pandas numpy
  python selecao_ativos.py

SAÍDAS:
  - selecao_ativos_log.txt         : log completo do processo de seleção
  - ativos_selecionados.csv        : lista final com ticker, empresa, setor
  - ativos_eliminados.csv          : ações eliminadas e motivo
  - resumo_selecao.csv             : resumo quantitativo por etapa
  - distribuicao_setorial.csv      : contagem final por setor
=======================================================================
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# =====================================================================
# CARTEIRA TEÓRICA DO IBOVESPA (B3)
# Fonte: B3 - Composição da carteira teórica do Ibovespa
# Base: Quadrimestre vigente em Jan/2025
# Inclui ticker Yahoo Finance (.SA), nome, setor e peso aproximado
# =====================================================================

CARTEIRA_IBOVESPA = [
    # Ticker YF       Nome                              Setor B3                              Peso%
    ('PETR4.SA',  'Petrobras S.A. (PN)',               'Petroleo, gas e biocombustiveis',     8.20),
    ('VALE3.SA',  'Vale S.A.',                         'Materiais basicos',                   7.50),
    ('ITUB4.SA',  'Itau Unibanco S.A. (PN)',           'Financeiro',                          6.80),
    ('BBDC4.SA',  'Banco Bradesco S.A. (PN)',          'Financeiro',                          3.90),
    ('PETR3.SA',  'Petrobras S.A. (ON)',               'Petroleo, gas e biocombustiveis',     3.80),
    ('BBAS3.SA',  'Banco do Brasil S.A.',              'Financeiro',                          3.60),
    ('B3SA3.SA',  'B3 S.A.',                           'Financeiro',                          3.20),
    ('WEGE3.SA',  'WEG S.A.',                          'Bens industriais',                    3.10),
    ('ABEV3.SA',  'Ambev S.A.',                        'Consumo nao ciclico',                 2.80),
    ('RENT3.SA',  'Localiza S.A.',                     'Consumo ciclico',                     2.50),
    ('PRIO3.SA',  'PRIO S.A.',                         'Petroleo, gas e biocombustiveis',     2.30),
    ('SUZB3.SA',  'Suzano S.A.',                       'Materiais basicos',                   2.20),
    ('EQTL3.SA',  'Equatorial Energia S.A.',           'Utilidade publica',                   2.10),
    ('JBSS3.SA',  'JBS S.A.',                          'Consumo nao ciclico',                 2.00),
    ('ELET3.SA',  'Eletrobras S.A. (ON)',              'Utilidade publica',                   1.90),
    ('RADL3.SA',  'Raia Drogasil S.A.',                'Saude',                               1.80),
    ('BPAC11.SA', 'BTG Pactual S.A.',                  'Financeiro',                          1.70),
    ('LREN3.SA',  'Lojas Renner S.A.',                 'Consumo ciclico',                     1.60),
    ('GGBR4.SA',  'Gerdau S.A. (PN)',                  'Materiais basicos',                   1.50),
    ('CSNA3.SA',  'Cia. Siderurgica Nacional',         'Materiais basicos',                   1.40),
    ('VIVT3.SA',  'Telefonica Brasil S.A.',            'Comunicacoes',                        1.30),
    ('CMIG4.SA',  'CEMIG S.A. (PN)',                   'Utilidade publica',                   1.20),
    ('SBSP3.SA',  'Sabesp S.A.',                       'Utilidade publica',                   1.10),
    ('CPLE6.SA',  'Copel S.A. (PNB)',                  'Utilidade publica',                   1.00),
    ('TOTS3.SA',  'TOTVS S.A.',                        'Tecnologia da informacao',            0.95),
    ('RAIL3.SA',  'Rumo S.A.',                         'Bens industriais',                    0.90),
    ('UGPA3.SA',  'Ultrapar Participacoes S.A.',       'Petroleo, gas e biocombustiveis',     0.85),
    ('KLBN11.SA', 'Klabin S.A.',                       'Materiais basicos',                   0.80),
    ('BBSE3.SA',  'BB Seguridade Participacoes S.A.',  'Financeiro',                          0.75),
    ('ENEV3.SA',  'Eneva S.A.',                        'Utilidade publica',                   0.70),
    ('CSAN3.SA',  'Cosan S.A.',                        'Petroleo, gas e biocombustiveis',     0.65),
    ('BRKM5.SA',  'Braskem S.A.',                      'Materiais basicos',                   0.60),
    ('CCRO3.SA',  'CCR S.A.',                          'Bens industriais',                    0.55),
    ('HAPV3.SA',  'Hapvida S.A.',                      'Saude',                               0.50),
    ('EMBR3.SA',  'Embraer S.A.',                      'Bens industriais',                    0.50),
    ('CMIN3.SA',  'CSN Mineracao S.A.',                'Materiais basicos',                   0.45),
    ('HYPE3.SA',  'Hypera Pharma S.A.',                'Saude',                               0.45),
    ('VBBR3.SA',  'Vibra Energia S.A.',                'Petroleo, gas e biocombustiveis',     0.40),
    ('MULT3.SA',  'Multiplan S.A.',                    'Financeiro',                          0.40),
    ('NTCO3.SA',  'Natura \u0026Co S.A.',              'Consumo nao ciclico',                 0.35),
    ('MRFG3.SA',  'Marfrig S.A.',                      'Consumo nao ciclico',                 0.35),
    ('AZUL4.SA',  'Azul S.A. (PN)',                    'Bens industriais',                    0.30),
    ('GOAU4.SA',  'Metalurgica Gerdau S.A.',           'Materiais basicos',                   0.30),
    ('MGLU3.SA',  'Magazine Luiza S.A.',               'Consumo ciclico',                     0.25),
    ('CRFB3.SA',  'Carrefour Brasil S.A.',             'Consumo nao ciclico',                 0.25),
    ('YDUQ3.SA',  'Yduqs S.A.',                        'Consumo ciclico',                     0.20),
    ('COGN3.SA',  'Cogna Educacao S.A.',               'Consumo ciclico',                     0.20),
    ('IRBR3.SA',  'IRB Brasil RE S.A.',                'Financeiro',                          0.20),
    ('BEEF3.SA',  'Minerva Foods S.A.',                'Consumo nao ciclico',                 0.18),
    ('SMTO3.SA',  'Sao Martinho S.A.',                 'Consumo nao ciclico',                 0.15),
]

# =====================================================================
# CONFIGURAÇÕES
# =====================================================================
DATA_INICIO = '2021-01-01'
DATA_FIM    = '2025-12-31'
TOLERANCIA_FALTANTES = 0.0   # 0% = dados 100% completos
MIN_ACOES_POR_SETOR  = 2     # diversificação setorial mínima

# =====================================================================
# FUNÇÕES
# =====================================================================

def log_msg(msg, log_lines):
    print(msg)
    log_lines.append(msg)

def main():
    log = []
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')

    log_msg("=" * 70, log)
    log_msg(f"  ME623 - SELEÇÃO FORMAL DE ATIVOS PARA O EXPERIMENTO TAGUCHI", log)
    log_msg(f"  Data de execução: {timestamp}", log)
    log_msg("=" * 70, log)

    # ── ETAPA 0: Ponto de partida ──
    df_carteira = pd.DataFrame(CARTEIRA_IBOVESPA,
                               columns=['Ticker', 'Empresa', 'Setor', 'Peso_pct'])

    n_inicial = len(df_carteira)
    n_setores_inicial = df_carteira['Setor'].nunique()

    log_msg(f"\n--- ETAPA 0: UNIVERSO INICIAL ---", log)
    log_msg(f"  Fonte: Carteira Teórica do Ibovespa (B3, Jan/2025)", log)
    log_msg(f"  Ações na carteira: {n_inicial}", log)
    log_msg(f"  Setores representados: {n_setores_inicial}", log)
    log_msg(f"  Peso total coberto: {df_carteira['Peso_pct'].sum():.2f}%", log)
    log_msg(f"  Período de dados: {DATA_INICIO} a {DATA_FIM}", log)

    # ── ETAPA 1: Download e verificação de dados completos ──
    log_msg(f"\n--- ETAPA 1: CRITÉRIO C2 - Dados 100% completos ---", log)
    log_msg(f"  Baixando cotações de {n_inicial} ações...", log)

    try:
        import yfinance as yf
    except ImportError:
        log_msg("  ERRO: pip install yfinance", log)
        sys.exit(1)

    tickers = df_carteira['Ticker'].tolist()
    dados = yf.download(tickers, start=DATA_INICIO, end=DATA_FIM,
                        auto_adjust=True, progress=True)

    if isinstance(dados.columns, pd.MultiIndex):
        precos = dados['Close']
    else:
        precos = dados

    total_dias = len(precos)
    log_msg(f"  Dias de negociação no período: {total_dias}", log)
    log_msg(f"  Período efetivo: {precos.index[0].date()} a {precos.index[-1].date()}", log)

    # Calcular % de dados faltantes por acao
    pct_faltantes = (precos.isnull().sum() / total_dias * 100).round(2)

    # Classificar: aprovadas vs reprovadas
    eliminados_c2 = []
    aprovados_c2 = []

    for _, row in df_carteira.iterrows():
        ticker = row['Ticker']
        if ticker in precos.columns:
            pct = pct_faltantes[ticker]
            if pct <= TOLERANCIA_FALTANTES * 100:
                aprovados_c2.append(row.to_dict())
            else:
                eliminados_c2.append({
                    **row.to_dict(),
                    'Motivo': f'Dados faltantes: {pct:.1f}%',
                    'Etapa': 'C2'
                })
        else:
            eliminados_c2.append({
                **row.to_dict(),
                'Motivo': 'Ticker não encontrado no Yahoo Finance',
                'Etapa': 'C2'
            })

    df_aprovados_c2 = pd.DataFrame(aprovados_c2)
    df_eliminados_c2 = pd.DataFrame(eliminados_c2)

    log_msg(f"\n  Resultado C2:", log)
    log_msg(f"    Aprovadas (dados completos): {len(df_aprovados_c2)}", log)
    log_msg(f"    Eliminadas (dados faltantes): {len(df_eliminados_c2)}", log)

    if len(df_eliminados_c2) > 0:
        log_msg(f"\n  Ações eliminadas por C2:", log)
        for _, row in df_eliminados_c2.iterrows():
            log_msg(f"    {row['Ticker']:12s} {row['Empresa']:35s} -> {row['Motivo']}", log)

    # ── ETAPA 2: Diversificacao setorial minima ──
    log_msg(f"\n--- ETAPA 2: CRITÉRIO C3 - Diversificação setorial ---", log)
    log_msg(f"  Requisito: mínimo {MIN_ACOES_POR_SETOR} ações por setor", log)

    contagem_setorial = df_aprovados_c2['Setor'].value_counts()
    setores_insuficientes = contagem_setorial[contagem_setorial < MIN_ACOES_POR_SETOR]

    eliminados_c3 = []
    if len(setores_insuficientes) > 0:
        log_msg(f"\n  Setores com representação insuficiente:", log)
        for setor, count in setores_insuficientes.items():
            log_msg(f"    {setor}: {count} ação(ões) < mínimo de {MIN_ACOES_POR_SETOR}", log)
            # Eliminar acoes desses setores
            mask = df_aprovados_c2['Setor'] == setor
            for _, row in df_aprovados_c2[mask].iterrows():
                eliminados_c3.append({
                    **row.to_dict(),
                    'Motivo': f'Setor "{setor}" com apenas {count} ação(ões)',
                    'Etapa': 'C3'
                })

        df_eliminados_c3 = pd.DataFrame(eliminados_c3)
        setores_remover = setores_insuficientes.index.tolist()
        df_aprovados_c3 = df_aprovados_c2[~df_aprovados_c2['Setor'].isin(setores_remover)].copy()
    else:
        df_eliminados_c3 = pd.DataFrame()
        df_aprovados_c3 = df_aprovados_c2.copy()
        log_msg(f"  Todos os setores atendem ao critério.", log)

    log_msg(f"\n  Resultado C3:", log)
    log_msg(f"    Aprovadas apos C3: {len(df_aprovados_c3)}", log)
    log_msg(f"    Eliminadas por C3: {len(df_eliminados_c3)}", log)

    # ── RESULTADO FINAL ──
    df_final = df_aprovados_c3.reset_index(drop=True)
    df_final.index += 1  # numerar a partir de 1

    # Todas as eliminadas
    all_eliminados = pd.concat([df_eliminados_c2, df_eliminados_c3], ignore_index=True)

    log_msg(f"\n{'='*70}", log)
    log_msg(f"  RESULTADO FINAL DA SELEÇÃO", log)
    log_msg(f"{'='*70}", log)
    log_msg(f"  Ações selecionadas: {len(df_final)}", log)
    log_msg(f"  Ações eliminadas:   {len(all_eliminados)}", log)
    log_msg(f"  Setores cobertos:   {df_final['Setor'].nunique()}", log)
    log_msg(f"  Peso acumulado no Ibovespa: {df_final['Peso_pct'].sum():.2f}%", log)

    log_msg(f"\n  Distribuição setorial final:", log)
    dist_setorial = df_final['Setor'].value_counts().sort_values(ascending=False)
    for setor, count in dist_setorial.items():
        peso_setor = df_final[df_final['Setor'] == setor]['Peso_pct'].sum()
        log_msg(f"    {setor:40s}: {count:2d} acoes ({peso_setor:.2f}% do Ibov)", log)

    log_msg(f"\n  Lista final de ações:", log)
    log_msg(f"  {'#':>3s}  {'Ticker':12s} {'Empresa':38s} {'Setor':35s} {'Peso%':>6s}", log)
    log_msg(f"  {'-'*97}", log)
    for idx, row in df_final.iterrows():
        log_msg(f"  {idx:3d}  {row['Ticker']:12s} {row['Empresa']:38s} "
                f"{row['Setor']:35s} {row['Peso_pct']:6.2f}", log)

    # ── SALVAR ARQUIVOS ──
    log_msg(f"\n--- SALVANDO ARQUIVOS ---", log)

    # Ativos selecionados
    df_export = df_final[['Ticker', 'Empresa', 'Setor', 'Peso_pct']].copy()
    df_export['Ticker_limpo'] = df_export['Ticker'].str.replace('.SA', '', regex=False)
    df_export.to_csv(os.path.join(OUTPUT_DIR, 'ativos_selecionados.csv'),
                     index=False, encoding='utf-8-sig')
    log_msg(f"  -> ativos_selecionados.csv", log)

    # Ativos eliminados
    if len(all_eliminados) > 0:
        all_eliminados.to_csv(os.path.join(OUTPUT_DIR, 'ativos_eliminados.csv'),
                              index=False, encoding='utf-8-sig')
        log_msg(f"  -> ativos_eliminados.csv", log)

    # Distribuicao setorial
    df_dist = pd.DataFrame({
        'Setor': dist_setorial.index,
        'Num_acoes': dist_setorial.values,
        'Peso_Ibov_pct': [df_final[df_final['Setor']==s]['Peso_pct'].sum()
                          for s in dist_setorial.index]
    })
    df_dist.to_csv(os.path.join(OUTPUT_DIR, 'distribuicao_setorial.csv'),
                   index=False, encoding='utf-8-sig')
    log_msg(f"  -> distribuicao_setorial.csv", log)

    # Resumo quantitativo
    resumo = pd.DataFrame({
        'Etapa': ['C1 - Carteira Ibovespa', 'C2 - Dados completos',
                  'C3 - Diversif. setorial', 'RESULTADO FINAL'],
        'Acoes_entrada': [n_inicial, n_inicial,
                          len(df_aprovados_c2), len(df_aprovados_c3)],
        'Eliminadas': [0, len(df_eliminados_c2),
                       len(df_eliminados_c3), len(all_eliminados)],
        'Acoes_restantes': [n_inicial, len(df_aprovados_c2),
                            len(df_aprovados_c3), len(df_final)],
        'Setores': [n_setores_inicial, df_aprovados_c2['Setor'].nunique(),
                    df_aprovados_c3['Setor'].nunique(), df_final['Setor'].nunique()],
    })
    resumo.to_csv(os.path.join(OUTPUT_DIR, 'resumo_selecao.csv'),
                  index=False, encoding='utf-8-sig')
    log_msg(f"  -> resumo_selecao.csv", log)

    # Log completo
    with open(os.path.join(OUTPUT_DIR, 'selecao_ativos_log.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(log))
    log_msg(f"  -> selecao_ativos_log.txt", log)

    log_msg(f"\n{'='*70}", log)
    log_msg(f"  SELEÇÃO CONCLUÍDA!", log)
    log_msg(f"  Próximo passo: executar experimento_taguchi_portfolio.py", log)
    log_msg(f"  usando os tickers de ativos_selecionados.csv", log)
    log_msg(f"{'='*70}\n", log)


if __name__ == '__main__':
    main()
