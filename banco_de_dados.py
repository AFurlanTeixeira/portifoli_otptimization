#!/usr/bin/env python3
"""
=======================================================================
ME623 - Planejamento e Pesquisa | Projeto I
Otimizacao de Portfolio utilizando o Metodo Taguchi
Nome: Arthur Agostinho Furlan Teixeira
RA: 164363

INSTRUÇÕES:
  1. Instale as dependências:
     pip install yfinance pandas numpy scipy openpyxl matplotlib

  2. Execute o script:
     python experimento_taguchi_portfolio.py

  3. O script irá:
     - Baixar cotações reais de ações do Ibovespa via Yahoo Finance
     - Executar o experimento L9(3^4) completo
     - Salvar os resultados em CSV e gerar gráficos
     - Exibir todas as tabelas de análise no terminal

SAÍDAS GERADAS:
  - dados_cotacoes.csv        : cotações de fechamento ajustadas
  - dados_retornos.csv        : retornos logarítmicos diários
  - resultados_L9.csv         : resultados do experimento Taguchi
  - tabela_resposta_medias.csv: tabela de resposta para médias (ANOM)
  - tabela_resposta_SN.csv    : tabela de resposta para razão S/N
  - tabela_anova.csv          : tabela ANOVA
  - grafico_efeitos_principais.png : gráfico de efeitos principais
  - grafico_SN.png            : gráfico de efeitos S/N
  - resumo_completo.txt       : resumo textual de todos os resultados
=======================================================================
"""

import numpy as np
import pandas as pd
from scipy import stats
from itertools import combinations
import warnings
import os
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

warnings.filterwarnings('ignore')

# =====================================================================
# CONFIGURAÇÕS DO EXPERIMENTO
# =====================================================================

# Ações do Ibovespa (tickers Yahoo Finance - sufixo .SA)
TICKERS_IBOVESPA = [
    'PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'BBDC4.SA', 'BBAS3.SA',
    'WEGE3.SA', 'ABEV3.SA', 'RENT3.SA', 'SUZB3.SA', 'JBSS3.SA',
    'GGBR4.SA', 'CSNA3.SA', 'BPAC11.SA', 'EQTL3.SA', 'RADL3.SA',
    'LREN3.SA', 'VIVT3.SA', 'CMIG4.SA', 'CPLE6.SA', 'BBSE3.SA',
    'KLBN11.SA', 'TOTS3.SA', 'UGPA3.SA', 'ENEV3.SA', 'PRIO3.SA',
    'RAIL3.SA', 'SBSP3.SA', 'ELET3.SA', 'BRKM5.SA', 'CSAN3.SA',
]

# Periodo de dados
DATA_INICIO = '2021-01-01'
DATA_FIM = '2025-12-31'

# Taxa Selic anualizada (refêrencia: média do período)
SELIC_ANUAL = 0.1275   # 12.75% a.a. (média aproximada 2022-2025)
SELIC_DIARIA = (1 + SELIC_ANUAL) ** (1/252) - 1

# Número de replicações por configuração
N_REPLICACOES = 30

# Semente para reprodutibilidade
SEED = 42

# =====================================================================
# FATORES E NÍVEIS DO EXPERIMENTO TAGUCHI L9(3^4)
# =====================================================================

FATORES = {
    'A': {'nome': 'Num. Ativos',      'niveis': [5, 10, 15]},
    'B': {'nome': 'Janela Historica',  'niveis': [126, 252, 504]},  # dias úteis: ~6m, ~12m, ~24m
    'C': {'nome': 'Lambda (aversao)',  'niveis': [0.25, 0.50, 0.75]},
    'D': {'nome': 'Metodo Ponderacao', 'niveis': ['igual', 'min_var', 'max_sharpe']},
}

# Correlação Entre Número de Ativos, Janela Histórica, Lambda e Método de Ponderação: 


# Arranjo Ortogonal L9(3^4) - índices 0, 1, 2 para níveis 1, 2, 3
ARRANJO_L9 = np.array([
    [0, 0, 0, 0],  # Exp 1
    [0, 1, 1, 1],  # Exp 2
    [0, 2, 2, 2],  # Exp 3
    [1, 0, 1, 2],  # Exp 4
    [1, 1, 2, 0],  # Exp 5
    [1, 2, 0, 1],  # Exp 6
    [2, 0, 2, 1],  # Exp 7
    [2, 1, 0, 2],  # Exp 8
    [2, 2, 1, 0],  # Exp 9
])

# =====================================================================
# FUNÇÕES DE COLETA DE DADOS
# =====================================================================

def baixar_cotacoes(tickers, inicio, fim):
    """Baixa cotações de fechamento ajustadas via yfinance."""
    try:
        import yfinance as yf
    except ImportError:
        print("ERRO: Instale o yfinance com: pip install yfinance")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  BAIXANDO COTAÇÕES DO IBOVESPA")
    print(f"  Periodo: {inicio} a {fim}")
    print(f"  Tickers: {len(tickers)} ações")
    print(f"{'='*60}\n")

    dados = yf.download(tickers, start=inicio, end=fim, auto_adjust=True)

    # Extrair preços de fechamento
    if isinstance(dados.columns, pd.MultiIndex):
        precos = dados['Close']
    else:
        precos = dados

    # Remover ações com muitos dados faltantes (> 20%)
    limite = len(precos) * 0.20
    precos = precos.dropna(axis=1, thresh=int(len(precos) - limite))

    # Preencher dados faltantes restantes (forward fill)
    precos = precos.ffill().bfill()

    print(f"\nAcoes com dados validos: {precos.shape[1]}")
    print(f"Periodo efetivo: {precos.index[0].date()} a {precos.index[-1].date()}")
    print(f"Dias de negociacao: {len(precos)}")

    return precos


def calcular_retornos(precos):
    """Calcula retornos logarítmicos diários."""
    retornos = np.log(precos / precos.shift(1)).dropna()
    return retornos

# =====================================================================
# FUNÇÕES DE OTIMIZAÇÃO DE PORTFÓLIO
# =====================================================================

def portfolio_igual(retornos_janela, Rf_diario, lam):
    """Portfólio igualmente ponderado (1/N)."""
    n = retornos_janela.shape[1]
    pesos = np.ones(n) / n
    return pesos


def portfolio_min_variancia(retornos_janela, Rf_diario, lam):
    """Portfólio de mínima variância (sem venda a descoberto)."""
    from scipy.optimize import minimize

    n = retornos_janela.shape[1]
    mu = retornos_janela.mean().values * 252     # anualizado
    cov = retornos_janela.cov().values * 252     # anualizado

    def objetivo(w):
        risco = np.sqrt(w @ cov @ w)
        return risco

    # Restricoes e limites
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = [(0.0, 1.0)] * n
    w0 = np.ones(n) / n

    result = minimize(objetivo, w0, method='SLSQP',
                      bounds=bounds, constraints=constraints,
                      options={'maxiter': 1000, 'ftol': 1e-12})

    if result.success:
        return result.x
    else:
        return np.ones(n) / n  # fallback


def portfolio_max_sharpe(retornos_janela, Rf_diario, lam):
    """Portfólio que maximiza o Índice Sharpe (sem venda a descoberto)."""
    from scipy.optimize import minimize

    n = retornos_janela.shape[1]
    mu = retornos_janela.mean().values * 252     # anualizado
    cov = retornos_janela.cov().values * 252     # anualizado
    Rf_anual = (1 + Rf_diario) ** 252 - 1

    def neg_sharpe(w):
        ret_p = w @ mu
        risco_p = np.sqrt(w @ cov @ w)
        if risco_p < 1e-10:
            return 1e10
        sharpe = (ret_p - Rf_anual) / risco_p
        return -sharpe

    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = [(0.0, 1.0)] * n
    w0 = np.ones(n) / n

    result = minimize(neg_sharpe, w0, method='SLSQP',
                      bounds=bounds, constraints=constraints,
                      options={'maxiter': 1000, 'ftol': 1e-12})

    if result.success:
        return result.x
    else:
        return np.ones(n) / n  # fallback


def portfolio_mean_variance(retornos_janela, Rf_diario, lam):
    """
    Portfólio com função-objetivo combinada:
       min z = lambda * [risco^2] - (1-lambda) * [retorno]
    Conforme Chen et al. (2020).
    """
    from scipy.optimize import minimize

    n = retornos_janela.shape[1]
    mu = retornos_janela.mean().values * 252
    cov = retornos_janela.cov().values * 252

    def objetivo(w):
        ret_p = w @ mu
        var_p = w @ cov @ w
        return lam * var_p - (1 - lam) * ret_p

    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = [(0.0, 1.0)] * n
    w0 = np.ones(n) / n

    result = minimize(objetivo, w0, method='SLSQP',
                      bounds=bounds, constraints=constraints,
                      options={'maxiter': 1000, 'ftol': 1e-12})

    if result.success:
        return result.x
    else:
        return np.ones(n) / n


METODOS = {
    'igual':      portfolio_igual,
    'min_var':    portfolio_min_variancia,
    'max_sharpe': portfolio_max_sharpe,
}


def calcular_desempenho_portfolio(pesos, retornos_validacao, Rf_diario):
    """Calcula retorno, risco e Sharpe do portfólio na janela de validação."""
    # Retornos diários do portfólio
    ret_portfolio = (retornos_validacao.values @ pesos)

    # Retorno anualizado
    ret_medio_diario = np.mean(ret_portfolio)
    retorno_anual = ret_medio_diario * 252

    # Risco anualizado
    risco_diario = np.std(ret_portfolio, ddof=1)
    risco_anual = risco_diario * np.sqrt(252)

    # Índice Sharpe
    Rf_anual = (1 + Rf_diario) ** 252 - 1
    if risco_anual > 1e-10:
        sharpe = (retorno_anual - Rf_anual) / risco_anual
    else:
        sharpe = 0.0

    return retorno_anual, risco_anual, sharpe

# =====================================================================
# EXECUÇÃO DO EXPERIMENTO TAGUCHI
# =====================================================================

def executar_experimento(retornos, arranjo, fatores, n_rep, seed):
    """Executa todas as configurações do arranjo L9 com replicações."""
    np.random.seed(seed)
    acoes_disponiveis = list(retornos.columns)
    n_acoes_total = len(acoes_disponiveis)

    # Janela de validação: últimos 126 dias úteis (~6 meses)
    JANELA_VALIDACAO = 126

    resultados = []

    print(f"\n{'='*60}")
    print(f"  EXECUTANDO EXPERIMENTO TAGUCHI L9(3^4)")
    print(f"  Replicacoes por configuracao: {n_rep}")
    print(f"  Total de simulacoes: {arranjo.shape[0] * n_rep}")
    print(f"{'='*60}\n")

    for exp_idx in range(arranjo.shape[0]):
        config = arranjo[exp_idx]

        n_ativos    = fatores['A']['niveis'][config[0]]
        janela_hist = fatores['B']['niveis'][config[1]]
        lam         = fatores['C']['niveis'][config[2]]
        metodo_nome = fatores['D']['niveis'][config[3]]

        metodo_label = {'igual': 'Igual (1/N)', 'min_var': 'Min. Var.', 'max_sharpe': 'Max. Sharpe'}

        sharpes_rep = []
        retornos_rep = []
        riscos_rep = []

        for rep in range(n_rep):
            try:
                # Selecionar ativos aleatorios
                if n_ativos > n_acoes_total:
                    ativos_sel = acoes_disponiveis
                else:
                    ativos_sel = list(np.random.choice(acoes_disponiveis, n_ativos, replace=False))

                ret_sel = retornos[ativos_sel].copy()

                # Dividir em janela de estimação e validação
                total_dias = len(ret_sel)
                fim_estimacao = total_dias - JANELA_VALIDACAO
                inicio_estimacao = max(0, fim_estimacao - janela_hist)

                if fim_estimacao - inicio_estimacao < 60:  # minimo 60 dias
                    continue

                ret_estimacao = ret_sel.iloc[inicio_estimacao:fim_estimacao]
                ret_validacao = ret_sel.iloc[fim_estimacao:]

                # Remover ações com dados faltantes na janela
                ret_estimacao = ret_estimacao.dropna(axis=1)
                ret_validacao = ret_validacao[ret_estimacao.columns]

                if ret_estimacao.shape[1] < 2:
                    continue

                # Calcular pesos
                func_metodo = METODOS[metodo_nome]
                pesos = func_metodo(ret_estimacao, SELIC_DIARIA, lam)

                # Avaliar na janela de validação
                ret_p, risco_p, sharpe_p = calcular_desempenho_portfolio(
                    pesos, ret_validacao, SELIC_DIARIA
                )

                if np.isfinite(sharpe_p):
                    sharpes_rep.append(sharpe_p)
                    retornos_rep.append(ret_p)
                    riscos_rep.append(risco_p)

            except Exception as e:
                continue

        # Média das replicações
        if len(sharpes_rep) > 0:
            sharpe_medio = np.mean(sharpes_rep)
            retorno_medio = np.mean(retornos_rep)
            risco_medio = np.mean(riscos_rep)
            sharpe_std = np.std(sharpes_rep, ddof=1)
            n_validas = len(sharpes_rep)
        else:
            sharpe_medio = 0.0
            retorno_medio = 0.0
            risco_medio = 0.0
            sharpe_std = 0.0
            n_validas = 0

        resultado = {
            'Exp': exp_idx + 1,
            'A_nivel': config[0] + 1,
            'B_nivel': config[1] + 1,
            'C_nivel': config[2] + 1,
            'D_nivel': config[3] + 1,
            'A_valor': n_ativos,
            'B_valor': f"{janela_hist // 21}m",  # converter dias uteis para meses aprox.
            'C_valor': lam,
            'D_valor': metodo_label[metodo_nome],
            'Retorno_medio': round(retorno_medio, 6),
            'Risco_medio': round(risco_medio, 6),
            'Sharpe_medio': round(sharpe_medio, 4),
            'Sharpe_std': round(sharpe_std, 4),
            'N_replicacoes_validas': n_validas,
        }

        resultados.append(resultado)

        print(f"  Exp {exp_idx+1}/9: A={n_ativos:2d} ativos | "
              f"B={janela_hist//21:2d}m | C={lam:.2f} | D={metodo_label[metodo_nome]:12s} "
              f"=> Sharpe = {sharpe_medio:+.4f}  (n={n_validas})")

    return pd.DataFrame(resultados)

# =====================================================================
# ANÁLISES TAGUCHI
# =====================================================================

def analise_medias(df_resultados, fatores):
    """Calcula a Tabela de Resposta para Médias (ANOM)."""
    tabela = {}
    for fator_key in ['A', 'B', 'C', 'D']:
        col_nivel = f'{fator_key}_nivel'
        medias_por_nivel = []
        for nivel in [1, 2, 3]:
            mask = df_resultados[col_nivel] == nivel
            media = df_resultados.loc[mask, 'Sharpe_medio'].mean()
            medias_por_nivel.append(round(media, 4))
        tabela[fator_key] = medias_por_nivel

    # Calcular Delta e Rank
    deltas = {}
    for fator_key, medias in tabela.items():
        deltas[fator_key] = round(max(medias) - min(medias), 4)

    # Ranking (maior delta = rank 1)
    sorted_fatores = sorted(deltas, key=deltas.get, reverse=True)
    ranks = {f: sorted_fatores.index(f) + 1 for f in deltas}

    # Montar DataFrame
    df_resp = pd.DataFrame({
        'Nivel': [1, 2, 3, 'Delta', 'Rank'],
    })
    for fk in ['A', 'B', 'C', 'D']:
        nome = fatores[fk]['nome']
        valores = tabela[fk] + [deltas[fk], ranks[fk]]
        df_resp[nome] = valores

    return df_resp, tabela, deltas, ranks


def analise_sn(df_resultados, fatores):
    """Calcula a Tabela de Resposta para Razão S/N (Maior-e-melhor)."""
    # S/N para cada experimento (Larger-is-better)
    # S/N = -10 * log10( (1/n) * sum(1/yi^2) )
    # Como usamos a média, simplificamos para:
    # S/N = -10 * log10( 1/y_medio^2 ) = 20 * log10(|y_medio|)  (para y > 0)
    # Mas para ser mais preciso com as replicações, usamos a fórmula completa

    # Para cada experimento, calculamos S/N a partir do Sharpe médio
    sn_values = []
    for _, row in df_resultados.iterrows():
        y = row['Sharpe_medio']
        if y > 0:
            sn = -10 * np.log10(1.0 / (y ** 2))
        elif y < 0:
            sn = -10 * np.log10(1.0 / (y ** 2))  # S/N pode ser negativo
        else:
            sn = -50  # valor muito baixo para Sharpe = 0
        sn_values.append(round(sn, 4))

    df_resultados = df_resultados.copy()
    df_resultados['SN'] = sn_values

    tabela_sn = {}
    for fator_key in ['A', 'B', 'C', 'D']:
        col_nivel = f'{fator_key}_nivel'
        medias_sn = []
        for nivel in [1, 2, 3]:
            mask = df_resultados[col_nivel] == nivel
            media = df_resultados.loc[mask, 'SN'].mean()
            medias_sn.append(round(media, 4))
        tabela_sn[fator_key] = medias_sn

    deltas_sn = {}
    for fk, medias in tabela_sn.items():
        deltas_sn[fk] = round(max(medias) - min(medias), 4)

    sorted_f = sorted(deltas_sn, key=deltas_sn.get, reverse=True)
    ranks_sn = {f: sorted_f.index(f) + 1 for f in deltas_sn}

    df_sn = pd.DataFrame({'Nivel': [1, 2, 3, 'Delta', 'Rank']})
    for fk in ['A', 'B', 'C', 'D']:
        nome = fatores[fk]['nome']
        valores = tabela_sn[fk] + [deltas_sn[fk], ranks_sn[fk]]
        df_sn[nome] = valores

    return df_sn, df_resultados, tabela_sn, deltas_sn, ranks_sn


def analise_anova(df_resultados, fatores):
    """Calcula a tabela ANOVA para contribuição percentual de cada fator."""
    y = df_resultados['Sharpe_medio'].values
    media_geral = np.mean(y)
    n_exp = len(y)

    # SS Total
    ss_total = np.sum((y - media_geral) ** 2)

    anova_rows = []
    ss_fatores = {}

    for fator_key in ['A', 'B', 'C', 'D']:
        col_nivel = f'{fator_key}_nivel'
        niveis = df_resultados[col_nivel].values
        ss_fator = 0
        for nivel in [1, 2, 3]:
            mask = niveis == nivel
            n_nivel = np.sum(mask)
            media_nivel = np.mean(y[mask])
            ss_fator += n_nivel * (media_nivel - media_geral) ** 2

        ss_fatores[fator_key] = ss_fator
        df_fator = 2  # 3 niveis - 1
        ms_fator = ss_fator / df_fator if df_fator > 0 else 0
        contrib_pct = (ss_fator / ss_total * 100) if ss_total > 0 else 0

        anova_rows.append({
            'Fonte': fatores[fator_key]['nome'],
            'GL': df_fator,
            'SS': round(ss_fator, 6),
            'MS': round(ms_fator, 6),
            '% Contribuicao': round(contrib_pct, 2),
        })

    # Erro (residual)
    ss_erro = ss_total - sum(ss_fatores.values())
    df_erro = n_exp - 1 - sum(2 for _ in ['A', 'B', 'C', 'D'])  # 9 - 1 - 8 = 0 para L9
    # No L9 com 4 fatores, GL_erro = 0 (saturado)
    # Reportamos mesmo assim para transparência
    ms_erro = ss_erro / df_erro if df_erro > 0 else 0
    contrib_erro = (ss_erro / ss_total * 100) if ss_total > 0 else 0

    anova_rows.append({
        'Fonte': 'Erro (Residual)',
        'GL': max(df_erro, 0),
        'SS': round(max(ss_erro, 0), 6),
        'MS': round(ms_erro, 6),
        '% Contribuicao': round(max(contrib_erro, 0), 2),
    })

    anova_rows.append({
        'Fonte': 'Total',
        'GL': n_exp - 1,
        'SS': round(ss_total, 6),
        'MS': '',
        '% Contribuicao': 100.00,
    })

    df_anova = pd.DataFrame(anova_rows)
    return df_anova


def determinar_combinacao_otima(tabela_medias, fatores):
    """Identifica a combinação ótima de níveis."""
    otima = {}
    for fk in ['A', 'B', 'C', 'D']:
        medias = tabela_medias[fk]
        melhor_nivel = np.argmax(medias)  # indice 0, 1, 2
        otima[fk] = {
            'nivel': melhor_nivel + 1,
            'valor': fatores[fk]['niveis'][melhor_nivel],
            'media': medias[melhor_nivel],
        }
    return otima

# =====================================================================
# GRÁFICOS
# =====================================================================

def gerar_graficos(tabela_medias, tabela_sn, fatores, output_dir):
    """Gera gráficos de efeitos principais."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # Gráfico de Efeitos Principais (Médias)
        fig, axes = plt.subplots(1, 4, figsize=(14, 4), sharey=True)
        fig.suptitle('Gráfico de Efeitos Principais para Médias (Índice Sharpe)',
                     fontsize=13, fontweight='bold')

        for i, fk in enumerate(['A', 'B', 'C', 'D']):
            ax = axes[i]
            niveis = [1, 2, 3]
            medias = tabela_medias[fk]
            ax.plot(niveis, medias, 'bo-', linewidth=2, markersize=8)
            ax.set_xlabel(f'Nível', fontsize=10)
            ax.set_title(fatores[fk]['nome'], fontsize=11, fontweight='bold')
            ax.set_xticks(niveis)
            labels_niveis = [str(fatores[fk]['niveis'][j]) for j in range(3)]
            ax.set_xticklabels(labels_niveis, fontsize=8)
            ax.grid(True, alpha=0.3)

            # Destacar o nível ótimo
            melhor = np.argmax(medias)
            ax.plot(niveis[melhor], medias[melhor], 'r*', markersize=15)

        axes[0].set_ylabel('Média do Indice Sharpe', fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'grafico_efeitos_principais.png'), dpi=150)
        plt.close()
        print("  -> grafico_efeitos_principais.png")

        # Gráfico S/N
        fig, axes = plt.subplots(1, 4, figsize=(14, 4), sharey=True)
        fig.suptitle('Gráfico de Efeitos Principais para Razão S/N (dB)',
                     fontsize=13, fontweight='bold')

        for i, fk in enumerate(['A', 'B', 'C', 'D']):
            ax = axes[i]
            niveis = [1, 2, 3]
            sn_vals = tabela_sn[fk]
            ax.plot(niveis, sn_vals, 'gs-', linewidth=2, markersize=8)
            ax.set_xlabel(f'Nivel', fontsize=10)
            ax.set_title(fatores[fk]['nome'], fontsize=11, fontweight='bold')
            ax.set_xticks(niveis)
            labels_niveis = [str(fatores[fk]['niveis'][j]) for j in range(3)]
            ax.set_xticklabels(labels_niveis, fontsize=8)
            ax.grid(True, alpha=0.3)

            melhor = np.argmax(sn_vals)
            ax.plot(niveis[melhor], sn_vals[melhor], 'r*', markersize=15)

        axes[0].set_ylabel('Razao S/N (dB)', fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'grafico_SN.png'), dpi=150)
        plt.close()
        print("  -> grafico_SN.png")

    except ImportError:
        print("  AVISO: matplotlib não instalado. Gráficos não gerados.")
        print("         Instale com: pip install matplotlib")

# =====================================================================
# FUNÇÃO PRINCIPAL
# =====================================================================

def main():
    print("\n" + "=" * 60)
    print("  ME623 - EXPERIMENTO TAGUCHI L9 PARA PORTFÓLIO")
    print("  Otimização de Portfólio via Método Taguchi")
    print(f"  Data de execução: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    output_dir = os.path.dirname(os.path.abspath(__file__))

    # ── 1. COLETA DE DADOS ──
    precos = baixar_cotacoes(TICKERS_IBOVESPA, DATA_INICIO, DATA_FIM)
    retornos = calcular_retornos(precos)

    # Salvar dados
    precos.to_csv(os.path.join(output_dir, 'dados_cotacoes.csv'))
    retornos.to_csv(os.path.join(output_dir, 'dados_retornos.csv'))
    print(f"\n  Dados salvos: dados_cotacoes.csv, dados_retornos.csv")

    # ── 2. EXECUTAR EXPERIMENTO L9 ──
    df_res = executar_experimento(retornos, ARRANJO_L9, FATORES, N_REPLICACOES, SEED)
    df_res.to_csv(os.path.join(output_dir, 'resultados_L9.csv'), index=False)

    # ── 3. ANÁLISE DE MÉDIAS (ANOM) ──
    print(f"\n{'='*60}")
    print(f"  ANÁLISE DE MÉDIAS (ANOM)")
    print(f"{'='*60}")
    df_resp, tabela_medias, deltas, ranks = analise_medias(df_res, FATORES)
    print(f"\n{df_resp.to_string(index=False)}")
    df_resp.to_csv(os.path.join(output_dir, 'tabela_resposta_medias.csv'), index=False)

    # ── 4. ANÁLISE S/N ──
    print(f"\n{'='*60}")
    print(f"  ANÁLISE RAZÃO SINAL-RUÍDO (S/N)")
    print(f"{'='*60}")
    df_sn, df_res_sn, tabela_sn, deltas_sn, ranks_sn = analise_sn(df_res, FATORES)
    print(f"\n{df_sn.to_string(index=False)}")
    df_sn.to_csv(os.path.join(output_dir, 'tabela_resposta_SN.csv'), index=False)

    # ── 5. ANOVA ──
    print(f"\n{'='*60}")
    print(f"  ANÁLISE DE VARIÂNCIA (ANOVA)")
    print(f"{'='*60}")
    df_anova = analise_anova(df_res, FATORES)
    print(f"\n{df_anova.to_string(index=False)}")
    df_anova.to_csv(os.path.join(output_dir, 'tabela_anova.csv'), index=False)

    # ── 6. COMBINACAO OTIMA ──
    print(f"\n{'='*60}")
    print(f"  COMBINAÇÃO ÓTIMA")
    print(f"{'='*60}")
    otima = determinar_combinacao_otima(tabela_medias, FATORES)
    for fk in ['A', 'B', 'C', 'D']:
        info = otima[fk]
        print(f"  {FATORES[fk]['nome']:20s}: Nivel {info['nivel']} "
              f"(valor = {info['valor']}) -> Media Sharpe = {info['media']:.4f}")

    # ── 7. GRÁFICOS ──
    print(f"\n{'='*60}")
    print(f"  GERANDO GRÁFICOS")
    print(f"{'='*60}")
    gerar_graficos(tabela_medias, tabela_sn, FATORES, output_dir)

    # ── 8. TABELA RESUMO COMPLETA ──
    print(f"\n{'='*60}")
    print(f"  TABELA DE RESULTADOS DO EXPERIMENTO L9")
    print(f"{'='*60}")
    cols_show = ['Exp', 'A_valor', 'B_valor', 'C_valor', 'D_valor',
                 'Retorno_medio', 'Risco_medio', 'Sharpe_medio', 'Sharpe_std']
    print(f"\n{df_res[cols_show].to_string(index=False)}")

    # ── 9. RESUMO EM ARQUIVO TEXTO ──
    with open(os.path.join(output_dir, 'resumo_completo.txt'), 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("  ME623 - RESULTADOS DO EXPERIMENTO TAGUCHI L9\n")
        f.write("  Otimização de Portfólio de Investimentos\n")
        f.write(f"  Data: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"  Ações disponíveis: {precos.shape[1]}\n")
        f.write(f"  Período: {precos.index[0].date()} a {precos.index[-1].date()}\n")
        f.write(f"  Taxa Selic (Rf): {SELIC_ANUAL*100:.2f}% a.a.\n")
        f.write(f"  Replicações: {N_REPLICACOES} por configuracao\n")
        f.write("=" * 70 + "\n\n")

        f.write("TABELA DE RESULTADOS L9:\n")
        f.write(df_res[cols_show].to_string(index=False))
        f.write("\n\n")

        f.write("TABELA DE RESPOSTA PARA MÉDIAS (ANOM):\n")
        f.write(df_resp.to_string(index=False))
        f.write("\n\n")

        f.write("TABELA DE RESPOSTA PARA RAZÃO S/N:\n")
        f.write(df_sn.to_string(index=False))
        f.write("\n\n")

        f.write("TABELA ANOVA:\n")
        f.write(df_anova.to_string(index=False))
        f.write("\n\n")

        f.write("COMBINAÇÃO ÓTIMA:\n")
        for fk in ['A', 'B', 'C', 'D']:
            info = otima[fk]
            f.write(f"  {FATORES[fk]['nome']:20s}: Nivel {info['nivel']} "
                    f"(valor = {info['valor']}) -> Media Sharpe = {info['media']:.4f}\n")
        f.write("\n")

    print(f"\n  -> resumo_completo.txt")

    print(f"\n{'='*60}")
    print(f"  EXECUÇÃO CONCLUÍDA COM SUCESSO!")
    print(f"  Arquivos gerados no diretório:")
    print(f"  {output_dir}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
